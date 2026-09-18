"""Two declared timing boundaries, synchronized only on explicitly selected CUDA."""

import ctypes
import os
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch

from ..algorithms.pg_vamp.model import PGVAMPDetector
from ..algorithms.prepared import Detector, PreparedDetector, timed_detect

MODES = ("per_observation_cold_H", "same_H_amortized")


def process_peak_bytes() -> int:
    """Process lifetime high-water RSS; never attribute it to one algorithm."""
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "working",
                    "paged_peak",
                    "paged",
                    "nonpaged_peak",
                    "nonpaged",
                    "pagefile",
                    "pagefile_peak",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.peak)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation: Callable[[], Any], device: torch.device) -> tuple[Any, float]:
    synchronize(device)
    start = perf_counter()
    value = operation()
    synchronize(device)
    return value, (perf_counter() - start) * 1000


def measure_detector(
    model: Detector,
    H: torch.Tensor,
    observations: torch.Tensor,
    sigma2: torch.Tensor,
    *,
    mode: str,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    """H[1,N,N], observations[K,N] independent noise draws for this exact H.

    Input generation and diagnostics are outside measurement. Batch observations
    are distinct draws, never different physical block matrices relabeled same-H.
    """
    if mode not in MODES or type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid timing mode/batch")
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 2:
        raise ValueError("timing requires warmup >= 0 and repeats >= 2")
    if H.shape[0] != 1 or observations.shape[0] < batch_size or sigma2.shape != (1,):
        raise ValueError("timing needs one H and enough observations for batch")
    model.eval()
    rows = []
    for batch in sorted({1, batch_size}):
        h = H.expand(batch, -1, -1).contiguous()
        noise = sigma2.expand(batch).contiguous()
        pool = observations.shape[0] // batch
        baseline_peak = process_peak_bytes()
        with torch.inference_mode():
            if H.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(H.device)
            prepared = None
            prepare_ms = 0.0
            if mode == "same_H_amortized":
                prepared, prepare_ms = _measure(lambda: PreparedDetector(model, h, noise), H.device)

            def operation(index: int) -> Any:
                y = observations[(index % pool) * batch : (index % pool + 1) * batch]
                return (
                    prepared.detect(h, y, noise)
                    if prepared is not None
                    else timed_detect(model, h, y, noise)
                )

            for i in range(warmup):
                operation(i)
            elapsed = []
            for i in range(repeats):
                result, duration = _measure(lambda: operation(i), H.device)
                if not bool(torch.isfinite(result.x_soft).all()):
                    raise FloatingPointError("nonfinite benchmark output")
                elapsed.append(duration)
        mean = float(np.mean(elapsed))
        n = H.shape[-1]
        # Explicit conservative matrix-slot estimate, not a measured allocator peak.
        slots = 12 + 5 * model.depth if isinstance(model, PGVAMPDetector) else 8
        rows.append(
            {
                "timing_mode": mode,
                "batch_size": batch,
                "warmup": warmup,
                "repeats": repeats,
                "device": str(H.device),
                "dtype": str(H.dtype),
                "cpu_threads": torch.get_num_threads(),
                "pid": os.getpid(),
                "mean_latency_ms": mean,
                "median_latency_ms": float(np.median(elapsed)),
                "p95_latency_ms": float(np.quantile(elapsed, 0.95)),
                "latency_kind": "single_block_online" if batch == 1 else "batch_latency",
                "amortized_per_block_ms": mean / batch,
                "blocks_per_second": batch * 1000 / mean,
                "data_bits_per_second": 2 * n * batch * 1000 / mean,
                "prepare_ms": prepare_ms,
                "observation_count": repeats * batch,
                "unique_noise_observations": min(repeats, pool) * batch,
                "total_amortized_per_block_ms": (prepare_ms + sum(elapsed)) / (repeats * batch),
                "cpu_process_peak_bytes": process_peak_bytes(),
                "cpu_process_peak_before_bytes": baseline_peak,
                "cpu_memory_scope": "process_lifetime_high_water_not_algorithm_attributable",
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(H.device)
                if H.device.type == "cuda"
                else None,
                "matrix_work_bytes_estimate": slots * batch * n * n * H.element_size(),
                "matrix_estimate_formula": f"{slots} * batch * N^2 * complex_element_bytes; "
                "conservative slots, not measured",
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "status": "complete",
                "includes": "validation,factorization_or_prepared_validation,"
                "all_layers,hard_decision",
                "excludes": "data_generation,checkpoint_load,IO,plotting,diagnostic_summary_copies",
                "raw_latency_ms": elapsed,
            }
        )
    return rows
