"""Explicit CPU/CUDA selection, with no CUDA access on the CPU path."""

import os
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Runtime:
    """Resolved runtime; real dtype is paired with the complex tensor dtype."""

    device: torch.device
    complex_dtype: torch.dtype
    real_dtype: torch.dtype
    cpu_threads: int


def resolve_runtime(
    device: str = "cpu",
    dtype: str = "complex128",
    cpu_threads: int = 4,
    deterministic: bool = True,
) -> Runtime:
    """Validate explicit settings and initialize threads; CPU never queries CUDA."""
    if dtype not in ("complex128", "complex64"):
        raise ValueError("runtime.dtype must be complex128 or complex64")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ValueError("runtime.device must be cpu or cuda[:index]")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("runtime.cpu_threads must be a positive integer")
    if type(deterministic) is not bool:
        raise ValueError("runtime.deterministic must be boolean")
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA requested ({device}) but unavailable; no CPU fallback")
        index = selected.index if selected.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        selected = torch.device("cuda", index)
    threads = min(cpu_threads, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    return Runtime(
        selected,
        getattr(torch, dtype),
        torch.float64 if dtype == "complex128" else torch.float32,
        threads,
    )
