"""Persist independent complete physical waveforms, FFTs, noise and cancellation evidence."""

from pathlib import Path
from typing import Any

import torch

from ..channel.affine import affine_waveform
from ..channel.effective_matrix import effective_matrix
from ..channel.noise import complex_awgn, sigma2_from_esn0
from ..channel.validity import validate_cp_support
from ..receiver.fft_receiver import fft_receive
from ..receiver.preprocessing import preprocess
from ..utils.device import resolve_runtime
from ..utils.random import generator
from ..waveform.frame import pad_recording
from .manifest import file_hash, load_manifest, save_json, save_tensors
from .records import record_frame, record_paths


@torch.no_grad()
def audit_dataset(
    manifest_path: str | Path, output: str | Path, *, waveform_frames: int | None = None
) -> dict[str, Any]:
    from ..cli import _provenance

    manifest, config, records = load_manifest(manifest_path)
    count = (
        config.values["data"]["audit_waveform_frames"]
        if waveform_frames is None
        else waveform_frames
    )
    if type(count) is not int or count <= 0:
        raise ValueError("waveform_frames must be a positive integer")
    unique = {r["frame_id"]: r for r in reversed(records)}
    # Prefer a nondegenerate affine case, then cover other available scenarios.
    candidates = sorted(
        unique.values(),
        key=lambda r: (not r["scenario"].startswith("affine"), r["scenario"], r["frame_id"]),
    )
    selected: list[dict[str, Any]] = []
    scenarios: set[str] = set()
    for record in candidates:
        if record["scenario"] not in scenarios:
            selected.append(record)
            scenarios.add(record["scenario"])
    selected.extend(r for r in candidates if r["frame_id"] not in {s["frame_id"] for s in selected})
    if count > len(selected):
        raise ValueError("requested more independent audit frames than dataset contains")
    output = Path(output)
    if output.exists():
        raise ValueError("audit output already exists; choose a fresh path")
    runtime = resolve_runtime("cpu", "complex128", config.values["runtime"]["cpu_threads"])
    # Audit reference is double precision independently of the dataset consumer dtype.
    output.mkdir(parents=True)
    reports = []
    for number, record in enumerate(selected[:count]):
        frame = record_frame(record, config, runtime)
        paths = record_paths(record)
        support = validate_cp_support(paths, frame.layout, config)
        arrival = record["arrival_offset_samples"]
        times = (
            torch.arange(frame.layout.frame_samples + arrival, dtype=torch.float64)
            / frame.layout.sample_rate_hz
        )
        received = affine_waveform(frame, paths, times, config, arrival_offset_samples=arrival)[0]
        windows, errors = [], []
        for block, (start, end) in enumerate(frame.layout.useful):
            wave = received[start + arrival : end + arrival]
            h = effective_matrix(paths, frame.layout, config, block)
            grid = frame.grid[0, block]
            observed = fft_receive(wave, frame.allocation)
            error = float(
                torch.linalg.vector_norm(observed - h @ grid)
                / torch.linalg.vector_norm(observed).clamp_min(torch.finfo(torch.float64).tiny)
            )
            if not error < 1e-9:
                raise ValueError(
                    f"waveform/H audit failed frame={record['frame_id']} block={block}: {error}"
                )
            sigma2 = sigma2_from_esn0(float(record["esn0_db"][block]))
            noise = complex_awgn(
                (config.values["waveform"]["n_fft_wave"],),
                sigma2,
                gen_r=generator(int(record["noise_seed_real"][block])),
                gen_i=generator(int(record["noise_seed_imag"][block])),
            )
            idx, pi = frame.allocation.data_grid_index, frame.allocation.pilot_grid_index
            noisy = fft_receive(wave + noise, frame.allocation)
            processed = preprocess(h, noisy, grid[pi], frame.allocation, sigma2)
            pilot = h[idx][:, pi] @ grid[pi]
            windows.append(
                {
                    "block": block,
                    "recording_span": [start + arrival, end + arrival],
                    "h_grid": h,
                    "iq_waveform": wave.clone(),
                    "noise": noise,
                    "observed_grid": observed,
                    "noisy_grid": noisy,
                    "pilot_contribution": pilot,
                    "H": processed.H,
                    "y": processed.y,
                    "sigma2": sigma2,
                    "snr_rx_db": float(
                        10 * torch.log10(processed.H.abs().square().sum() / (400 * sigma2))
                    ),
                }
            )
            errors.append(error)
        artifact = output / f"frame-{number:03d}.pt"
        save_tensors(
            artifact,
            {
                "record": record,
                "grid": frame.grid[0],
                "cp_endpoints_s": support,
                "transmit_real": frame.real[0],
                "transmit_analytic": frame.analytic[0],
                "transmit_recording_real": pad_recording(frame.real[0], arrival),
                "received_iq": received,
                "windows": windows,
            },
        )
        reports.append(
            {
                "frame_id": record["frame_id"],
                "scenario": record["scenario"],
                "artifact": artifact.name,
                "sha256": file_hash(artifact),
                "relative_errors_complex128": errors,
            }
        )
    report = {
        "schema_version": 1,
        "status": "passed",
        "source_manifest_sha256": file_hash(Path(manifest_path)),
        "resolved_config": config.values,
        "allocation": manifest["allocation"],
        "environment": _provenance(runtime),
        "frames": reports,
        "scope": (
            "independent complete waveforms; ideal IQ/oracle timing/perfect CSI; development audit"
        ),
        "training_main_sweeps": "not executed",
    }
    save_json(output / "audit.json", report)
    return {
        "status": "passed",
        "frames": count,
        "output": str(output),
        "max_relative_error_complex128": max(
            e for r in reports for e in r["relative_errors_complex128"]
        ),
    }
