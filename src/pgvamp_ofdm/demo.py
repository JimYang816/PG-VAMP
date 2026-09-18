"""One complete physical frame, with independent waveform/FFT evidence (§21.6)."""

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .channel.affine import affine_waveform
from .channel.effective_matrix import effective_matrix
from .channel.noise import complex_awgn, sigma2_from_esn0
from .channel.parameters import sample_paths
from .channel.validity import validate_cp_support
from .cli import _provenance
from .config import Config
from .modulation.allocation import _physical_settings
from .modulation.qpsk import classes_to_symbols
from .receiver.fft_receiver import fft_receive
from .receiver.preprocessing import preprocess
from .receiver.synchronization import normalized_correlation
from .utils.device import resolve_runtime
from .waveform.frame import build_frame


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _figures(artifact: Path, output: Path) -> None:
    """Render only persisted tensors; no new simulation or inferred performance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    fs = saved["layout"]["sample_rate_hz"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    for ax, key, title in zip(
        axes[:2],
        ("tx_real", "rx_real_noiseless"),
        ("Complete transmitted real passband", "Complete received real passband (noiseless)"),
    ):
        values = saved[key][0].numpy()
        ax.plot(torch.arange(len(values)).numpy() / fs, values, linewidth=0.4)
        ax.set(xlabel="Recording time (s)", ylabel="Amplitude", title=title)
    scores = saved["sync"]["scores"][0].numpy()
    axes[2].plot(torch.arange(len(scores)).numpy() / fs, scores)
    axes[2].axhline(saved["sync"]["threshold"], color="red", linestyle="--", label="Threshold")
    axes[2].axvline(
        saved["sync"]["peak_start"].item() / fs, color="black", linestyle=":", label="Selected peak"
    )
    axes[2].set(
        xlabel="Template start time (s)",
        ylabel="Normalized score",
        title="LFM demonstration; not receiver timing",
    )
    axes[2].legend()
    fig.savefig(output / "waveform_and_sync.png", dpi=150)
    plt.close(fig)
    block = saved["windows"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    magnitude = block["H"].abs().clamp_min(1e-12)
    shown = axes[0].imshow((20 * magnitude.log10()).numpy(), origin="lower", aspect="auto")
    axes[0].set(title="Block 0: full 400 x 400 H (dB)", xlabel="Data column", ylabel="Data row")
    fig.colorbar(shown, ax=axes[0])
    axes[1].scatter(
        block["y"].real.numpy(), block["y"].imag.numpy(), s=5, label="y after pilot cancellation"
    )
    axes[1].scatter(
        block["x"].real.numpy(),
        block["x"].imag.numpy(),
        marker="x",
        label="Reference QPSK (audit only)",
    )
    axes[1].set(title="Received observation; no equalizer", xlabel="Real", ylabel="Imaginary")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "channel_and_constellation.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def demo_frame(config: Config, output: Path, *, argv: list[str] | None = None) -> dict[str, Any]:
    """Save one 8-block 512/400/8192/CP2048 frame and safe tensor audit bundle.

    Continuous physical references use complex128. H/y/sigma2 examples use the
    requested runtime precision. Labels are saved only for independent audits.
    The LFM peak from noiseless real RF never changes oracle FFT windows.
    Existing output paths are rejected, including empty directories.
    """
    w, f = _physical_settings(config)
    if (w["n_grid"], w["n_data"], w["n_fft_wave"], w["cp_samples"], f["n_ofdm_symbols"]) != (
        512,
        400,
        8192,
        2048,
        8,
    ):
        raise ValueError("demo-frame requires full 512/400/8192/CP2048/eight-block dimensions")
    if config.values["receiver"]["sync_mode"] != "oracle_timing":
        raise ValueError(
            "demo-frame receive examples require oracle_timing; LFM is a separate demonstration"
        )
    if output.exists():
        raise ValueError("demo-frame output must be a fresh directory; preserve existing evidence")
    rt = config.values["runtime"]
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    rt["device"], rt["cpu_threads"] = str(runtime.device), runtime.cpu_threads
    started = datetime.now(timezone.utc).isoformat()
    seeds: dict[str, int] = {}

    def gen(domain: str) -> torch.Generator:
        seed = int.from_bytes(
            hashlib.sha256(f"demo-frame-v1:{config.values['seed']}:{domain}".encode()).digest()[:8],
            "big",
        ) % (2**63)
        seeds[domain] = seed
        return torch.Generator(device=runtime.device).manual_seed(seed)

    bits = torch.randint(2, (1, 8, 400, 2), generator=gen("bits"), device=runtime.device)
    pilots = classes_to_symbols(
        torch.randint(4, (1, 8, 64), generator=gen("pilots"), device=runtime.device),
        dtype=runtime.complex_dtype,
    )
    frame = build_frame(bits, pilots, config, runtime)
    for attempt in range(config.values["data"]["max_channel_attempts"]):
        paths = sample_paths(config, "affine_doppler_strong", generator=gen(f"paths:{attempt}"))
        try:
            support = validate_cp_support(paths, frame.layout, config)
        except ValueError:
            continue
        if torch.unique(paths.epsilon[paths.epsilon != 0]).numel() >= 2:
            break
    else:
        raise ValueError(
            "demo-frame could not sample CP-valid paths with distinct nonzero time scaling"
        )
    arrival = int(
        torch.randint(
            f["arrival_offset_max_samples"] + 1, (), generator=gen("arrival"), device=runtime.device
        )
    )
    fs = w["sample_rate_hz"]
    # Include all delayed/stretched frame support, not merely the transmitted length.
    end_s = ((frame.layout.frame_samples / fs + paths.delay_s) / (1 + paths.epsilon)).max()
    recording_samples = arrival + math.ceil(float(end_s) * fs) + 1
    times = torch.arange(recording_samples, dtype=torch.float64, device=runtime.device) / fs
    rx_analytic = affine_waveform(
        frame, paths, times, config, arrival_offset_samples=arrival, downconvert=False
    )
    rx_iq = rx_analytic * torch.exp(-2j * math.pi * w["carrier_hz"] * (times - arrival / fs))
    rx_real = math.sqrt(2) * rx_analytic.real
    sync = normalized_correlation(
        rx_real, frame.lfm.real.to(torch.float64), config.values["receiver"]["lfm_threshold"]
    )
    sigma2 = sigma2_from_esn0(10.0)
    noise = complex_awgn(rx_iq.shape, sigma2, gen_r=gen("noise-real"), gen_i=gen("noise-imag"))
    windows, errors = [], []
    idx = frame.allocation.data_grid_index
    for block, (start, end) in enumerate(frame.layout.useful):
        clean_iq = rx_iq[0, arrival + start : arrival + end]
        noise_iq = noise[0, arrival + start : arrival + end]
        observed = fft_receive(clean_iq, frame.allocation)
        h = effective_matrix(paths, frame.layout, config, block)
        grid = frame.grid[0, block].to(torch.complex128)
        error = float(
            torch.linalg.vector_norm(observed - h @ grid) / torch.linalg.vector_norm(observed)
        )
        if not math.isfinite(error) or error >= 1e-9:
            raise ValueError(
                f"demo-frame independent waveform/H check failed at block {block}: {error}"
            )
        converted_h = h.to(runtime.complex_dtype)
        noisy_grid = fft_receive((clean_iq + noise_iq).to(runtime.complex_dtype), frame.allocation)
        received = preprocess(converted_h, noisy_grid, pilots[0, block], frame.allocation, sigma2)
        errors.append(error)
        windows.append(
            {
                "block": block,
                "recording_start": arrival + start,
                "recording_end": arrival + end,
                "window_offset_s": 0.0,
                "h_grid_reference": h,
                "observed_grid_reference": observed,
                "H": received.H,
                "y": received.y,
                "sigma2": received.sigma2,
                "x": frame.grid[0, block, idx],
                "bits": bits[0, block],
                "pilots": pilots[0, block],
                "relative_error_complex128": error,
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "resolved_config.yaml")
    _write_json(output / "environment.json", _provenance(runtime))
    artifact = output / "frame.pt"
    torch.save(
        {
            "schema_version": 1,
            "layout": asdict(frame.layout),
            "allocation": asdict(frame.allocation),
            "grid": frame.grid,
            "bits": bits,
            "pilots": pilots,
            "tx_analytic": frame.analytic,
            "tx_real": frame.real,
            "rx_analytic_noiseless": rx_analytic,
            "rx_real_noiseless": rx_real,
            "rx_iq_noiseless": rx_iq,
            "iq_noise": noise,
            "rx_iq_noisy": rx_iq + noise,
            "lfm_real": frame.lfm.real.to(torch.float64),
            "sync": asdict(sync),
            "paths": asdict(paths),
            "cp_endpoints_s": support,
            "arrival_offset_samples": arrival,
            "seeds": seeds,
            "esn0_db": 10.0,
            "sigma2": sigma2,
            "windows": windows,
        },
        artifact,
    )
    figures = output / "figures"
    figures.mkdir()
    _figures(artifact, figures)
    hashes = {
        str(p.relative_to(output)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    result = {
        "schema_version": 1,
        "status": "passed",
        "output": str(output),
        "argv": argv,
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "one physical frame; ideal I/Q, oracle timing, perfect CSI, uncoded; "
            "no detector or performance conclusions"
        ),
        "dimensions": {
            "n_grid": 512,
            "n_data": 400,
            "n_fft_wave": 8192,
            "cp_samples": 2048,
            "n_blocks": 8,
            "data_bits": bits.numel(),
            "frame_samples": frame.layout.frame_samples,
            "recording_samples": recording_samples,
        },
        "reference_dtype": "complex128",
        "example_dtype": str(runtime.complex_dtype),
        "seed": config.values["seed"],
        "seeds": seeds,
        "path_resampling_count": attempt,
        "esn0_db": 10.0,
        "sigma2": sigma2,
        "arrival_offset_samples": arrival,
        "max_relative_error_complex128": max(errors),
        "sync": {
            "input": "noiseless real passband; no real-noise model",
            "peak_start": int(sync.peak_start.item()),
            "peak_score": float(sync.peak_score.item()),
            "detected": bool(sync.detected.item()),
            "candidate_arrival": int(sync.candidate_arrival.item()),
            "threshold": sync.threshold,
            "peak_meaning": "selected template start, not earliest path or frame origin",
            "used_for_fft_timing": False,
            "oracle_channel_assisted_timing": False,
            "injected_frame_origin_samples": arrival,
            "earliest_physical_lfm_start_samples": arrival
            + float(((frame.layout.lfm[0] / fs + paths.delay_s) / (1 + paths.epsilon)).min()) * fs,
        },
        "receiver_sync_mode": "oracle_timing",
        "training_and_sweeps": "未执行",
        "artifact_sha256": hashes,
    }
    # The success receipt is last: incomplete outputs never advertise success.
    _write_json(output / "demo.json", result)
    return result
