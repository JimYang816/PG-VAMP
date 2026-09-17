"""Full-size WP1 transmit audit only; not a dataset audit or full-system smoke."""

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgvamp_ofdm.cli import _provenance
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.modulation.qpsk import classes_to_bits, classes_to_symbols, hard_decision
from pgvamp_ofdm.receiver.synchronization import normalized_correlation
from pgvamp_ofdm.utils.device import resolve_runtime
from pgvamp_ofdm.waveform.frame import build_frame, pad_recording


def _seed(master: int, domain: str) -> int:
    payload = f"wp1-audit-v1:{master}:{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _periodogram(
    samples: np.ndarray[Any, Any], fs: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
    # Preserve original segment length in the power normalization when zero padding.
    count = len(samples)
    nfft = 1 << (4 * count - 1).bit_length()
    frequency = np.fft.fftfreq(nfft, d=1 / fs)
    psd = np.abs(np.fft.fft(samples, n=nfft)) ** 2 / (fs * count)
    return (
        frequency,
        psd,
        {
            "samples": count,
            "nfft": nfft,
            "estimator": "two-sided rectangular periodogram; >=4x zero padding",
            "normalization": "abs(FFT_unscaled)**2/(Fs*original_segment_N)",
            "mean_real_power": float(np.mean(samples**2)),
            "psd_integral": float(psd.sum() * fs / nfft),
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Audit a validated physical config on explicit device/dtype, saving fresh artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cpu_dev.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    args = parser.parse_args(argv)
    config = load_config(args.config, device=args.device, dtype=args.dtype)
    if config.mode != "physical":
        parser.error("WP1 audit requires a physical configuration")
    rt = config.values["runtime"]
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    rt.update(device=str(runtime.device), cpu_threads=runtime.cpu_threads)
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error("output must be absent or empty; use a fresh directory to preserve evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    master = config.values["seed"]
    random.seed(master)
    np.random.seed(master % (1 << 32))
    torch.random.default_generator.manual_seed(master)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(master)
    domains = ("bits", "pilots", "arrival_offset", "sync_noise_real", "sync_noise_imag")
    seeds = {name: _seed(master, name) for name in domains}
    generators = {
        name: torch.Generator(device=runtime.device).manual_seed(seed)
        for name, seed in seeds.items()
    }
    w, f = config.values["waveform"], config.values["frame"]
    m = f["n_ofdm_symbols"]
    bits = torch.randint(
        2, (1, m, 400, 2), dtype=torch.uint8, device=runtime.device, generator=generators["bits"]
    )
    pilots = classes_to_symbols(
        torch.randint(4, (1, m, 64), device=runtime.device, generator=generators["pilots"]),
        dtype=runtime.complex_dtype,
    )
    frame = build_frame(bits, pilots, config, runtime)
    offset = int(
        torch.randint(
            f["arrival_offset_max_samples"] + 1,
            (),
            device=runtime.device,
            generator=generators["arrival_offset"],
        )
    )
    recording = pad_recording(frame.real, offset)
    analytic_recording = pad_recording(frame.analytic, offset)
    threshold = config.values["receiver"]["lfm_threshold"]
    sync = normalized_correlation(recording, frame.lfm.real, threshold)
    analytic_sync = normalized_correlation(analytic_recording, frame.lfm.analytic, threshold)
    truth = offset + frame.layout.lfm[0]
    if not bool(sync.detected.all()) or int(sync.peak_start[0]) != truth:
        raise AssertionError("real LFM peak failed known-offset audit")
    if not bool(analytic_sync.detected.all()) or int(analytic_sync.peak_start[0]) != truth:
        raise AssertionError("analytic LFM peak failed known-offset audit")
    atol, rtol = (1e-9, 1e-8) if runtime.complex_dtype == torch.complex128 else (2e-5, 2e-4)
    recovery_errors, bit_errors = [], 0
    for block, (start, stop) in enumerate(frame.layout.useful):
        fft = torch.fft.fft(frame.real[:, start:stop], dim=-1, norm="ortho")
        phase = torch.exp(
            torch.tensor(
                2j * math.pi * w["carrier_hz"] * start / w["sample_rate_hz"],
                dtype=torch.complex128,
                device=runtime.device,
            )
        ).to(runtime.complex_dtype)
        recovered = math.sqrt(2) * fft[:, frame.allocation.passband_fft_index] / phase
        torch.testing.assert_close(recovered, frame.grid[:, block], atol=atol, rtol=rtol)
        recovery_errors.append(float((recovered - frame.grid[:, block]).abs().max()))
        bits_hat = classes_to_bits(hard_decision(recovered[:, frame.allocation.data_grid_index]))
        bit_errors += int((bits_hat != bits[:, block]).sum())
    if bit_errors:
        raise AssertionError("no-channel bit recovery failed")
    expected_power = 464 / w["n_fft_wave"]
    useful_power = frame.ofdm.useful.abs().square().mean(-1)
    torch.testing.assert_close(
        useful_power, torch.full_like(useful_power, expected_power), atol=atol, rtol=rtol
    )
    assert torch.equal(
        frame.ofdm.with_cp[..., : w["cp_samples"]], frame.ofdm.useful[..., -w["cp_samples"] :]
    )
    nl = f["lfm_samples"]
    noise_length = max(nl + 1, math.ceil(1.5625 * nl))
    noise_real = torch.randn(
        12,
        noise_length,
        dtype=runtime.real_dtype,
        device=runtime.device,
        generator=generators["sync_noise_real"],
    )
    noise_imag = torch.randn(
        12,
        noise_length,
        dtype=runtime.real_dtype,
        device=runtime.device,
        generator=generators["sync_noise_imag"],
    )
    noise_complex = torch.complex(noise_real, noise_imag) / math.sqrt(2)
    noise_sync_real = normalized_correlation(noise_real, frame.lfm.real, threshold)
    noise_sync_complex = normalized_correlation(noise_complex, frame.lfm.analytic, threshold)
    config.save(output / "resolved_config.yaml")
    provenance = _provenance(runtime)
    provenance.update(
        audit="wp1-transmit-v1",
        seeds=seeds,
        master_seed=master,
        seed_derivation=(
            "int.from_bytes(sha256('wp1-audit-v1:{master}:{domain}').digest()[:8], "
            "'big') & (2**63-1)"
        ),
        time_phase_dtype="float64 then explicit output dtype conversion",
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    _json(output / "environment.json", provenance)
    torch.save(
        {
            "analytic": frame.analytic.cpu(),
            "real": frame.real.cpu(),
            "grid": frame.grid.cpu(),
            "useful": frame.ofdm.useful.cpu(),
            "with_cp": frame.ofdm.with_cp.cpu(),
            "data_bits": bits.cpu(),
            "pilot_symbols": pilots.cpu(),
            "lfm_analytic": frame.lfm.analytic.cpu(),
            "lfm_real": frame.lfm.real.cpu(),
            "lfm_window": frame.lfm.window.cpu(),
            "lfm_metadata": frame.lfm.metadata,
            "lfm_normalization": frame.lfm.normalization,
            "layout": asdict(frame.layout),
            "allocation": {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in asdict(frame.allocation).items()
            },
            "sample_rate_hz": w["sample_rate_hz"],
            "arrival_offset_samples": offset,
            "recording_real": recording.cpu(),
            "recording_analytic": analytic_recording.cpu(),
        },
        output / "waveforms.pt",
    )
    first_start, first_stop = frame.layout.useful[0]
    segments = {
        "frame": frame.real[0],
        "ofdm_useful_block0": frame.real[0, first_start:first_stop],
        "lfm": frame.lfm.real,
    }
    psd_arrays: dict[str, Any] = {}
    psd_metrics = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for axis, (name, values) in zip(axes, segments.items(), strict=True):
        frequency, psd, metrics = _periodogram(values.double().cpu().numpy(), w["sample_rate_hz"])
        mask = (np.abs(frequency) >= w["band_hz"][0]) & (np.abs(frequency) <= w["band_hz"][1])
        metrics["outside_design_band_energy_ratio"] = float(psd[~mask].sum() / psd.sum())
        metrics["integration_absolute_error"] = abs(
            metrics["psd_integral"] - metrics["mean_real_power"]
        )
        if not math.isclose(
            metrics["psd_integral"], metrics["mean_real_power"], rel_tol=1e-12, abs_tol=1e-14
        ):
            raise AssertionError("PSD integral does not match stated interval power")
        psd_metrics[name] = metrics
        psd_arrays[name + "_frequency_hz"] = frequency
        psd_arrays[name + "_density"] = psd
        positive = frequency >= 0
        axis.plot(
            frequency[positive] / 1000, 10 * np.log10(np.maximum(psd[positive], 1e-30)), lw=0.7
        )
        axis.axvspan(w["band_hz"][0] / 1000, w["band_hz"][1] / 1000, alpha=0.1, color="green")
        axis.set(
            xlim=(0, w["sample_rate_hz"] / 2000),
            ylim=(-160, -20),
            xlabel="Frequency (kHz)",
            ylabel="Two-sided PSD (dB/Hz)",
            title=f"{name}: {metrics['samples']} samples, NFFT={metrics['nfft']}, rectangular",
        )
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Real transmit PSD: positive half shown; negative mirror included in power/OOB\n"
        ">=4x zero padding; no OFDM transmit window; LFM uses Tukey alpha="
        f"{f['lfm_tukey_alpha']}",
        fontsize=10,
    )
    fig.savefig(output / "figures" / "psd.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(output / "psd.npz", **psd_arrays)
    lags = np.arange(sync.scores.shape[-1])
    correlation_arrays = {
        "lag_samples": lags,
        "lag_seconds": lags / w["sample_rate_hz"],
        "real_scores": sync.scores.cpu().numpy(),
        "analytic_scores": analytic_sync.scores.cpu().numpy(),
        "real_noise": noise_real.cpu().numpy(),
        "complex_noise": noise_complex.cpu().numpy(),
        "real_noise_scores": noise_sync_real.scores.cpu().numpy(),
        "complex_noise_scores": noise_sync_complex.scores.cpu().numpy(),
        "threshold": threshold,
        "injected_template_start": truth,
    }
    np.savez_compressed(output / "correlation.npz", **correlation_arrays)
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.plot(
        lags, correlation_arrays["real_scores"][0], label="Real RF matched correlation", lw=0.8
    )
    axis.axhline(threshold, linestyle="--", color="red", label=f"Threshold {threshold}")
    axis.axvline(truth, linestyle=":", color="green", label=f"Injected template start {truth}")
    axis.set(
        xlabel="Template start (samples in recording)",
        ylabel="Normalized squared correlation",
        xlim=(0, int(lags[-1])),
        ylim=(0, 1.05),
        title="WP1 no-channel LFM match: selected peak is a template start",
    )
    secondary = axis.secondary_xaxis(
        "top", functions=(lambda x: x / w["sample_rate_hz"], lambda x: x * w["sample_rate_hz"])
    )
    secondary.set_xlabel("Template start (seconds in recording)")
    axis.legend(loc="upper right")
    fig.savefig(output / "figures" / "correlation.png", dpi=160)
    plt.close(fig)
    summary = {
        "status": "passed",
        "scope": "WP1 transmit audit; not WP2 channel validation or full-system smoke",
        "device": str(runtime.device),
        "dtype": str(runtime.complex_dtype),
        "frame_samples": frame.layout.frame_samples,
        "recording_samples": recording.shape[-1],
        "frame_duration_s": frame.layout.frame_samples / w["sample_rate_hz"],
        "data_bits_per_frame": frame.layout.data_bits_per_frame,
        "arrival_offset_samples": offset,
        "recovery_max_abs_error_per_block": recovery_errors,
        "bit_errors": bit_errors,
        "recovery_tolerance": {"atol": atol, "rtol": rtol},
        "ofdm_useful_analytic_power": useful_power.cpu().tolist(),
        "target_analytic_power": expected_power,
        "lfm_analytic_power": float(frame.lfm.analytic.abs().square().mean()),
        "lfm_real_power": float(frame.lfm.real.square().mean()),
        "lfm_metadata": frame.lfm.metadata,
        "psd": psd_metrics,
        "sync": {
            "threshold": threshold,
            "epsilon_sync": sync.epsilon_sync,
            "truth_template_start": truth,
            "peak_start": int(sync.peak_start[0]),
            "peak_score": float(sync.peak_score[0]),
            "analytic_peak_start": int(analytic_sync.peak_start[0]),
            "analytic_peak_score": float(analytic_sync.peak_score[0]),
            "no_channel_frame_origin_error_samples": int(sync.peak_start[0])
            - f["leading_silence_samples"]
            - offset,
        },
        "noise_fixture": {
            "windows": 12,
            "samples_per_window": noise_length,
            "real_peaks": noise_sync_real.peak_score.cpu().tolist(),
            "complex_peaks": noise_sync_complex.peak_score.cpu().tolist(),
            "real_detections": int(noise_sync_real.detected.sum()),
            "complex_detections": int(noise_sync_complex.detected.sum()),
            "complex_union_bound_at_0_1": 12
            * (noise_length - nl + 1)
            * math.exp((nl - 1) * math.log1p(-0.1)),
            "interpretation": "fixed-seed fixtures, not a measured universal false-alarm guarantee",
        },
        "not_executed": [
            "physical channel",
            "CP path validity",
            "effective H",
            "detector comparison",
            "training",
            "full-system smoke",
        ],
    }
    _json(output / "summary.json", summary)
    _json(
        output / "artifact_hashes.json",
        {
            str(path.relative_to(output)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
