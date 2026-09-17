"""Full-size CPU physical-channel audit; not dataset generation or system smoke."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from pgvamp_ofdm.channel.affine import receive_window
from pgvamp_ofdm.channel.effective_matrix import effective_matrix
from pgvamp_ofdm.channel.noise import complex_awgn, sigma2_from_esn0
from pgvamp_ofdm.channel.parameters import sample_paths
from pgvamp_ofdm.channel.validity import validate_cp_support
from pgvamp_ofdm.cli import _provenance
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.modulation.qpsk import classes_to_symbols
from pgvamp_ofdm.receiver.fft_receiver import fft_receive
from pgvamp_ofdm.receiver.preprocessing import preprocess
from pgvamp_ofdm.utils.device import resolve_runtime
from pgvamp_ofdm.waveform.frame import build_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cpu_dev.yaml")
    parser.add_argument("--output", default="runs/wp2-audit")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.values["runtime"]["device"] != "cpu":
        raise ValueError("this audit records the CPU acceptance gate")
    runtime = resolve_runtime(dtype=config.values["runtime"]["dtype"])
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("audit output must be empty; preserve earlier evidence in a separate path")
    output.mkdir(parents=True, exist_ok=True)
    seeds = {}

    def gen(domain):
        seed = int.from_bytes(
            hashlib.sha256(f"wp2-v1:{config.values['seed']}:{domain}".encode()).digest()[:8], "big"
        ) % (2**63)
        seeds[domain] = seed
        return torch.Generator().manual_seed(seed)

    records = []
    for frame_index in range(2):
        bits = torch.randint(2, (1, 8, 400, 2), generator=gen(f"bits:{frame_index}"))
        pilots = classes_to_symbols(
            torch.randint(4, (1, 8, 64), generator=gen(f"pilots:{frame_index}")),
            dtype=runtime.complex_dtype,
        )
        frame = build_frame(bits, pilots, config, runtime)
        paths = sample_paths(config, "affine_doppler_strong", generator=gen(f"paths:{frame_index}"))
        support = validate_cp_support(paths, frame.layout, config)
        path_record = {
            "gain_real": paths.gain.real.tolist(),
            "gain_imag": paths.gain.imag.tolist(),
            "delay_s": paths.delay_s.tolist(),
            "epsilon": paths.epsilon.tolist(),
            "scenario": paths.scenario,
            "resampling_count": 0,
        }
        windows = []
        tensor_windows = []
        for block in range(8):
            # Audit every block, with extra independently checked nonzero windows at both ends.
            offsets = [0.0, -0.0002 if block == 0 else 0.0003] if block in (0, 7) else [0.0]
            for offset in offsets:
                h = effective_matrix(paths, frame.layout, config, block, window_offset_s=offset)
                wave = receive_window(frame, paths, config, block, window_offset_s=offset)
                observed = fft_receive(wave, frame.allocation)[0]
                predicted = h @ frame.grid[0, block].to(torch.complex128)
                relative = float(
                    torch.linalg.vector_norm(observed - predicted)
                    / torch.linalg.vector_norm(observed)
                )
                observed64 = fft_receive(wave.to(torch.complex64), frame.allocation)[0]
                predicted64 = h.to(torch.complex64) @ frame.grid[0, block].to(torch.complex64)
                error64 = float(
                    torch.linalg.vector_norm(observed64 - predicted64)
                    / torch.linalg.vector_norm(observed64)
                )
                idx, pi = frame.allocation.data_grid_index, frame.allocation.pilot_grid_index
                clean = preprocess(
                    h,
                    observed,
                    frame.grid[0, block, pi].to(torch.complex128),
                    frame.allocation,
                    0.1,
                    window_offset_s=offset,
                )
                cancellation = float(
                    torch.linalg.vector_norm(
                        clean.y - clean.H @ frame.grid[0, block, idx].to(torch.complex128)
                    )
                )
                noise = complex_awgn(
                    (8192,),
                    0.1,
                    gen_r=gen(f"noise-r:{frame_index}:{block}:{offset}"),
                    gen_i=gen(f"noise-i:{frame_index}:{block}:{offset}"),
                )
                noisy = preprocess(
                    h,
                    fft_receive(wave[0] + noise, frame.allocation),
                    frame.grid[0, block, pi].to(torch.complex128),
                    frame.allocation,
                    0.1,
                    window_offset_s=offset,
                )
                noisy_residual = float(
                    torch.linalg.vector_norm(
                        noisy.y
                        - clean.H @ frame.grid[0, block, idx].to(torch.complex128)
                        - fft_receive(noise, frame.allocation)[idx]
                    )
                )
                leakage = float(
                    (h[idx][:, pi] @ frame.grid[0, block, pi].to(torch.complex128))
                    .abs()
                    .square()
                    .sum()
                )
                snr_rx_db = 10 * math.log10(float(clean.H.abs().square().sum()) / (400 * 0.1))
                assert (
                    relative < 1e-9
                    and error64 < 2e-6
                    and cancellation < 1e-8
                    and noisy_residual < 1e-8
                )
                windows.append(
                    {
                        "block": block,
                        "offset_s": offset,
                        "absolute_start_s": frame.layout.useful[block][0] / 96000 + offset,
                        "relative_error_complex128": relative,
                        "relative_error_complex64": error64,
                        "pilot_leakage_energy": leakage,
                        "cancellation_residual_norm": cancellation,
                        "noisy_cancellation_residual_norm": noisy_residual,
                        "sigma2": 0.1,
                        "snr_rx_db": snr_rx_db,
                    }
                )
                tensor_windows.append(
                    {
                        "block": block,
                        "offset_s": offset,
                        "h_grid": h,
                        "iq_waveform": wave,
                        "noise": noise,
                        "observed_grid": observed,
                        "y_clean": clean.y,
                        "y_noisy": noisy.y,
                    }
                )
        artifact = output / f"frame-{frame_index}.pt"
        torch.save(
            {
                "grid": frame.grid,
                "bits": bits,
                "gain": paths.gain,
                "delay_s": paths.delay_s,
                "epsilon": paths.epsilon,
                "windows": tensor_windows,
            },
            artifact,
        )
        records.append(
            {
                "frame_index": frame_index,
                "tensor_artifact": artifact.name,
                "tensor_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "paths": path_record,
                "cp_endpoints_s": support.tolist(),
                "windows": windows,
            }
        )

    variance = sigma2_from_esn0(4.0)
    noise = complex_awgn(
        (4096, 8192), variance, gen_r=gen("statistics-real"), gen_i=gen("statistics-imag")
    )
    frequency = fft_receive(noise, frame.allocation)
    selected = frequency[:, [0, 17, 91, 200, 301, 400, 450, 511]]
    covariance = selected.T @ selected.conj() / selected.shape[0]
    delta = covariance - variance * torch.eye(8, dtype=torch.complex128)
    # Union-bound conservative 7-sigma threshold for 64 complex covariance entries.
    covariance_limit = 7 * variance / math.sqrt(selected.shape[0])
    assert float(delta.abs().max()) < covariance_limit
    bits = torch.randint(2, (256, 400, 2), generator=gen("ber-bits"))
    symbols = torch.complex(
        1 - 2 * bits[..., 0].double(), 1 - 2 * bits[..., 1].double()
    ) / math.sqrt(2)
    spectrum = torch.zeros((256, 8192), dtype=torch.complex128)
    spectrum[:, frame.allocation.baseband_fft_index[frame.allocation.data_grid_index]] = symbols
    received = fft_receive(torch.fft.ifft(spectrum, norm="ortho") + noise[:256], frame.allocation)
    data_y = received[:, frame.allocation.data_grid_index]
    errors = int((torch.stack((data_y.real < 0, data_y.imag < 0), -1) != bits).sum())
    theory = 0.5 * math.erfc(math.sqrt(1 / variance) / math.sqrt(2))
    tolerance = 6 * math.sqrt(bits.numel() * theory * (1 - theory))
    assert abs(errors - bits.numel() * theory) < tolerance
    stats = {
        "esn0_db": 4.0,
        "sigma2": variance,
        "time_samples": noise.numel(),
        "fft_windows": 4096,
        "real_mean": float(noise.real.mean()),
        "imag_mean": float(noise.imag.mean()),
        "real_variance": float(noise.real.var()),
        "imag_variance": float(noise.imag.var()),
        "real_imag_covariance": float((noise.real * noise.imag).mean()),
        "fft_complex_variance": float(frequency.abs().square().mean()),
        "selected_bin_covariance_real": covariance.real.tolist(),
        "selected_bin_covariance_imag": covariance.imag.tolist(),
        "covariance_max_error": float(delta.abs().max()),
        "covariance_limit": covariance_limit,
        "ber_errors": errors,
        "ber_bits": bits.numel(),
        "ber_measured": errors / bits.numel(),
        "ber_theory": theory,
        "ber_count_tolerance_6sigma": tolerance,
    }
    assert abs(stats["real_mean"]) < 6 * math.sqrt(variance / (2 * noise.numel()))
    assert abs(stats["imag_mean"]) < 6 * math.sqrt(variance / (2 * noise.numel()))
    assert abs(stats["real_imag_covariance"]) < 6 * variance / (2 * math.sqrt(noise.numel()))
    assert abs(stats["real_variance"] - variance / 2) < 0.006 * variance / 2
    assert abs(stats["imag_variance"] - variance / 2) < 0.006 * variance / 2
    assert abs(stats["fft_complex_variance"] - variance) < 0.006 * variance
    report = {
        "status": "passed",
        "scope": "WP2 physical channel only; ideal IQ/timing, perfect CSI, uncoded",
        "command": sys.argv,
        "source": _provenance(runtime),
        "seeds": seeds,
        "n_fft_wave": 8192,
        "n_grid": 512,
        "n_data": 400,
        "frames": records,
        "noise_statistics": stats,
        "cuda": "not executed; CPU audit",
        "system_smoke_training_sweeps": "未执行",
    }
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config.values), encoding="utf-8")
    report["resolved_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": "passed",
                "frames": 2,
                "windows": sum(len(f["windows"]) for f in records),
                "max_relative_error": max(
                    w["relative_error_complex128"] for f in records for w in f["windows"]
                ),
                "noise": stats,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
