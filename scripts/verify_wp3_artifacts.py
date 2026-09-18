"""Independent persisted-array FFT/cancellation audit and compact/dense replay check."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from pgvamp_ofdm.data.dataset import EffectiveDataset
from pgvamp_ofdm.data.materialize import load_materialized
from pgvamp_ofdm.data.records import tensor_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.audit / "audit.json").read_text())
    assert report["resolved_config"]["runtime"]["dtype"] == "complex128", (
        "This bitwise persisted-artifact audit requires complex128; "
        "complex64 waveform parity is checked separately with explicit tolerances."
    )
    assert (
        report["source_manifest_sha256"] == hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    )
    allocation = report["allocation"]
    idx, pi = allocation["data_grid_index"], allocation["pilot_grid_index"]
    fft_idx = allocation["baseband_fft_index"]
    maximum = 0.0
    windows = 0
    hashes = []
    affine_verified = False
    for entry in report["frames"]:
        path = args.audit / entry["artifact"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        epsilon = saved["record"]["path_epsilon"].numpy()
        affine_verified |= len(np.unique(epsilon[epsilon != 0])) >= 2
        dataset = EffectiveDataset(args.manifest, saved["record"]["split"], cache_entries=0)
        record_index = next(
            i
            for i, r in enumerate(dataset.records)
            if r["frame_id"] == saved["record"]["frame_id"]
            and r["snr_copy"] == saved["record"]["snr_copy"]
        )
        for window in saved["windows"]:
            block = window["block"]
            start, end = window["recording_span"]
            wave = saved["received_iq"][start:end].numpy()
            np.testing.assert_array_equal(wave, window["iq_waveform"].numpy())
            observed = np.fft.fft(wave, norm="ortho")[fft_idx]
            h, grid = window["h_grid"].numpy(), saved["grid"][block].numpy()
            relative = float(np.linalg.norm(observed - h @ grid) / np.linalg.norm(observed))
            assert relative < 1e-9
            maximum = max(maximum, relative)
            np.testing.assert_allclose(
                observed, window["observed_grid"].numpy(), atol=1e-12, rtol=1e-12
            )
            noise = window["noise"].numpy()
            sigma2 = 10 ** (-float(saved["record"]["esn0_db"][block]) / 10)
            quadratures = []
            for part in ("real", "imag"):
                seed = int(saved["record"][f"noise_seed_{part}"][block])
                identity = [saved["record"]["frame_id"], block, saved["record"]["snr_copy"], part]
                canonical = json.dumps(
                    ["sha256-json-v1", report["resolved_config"]["seed"], "noise", identity],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                assert seed == int(hashlib.sha256(canonical.encode()).hexdigest()[:16], 16) % (
                    2**63
                )
                quadratures.append(
                    torch.randn(
                        len(noise),
                        generator=torch.Generator().manual_seed(seed),
                        dtype=torch.float64,
                    ).numpy()
                )
            # Preserve the declared underflow-safe scalar evaluation order for a
            # bitwise check; sqrt(sigma2/2) differs by an ULP for some variances.
            scale = math.sqrt(sigma2) / math.sqrt(2)
            np.testing.assert_array_equal(noise.real, scale * quadratures[0])
            np.testing.assert_array_equal(noise.imag, scale * quadratures[1])
            fft_noise = np.fft.fft(noise, norm="ortho")[fft_idx]
            pilot = h[np.ix_(idx, pi)] @ grid[pi]
            y = (observed + fft_noise)[idx] - pilot
            np.testing.assert_allclose(
                pilot, window["pilot_contribution"].numpy(), atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(y, window["y"].numpy(), atol=1e-12, rtol=1e-12)
            sigma2 = 10 ** (-float(saved["record"]["esn0_db"][block]) / 10)
            assert sigma2 == window["sigma2"]
            hdd = h[np.ix_(idx, idx)]
            snr = 10 * np.log10(np.sum(np.abs(hdd) ** 2) / (400 * sigma2))
            assert abs(snr - window["snr_rx_db"]) < 1e-12
            sample = dataset[record_index * dataset.blocks + block]
            np.testing.assert_allclose(sample["y"].numpy(), y, atol=1e-9, rtol=1e-8)
            np.testing.assert_allclose(sample["H"].numpy(), hdd, atol=0, rtol=0)
            hashes.append({"sample_id": sample["sample_id"], "payload_sha256": tensor_hash(sample)})
            windows += 1
    assert affine_verified, "audit requires at least two distinct nonzero path epsilons"
    dense = load_materialized(args.materialized, purpose="evaluation")
    dataset = EffectiveDataset(args.manifest, dense["metadata"]["split"], cache_entries=0)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["sample_id"] == dense["metadata"]["sample_ids"][i]
        for key in ("H", "y", "sigma2", "x", "bits"):
            assert torch.equal(sample[key], dense[key][i])
    result = {
        "status": "passed",
        "audit_windows": windows,
        "affine_verified": affine_verified,
        "max_relative_error_numpy": maximum,
        "materialized_samples": len(dataset),
        "replay_hashes": hashes,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "dense_sha256": hashlib.sha256(args.materialized.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "replay_hashes"}))


if __name__ == "__main__":
    main()
