"""Audit saved demo arrays with NumPy; torch is used only as safe deserializer."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RESEARCH = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, dict):
        return {key: arrays(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [arrays(item) for item in value]
    return value


results = []
for dtype in ("complex128", "complex64"):
    directory = ROOT / f"runs/wp8-check-demo-{dtype}"
    receipt = json.loads((directory / "demo.json").read_text())
    for name, digest in receipt["artifact_sha256"].items():
        assert sha(directory / name) == digest
    saved = arrays(torch.load(directory / "frame.pt", map_location="cpu", weights_only=True))
    allocation, layout, paths = saved["allocation"], saved["layout"], saved["paths"]
    assert saved["tx_real"].shape == (1, 91776)
    assert saved["grid"].shape == (1, 8, 512)
    assert saved["bits"].size == 6400
    assert len(saved["windows"]) == 8
    assert np.unique(paths["epsilon"][paths["epsilon"] != 0]).size >= 2
    di, pi, bins = [
        allocation[key] for key in ("data_grid_index", "pilot_grid_index", "baseband_fft_index")
    ]
    fs = layout["sample_rate_hz"]
    arrival = saved["arrival_offset_samples"]
    assert 0 <= arrival <= 1920
    receive = saved["rx_iq_noiseless"][0]
    noise = saved["iq_noise"][0]
    np.testing.assert_array_equal(saved["rx_iq_noisy"][0], receive + noise)
    expected_length = (
        arrival
        + int(np.ceil(np.max((91776 / fs + paths["delay_s"]) / (1 + paths["epsilon"])) * fs))
        + 1
    )
    assert receive.size == expected_length
    tolerance = 2e-6 if dtype == "complex64" else 1e-9
    np.testing.assert_allclose(
        saved["tx_real"], np.sqrt(2) * saved["tx_analytic"].real, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        saved["rx_real_noiseless"],
        np.sqrt(2) * saved["rx_analytic_noiseless"].real,
        rtol=1e-14,
        atol=1e-14,
    )
    phase = np.exp(-2j * np.pi * 24000 * (np.arange(receive.size) / fs - arrival / fs))
    np.testing.assert_allclose(
        receive, saved["rx_analytic_noiseless"][0] * phase, rtol=1e-10, atol=1e-10
    )
    fft_errors, pilot_errors, noise_errors, cp_errors = [], [], [], []
    for block, (start, end), (cp_start, cp_end) in zip(
        saved["windows"], layout["useful"], layout["cp"]
    ):
        m = block["block"]
        assert end - start == 8192 and cp_end - cp_start == 2048
        assert (
            block["recording_start"] == arrival + start and block["recording_end"] == arrival + end
        )
        # Recompute CP support independently from time mapping for every path/block.
        endpoint = (
            (1 + paths["epsilon"][:, None]) * np.array([0.0, 8191 / fs])
            - paths["delay_s"][:, None]
            + paths["epsilon"][:, None] * (start / fs)
        )
        assert np.all(endpoint >= -2048 / fs) and np.all(endpoint < 8192 / fs)
        np.testing.assert_allclose(endpoint, saved["cp_endpoints_s"][m], rtol=1e-13, atol=1e-15)
        tx = saved["tx_analytic"][0, cp_start:end]
        baseband = tx * np.exp(-2j * np.pi * 24000 * np.arange(cp_start, end) / fs)
        cp_errors.append(float(np.max(np.abs(baseband[:2048] - baseband[-2048:]))))
        assert cp_errors[-1] < tolerance
        txgrid = np.fft.fft(baseband[2048:], norm="ortho")[bins]
        np.testing.assert_allclose(txgrid, saved["grid"][0, m], rtol=tolerance, atol=tolerance)
        h = block["h_grid_reference"]
        spectrum = np.fft.fft(receive[arrival + start : arrival + end], norm="ortho")[bins]
        relative = np.linalg.norm(spectrum - h @ saved["grid"][0, m]) / np.linalg.norm(spectrum)
        fft_errors.append(float(relative))
        assert relative < 1e-9
        np.testing.assert_array_equal(block["H"], h[np.ix_(di, di)].astype(dtype))
        assert block["H"].shape == (400, 400) and block["y"].shape == (400,)
        assert np.isclose(block["sigma2"], 0.1)
        noisy = (receive + noise)[arrival + start : arrival + end].astype(dtype)
        independent_y = (
            np.fft.fft(noisy, norm="ortho")[bins[di]]
            - h[np.ix_(di, pi)].astype(dtype) @ block["pilots"]
        )
        pilot_errors.append(
            float(np.linalg.norm(block["y"] - independent_y) / np.linalg.norm(block["y"]))
        )
        residual = block["y"] - block["H"] @ block["x"]
        noise_fft = np.fft.fft(noise[arrival + start : arrival + end], norm="ortho")[bins[di]]
        noise_errors.append(
            float(np.linalg.norm(residual - noise_fft) / np.linalg.norm(block["y"]))
        )
        assert pilot_errors[-1] < tolerance and noise_errors[-1] < tolerance
    # Independent full-lag linear matched filter and direct energies.
    rf, template = saved["rx_real_noiseless"][0], saved["lfm_real"]
    correlation = np.correlate(rf, template, mode="valid")
    window_energy = np.convolve(rf**2, np.ones(template.size), mode="valid")
    scores = correlation**2 / (window_energy * np.sum(template**2) + saved["sync"]["epsilon_sync"])
    np.testing.assert_allclose(scores, saved["sync"]["scores"][0], rtol=1e-8, atol=1e-10)
    peak = int(np.argmax(scores))
    assert peak == int(saved["sync"]["peak_start"][0])
    assert bool(scores[peak] > saved["sync"]["threshold"]) == receipt["sync"]["detected"]
    assert receipt["sync"]["used_for_fft_timing"] is False
    assert receipt["receiver_sync_mode"] == "oracle_timing"
    earliest = (
        arrival + np.min((layout["lfm"][0] / fs + paths["delay_s"]) / (1 + paths["epsilon"])) * fs
    )
    assert np.isclose(earliest, receipt["sync"]["earliest_physical_lfm_start_samples"])
    results.append(
        {
            "dtype": dtype,
            "status": "passed",
            "directory": str(directory.relative_to(ROOT)),
            "numpy_fft_max_relative_error": max(fft_errors),
            "numpy_pilot_max_relative_error": max(pilot_errors),
            "numpy_noise_max_relative_error": max(noise_errors),
            "cp_max_absolute_error": max(cp_errors),
            "sync_peak": peak,
            "sync_score": float(scores[peak]),
            "earliest_lfm_start": float(earliest),
            "recording_samples": receive.size,
            "rehashed_artifacts": receipt["artifact_sha256"],
            "receipt_sha256": sha(directory / "demo.json"),
            "iq_noise_variance_real_observed": float(noise.real.var()),
            "iq_noise_variance_imag_observed": float(noise.imag.var()),
            "noise_statistics_scope": (
                "descriptive single frame; statistical acceptance resides in test_wp2"
            ),
        }
    )
(RESEARCH / "check-demo-artifact-audit.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        [{key: value for key, value in r.items() if key != "rehashed_artifacts"} for r in results],
        indent=2,
    )
)
