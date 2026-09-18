"""Persisted full-size demo replay, independent FFT/cancellation and CLI boundaries."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.demo import demo_frame


@pytest.fixture(scope="module", params=["complex128", "complex64"])
def demo(request, tmp_path_factory):
    output = tmp_path_factory.mktemp("demo") / "frame"
    config = load_config("configs/cpu_dev.yaml", dtype=request.param)
    receipt = demo_frame(config, output, argv=["test_demo", request.param])
    return output, receipt, torch.load(output / "frame.pt", weights_only=True)


def test_full_frame_independent_fft_and_pilot_cancellation(demo):
    output, receipt, saved = demo
    assert receipt["status"] == "passed"
    assert saved["tx_real"].shape == (1, 91776)
    assert saved["grid"].shape == (1, 8, 512)
    assert saved["bits"].numel() == 6400
    assert len(saved["windows"]) == 8
    assert torch.unique(saved["paths"]["epsilon"]).numel() >= 2
    assert (saved["paths"]["epsilon"] != 0).all()
    assert saved["cp_endpoints_s"][..., 0].min() >= -2048 / 96000
    assert saved["cp_endpoints_s"][..., 1].max() < 8192 / 96000
    allocation = saved["allocation"]
    idx, pi = allocation["data_grid_index"], allocation["pilot_grid_index"]
    bins = allocation["baseband_fft_index"]
    for block in saved["windows"]:
        start, end = block["recording_start"], block["recording_end"]
        wave = saved["rx_iq_noiseless"][0, start:end]
        spectrum = torch.fft.fft(wave, norm="ortho")[bins]
        h = block["h_grid_reference"]
        expected = h @ saved["grid"][0, block["block"]].to(torch.complex128)
        assert (
            torch.linalg.vector_norm(spectrum - expected) / torch.linalg.vector_norm(spectrum)
            < 1e-9
        )
        assert block["H"].shape == (400, 400) and block["y"].shape == (400,)
        assert block["sigma2"] == 0.1
        noisy = saved["rx_iq_noisy"][0, start:end].to(block["H"].dtype)
        y_grid = torch.fft.fft(noisy, norm="ortho")[bins]
        expected_y = y_grid[idx] - h[idx][:, pi].to(block["H"].dtype) @ block["pilots"]
        # Independent indexing changes BLAS strides/reduction order, so compare
        # numerical agreement, not bits. Keep the physical double gate above.
        tolerance = 2e-6 if block["H"].dtype == torch.complex64 else 1e-9
        torch.testing.assert_close(block["y"], expected_y, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(block["H"], h[idx][:, idx].to(block["H"].dtype), rtol=0, atol=0)
        residual = block["y"] - block["H"] @ block["x"]
        expected_noise = torch.fft.fft(saved["iq_noise"][0, start:end], norm="ortho")[bins[idx]]
        if block["H"].dtype == torch.complex64:
            # Dense 400-term products accumulate single-precision rounding.
            # Use WP2's normwise 2e-6 gate relative to the observation, not a
            # componentwise relative error on near-zero noise after subtraction.
            delta = torch.linalg.vector_norm(residual.to(torch.complex128) - expected_noise)
            assert delta / torch.linalg.vector_norm(block["y"]) < 2e-6
        else:
            torch.testing.assert_close(residual, expected_noise, rtol=1e-9, atol=1e-9)
    for name, digest in receipt["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads((output / "environment.json").read_text())["package_source_hashes"]


def test_sync_is_separate_and_direct_correlation_matches(demo):
    _, receipt, saved = demo
    sync = saved["sync"]
    assert not receipt["sync"]["used_for_fft_timing"]
    assert receipt["receiver_sync_mode"] == "oracle_timing"
    lag = int(sync["peak_start"].item())
    template = saved["lfm_real"]
    section = saved["rx_real_noiseless"][0, lag : lag + len(template)]
    score = (section @ template).square() / (
        section.square().sum() * template.square().sum() + sync["epsilon_sync"]
    )
    torch.testing.assert_close(score, sync["peak_score"][0], rtol=1e-10, atol=1e-12)
    for block, (start, _) in zip(saved["windows"], saved["layout"]["useful"]):
        assert block["recording_start"] == saved["arrival_offset_samples"] + start
    # Reconstruct the recorded random streams without any simulation helper.
    seeds = saved["seeds"]
    assert len(set(seeds.values())) == len(seeds)
    bits = torch.randint(2, (1, 8, 400, 2), generator=torch.Generator().manual_seed(seeds["bits"]))
    assert torch.equal(bits, saved["bits"])
    real = torch.randn(
        saved["iq_noise"].shape,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(seeds["noise-real"]),
    )
    imag = torch.randn(
        saved["iq_noise"].shape,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(seeds["noise-imag"]),
    )
    scale = 0.1**0.5 / 2**0.5
    assert torch.equal(torch.complex(real * scale, imag * scale), saved["iq_noise"])


def test_reject_existing_and_unsupported_config(tmp_path):
    with pytest.raises(ValueError, match="fresh directory"):
        demo_frame(load_config(), tmp_path)
    with pytest.raises(ValueError, match="algebra_fixture"):
        demo_frame(load_config("configs/smoke_math.yaml"), tmp_path / "math")
    config = load_config()
    config.values["receiver"]["sync_mode"] = "lfm_detect"
    with pytest.raises(ValueError, match="oracle_timing"):
        demo_frame(config, tmp_path / "sync")
    config = load_config()
    config.values["frame"]["n_ofdm_symbols"] = 1
    with pytest.raises(ValueError, match="eight-block"):
        demo_frame(config, tmp_path / "small")
    assert list(tmp_path.iterdir()) == []


def test_deterministic_replay_and_failed_plot_has_no_success(demo, tmp_path, monkeypatch):
    output, _, saved = demo

    def fail_figures(*args):
        raise RuntimeError("injected figure failure")

    monkeypatch.setattr("pgvamp_ofdm.demo._figures", fail_figures)
    destination = tmp_path / "repeat"
    with pytest.raises(RuntimeError, match="injected figure failure"):
        demo_frame(load_config(output / "resolved_config.yaml"), destination)
    assert not (destination / "demo.json").exists()
    repeated = torch.load(destination / "frame.pt", weights_only=True)
    for key in ("grid", "bits", "tx_analytic", "rx_iq_noiseless", "iq_noise"):
        assert torch.equal(saved[key], repeated[key])
    for left, right in zip(saved["windows"], repeated["windows"]):
        assert torch.equal(left["H"], right["H"])
        assert torch.equal(left["y"], right["y"])


def test_cli_dispatch_and_errors(monkeypatch, tmp_path, capsys):
    from pgvamp_ofdm.cli import main

    def fake(config, output, *, argv):
        assert config.values["seed"] == 17
        assert config.values["runtime"]["dtype"] == "complex64"
        assert output == tmp_path / "demo"
        assert argv[0] == "demo-frame"
        return {"status": "dispatched"}

    monkeypatch.setattr("pgvamp_ofdm.demo.demo_frame", fake)
    assert (
        main(
            [
                "demo-frame",
                "--config",
                "configs/cpu_dev.yaml",
                "--dtype",
                "complex64",
                "--seed",
                "17",
                "--output",
                str(tmp_path / "demo"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "dispatched"
    env = dict(os.environ, MKL_THREADING_LAYER="TBB")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pgvamp_ofdm",
            "demo-frame",
            "--config",
            "configs/smoke_math.yaml",
            "--output",
            str(tmp_path / "error"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2 and "algebra_fixture" in result.stderr
    assert not (tmp_path / "error").exists()
