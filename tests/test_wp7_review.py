"""Independent WP7 review probes for timing, caches and physical examples."""

import json
from pathlib import Path

import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector, PGVAMPDetector, VAMPDetector
from pgvamp_ofdm.algorithms.prepared import PreparedDetector
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.data.generate import generate_dataset
from pgvamp_ofdm.evaluation import timing
from pgvamp_ofdm.evaluation.metrics import block_counts
from pgvamp_ofdm.evaluation.runner import physical_example, setup
from pgvamp_ofdm.reference.dense_pg_vamp import DensePGVAMP
from pgvamp_ofdm.reference.dense_vamp_cholesky import dense_vamp_cholesky
from pgvamp_ofdm.reporting.validation import _stream_diagnostics


@pytest.mark.parametrize(
    "algorithm,section,key", [("pg_vamp", "pg_vamp", "depth"), ("vamp", "vamp", "iterations")]
)
def test_main_evaluation_rejects_nonstandard_fixed_rounds_before_io(algorithm, section, key):
    config = load_config()
    config.values["evaluation"]["label"] = "main_simulation"
    config.values[section][key] = 3
    with pytest.raises(ValueError, match="main_simulation requires"):
        setup(config, Path("nonexistent-manifest.json"), None, [algorithm], True)


def test_report_accepts_finite_unresolved_negative_contraction(tmp_path):
    shared = dict(sample_id="s", input_hash="h", frame_id="f", scenario="s", esn0_db=0, block=0)
    record = {
        **shared,
        "algorithm": "PG",
        "status": "complete",
        "message_opportunities": 0,
        "no_information_opportunities": 1,
        "no_information": [1],
        "layer_summaries": [
            {
                "effective_candidate_ratio": [0],
                "effective_all_ratio": [0],
                "safety_relative": [0],
                "c": [-1e-18],
                "c_tolerance": [1e-16],
            }
        ],
    }
    path = tmp_path / "diagnostics.jsonl"
    path.write_text(json.dumps(record) + "\n")
    _, _, ranges, _ = _stream_diagnostics(path, {"s": shared}, {("PG", "s")})
    assert ranges["c"]["min"] == -1e-18


def test_finite_prediction_energy_overflow_is_hard_failure():
    bits = torch.zeros(2, 2, dtype=torch.uint8)
    x = torch.ones(2, dtype=torch.complex128)
    with pytest.raises(FloatingPointError, match="metric energy"):
        block_counts(bits, x * 1e200, bits, x)


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_physical_example_matches_baseband_lfm_and_expected_start(tmp_path, dtype):
    config = load_config()
    config.values["runtime"]["dtype"] = dtype
    config.values["data"].update(train_frames=1, val_frames=1)
    config.values["evaluation"].update(scenarios=["identity_awgn"], esn0_db=[0], frames_per_cell=1)
    config.values["receiver"]["lfm_threshold"] = 0.7
    manifest = generate_dataset(config, tmp_path / "data")
    data, _, _, _ = setup(config, manifest, None, ["mmse"], False)
    destination = tmp_path / "example.pt"
    physical_example(data, data.records[0], data[0], destination, "review")
    saved = torch.load(destination, weights_only=True)
    expected = (
        data.records[0]["arrival_offset_samples"]
        + config.values["frame"]["leading_silence_samples"]
    )
    assert saved["sync_peak"] == saved["expected_template_start"] == expected
    assert saved["sync_detected"] and saved["sync_threshold"] == 0.7
    assert saved["sync_scores"][expected] == pytest.approx(1, abs=2e-6)
    assert saved["received_iq"].dtype == getattr(torch, dtype)


@pytest.mark.parametrize("mode", timing.MODES)
def test_timer_boundary_contains_decomposition_and_counts_measured_draws(monkeypatch, mode):
    gen = torch.Generator().manual_seed(73)
    h = torch.randn(1, 4, 4, dtype=torch.complex128, generator=gen)
    y = torch.randn(10, 4, dtype=h.dtype, generator=gen)
    noise = torch.ones(1, dtype=torch.float64)
    inside = False
    factor_calls = []
    original_factor = MMSEDetector._factor

    def measured(operation, device):
        nonlocal inside
        assert not inside
        inside = True
        result = operation()
        inside = False
        return result, 2.0

    def factor(self, *args):
        factor_calls.append(inside)
        return original_factor(self, *args)

    monkeypatch.setattr(timing, "_measure", measured)
    monkeypatch.setattr(MMSEDetector, "_factor", factor)
    rows = timing.measure_detector(
        MMSEDetector(), h, y, noise, mode=mode, batch_size=2, warmup=0, repeats=3
    )
    assert all(factor_calls)
    assert len(factor_calls) == (6 if mode == timing.MODES[0] else 2)
    assert [r["unique_noise_observations"] for r in rows] == [3, 6]
    for row in rows:
        assert (
            row["total_amortized_per_block_ms"]
            == (row["prepare_ms"] + 6) / row["observation_count"]
        )


@pytest.mark.parametrize("factory", [VAMPDetector, PGVAMPDetector])
def test_prepared_predictions_match_independent_oracles_and_validate_y(factory):
    gen = torch.Generator().manual_seed(101)
    h = torch.randn(2, 8, 8, dtype=torch.complex128, generator=gen) / 8**0.5
    h[1, :, 5:] = 0
    noise = torch.tensor([0.5, 0.7], dtype=torch.float64)
    model = factory().eval()
    oracle = DensePGVAMP().eval()
    if factory is PGVAMPDetector:
        oracle.load_state_dict(model.state_dict())
    with torch.inference_mode():
        prepared = PreparedDetector(model, h, noise)
        for _ in range(3):
            y = torch.randn(2, 8, dtype=h.dtype, generator=gen)
            actual = prepared.detect(h, y, noise)
            expected = (
                oracle(h, y, noise)
                if factory is PGVAMPDetector
                else dense_vamp_cholesky(h, y, noise, iterations=8)
            )
            torch.testing.assert_close(actual.x_soft, expected.x_soft, atol=1e-9, rtol=1e-8)
        y[0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            prepared.detect(h, y, noise)
