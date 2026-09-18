import json

import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.data.generate import generate_dataset
from pgvamp_ofdm.evaluation.artifacts import csv_read, load_bundle
from pgvamp_ofdm.evaluation.runner import evaluate, setup, timing_observations


@pytest.fixture
def evaluation_data(tmp_path):
    config = load_config()
    config.values["data"].update(train_frames=1, val_frames=1)
    config.values["evaluation"].update(
        scenarios=["identity_awgn"],
        esn0_db=[0],
        frames_per_cell=1,
        timing_warmup=0,
        timing_repeats=2,
        batch_size=2,
        frame_cluster_bootstrap_repeats=30,
        label="smoke_system",
    )
    return config, generate_dataset(config, tmp_path / "data")


def test_real_evaluate_complete_pairing_counts_and_artifacts(
    evaluation_data, tmp_path, monkeypatch
):
    config, manifest = evaluation_data

    def forbidden(*a, **k):
        pytest.fail("CPU evaluation called CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    result = evaluate(config, manifest, tmp_path / "result", allow_untrained=True)
    assert result["samples"] == 8 and result["status"] == "complete"
    root = tmp_path / "result"
    bundle = load_bundle(root)
    rows = csv_read(root / "aggregate_metrics.csv")
    assert len(rows) == 3
    assert all(int(r["n_bits"]) == 6400 and int(r["n_frames"]) == 1 for r in rows)
    lineage = [json.loads(s) for s in (root / "lineage.jsonl").read_text().splitlines()]
    assert len({v["sample_id"] for v in lineage}) == 8
    assert all(v["algorithms"] == bundle["algorithms"] for v in lineage)
    frames = csv_read(root / "per_frame_metrics.csv")
    assert len({r["input_hashes"] for r in frames}) == 1
    timing = csv_read(root / "timing.csv")
    assert len([r for r in timing if r["algorithm"] != "shared_preprocessing"]) == 12
    example = torch.load(root / "example.pt", weights_only=True)
    assert example["input_hash"] == lineage[0]["input_hash"]
    pg = next(r for r in rows if r["algorithm"].startswith("PG"))
    assert pg["algorithm"] == "PG-VAMP-untrained"
    assert int(pg["message_opportunities"]) == 8 * 7
    assert int(pg["no_information_opportunities"]) == 8 * 8
    (root / "aggregate_metrics.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(root)


@pytest.mark.parametrize("failure", ["exception", "nonfinite", "infinite"])
@pytest.mark.parametrize("position", [0, 3, 7])
def test_failure_preserves_denominators(evaluation_data, tmp_path, monkeypatch, failure, position):
    config, manifest = evaluation_data
    monkeypatch.setattr("pgvamp_ofdm.evaluation.runner._benchmark_sample", lambda *a: [])
    original = MMSEDetector.detect
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == position + 1:
            if failure == "exception":
                raise FloatingPointError("injected Cholesky failed")
            result = original(self, *args, **kwargs)
            result.x_soft.fill_(float("nan") if failure == "nonfinite" else float("inf"))
            return result
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MMSEDetector, "detect", fail_once)
    root = tmp_path / "failed"
    result = evaluate(config, manifest, root, algorithms=["mmse"])
    assert result["status"] == "incomplete_or_failed"
    row = csv_read(root / "aggregate_metrics.csv")[0]
    assert row["ber"] == row["ber_ci_low"] == row["fer"] == ""
    assert int(row["n_bits"]) == 6400 and int(row["successful_blocks"]) == 7
    assert int(row["conditional_success_bits"]) == 5600
    failures = [json.loads(s) for s in (root / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1 and failures[0]["block"] == position
    assert not csv_read(root / "paired_comparisons.csv")
    load_bundle(root)


def test_preflight_checkpoint_population_and_target_isolation(evaluation_data, tmp_path):
    config, manifest = evaluation_data
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate(config, manifest, tmp_path / "no-checkpoint")
    assert not (tmp_path / "no-checkpoint.incomplete").exists()
    data, runtime, _, _ = setup(config, manifest, None, ["mmse"], False)
    sample = data[0]
    first = timing_observations(data, data.records[0], sample, runtime, 2, 7)
    sample["x"].zero_()
    sample["bits"].zero_()
    second = timing_observations(data, data.records[0], sample, runtime, 2, 7)
    assert torch.equal(first, second) and not torch.equal(first[0], first[1])
    config.values["evaluation"]["frames_per_cell"] = 2
    with pytest.raises(ValueError, match="plan differs"):
        setup(config, manifest, None, ["mmse"], False)
