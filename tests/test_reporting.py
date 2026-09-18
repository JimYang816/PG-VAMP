"""Read-only reports, persisted tamper rejection and seed accounting."""

import copy
import json
import shutil

import pytest
import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.data.generate import generate_dataset
from pgvamp_ofdm.data.manifest import file_hash
from pgvamp_ofdm.evaluation.artifacts import csv_read, csv_write, load_bundle
from pgvamp_ofdm.evaluation.runner import evaluate
from pgvamp_ofdm.reporting import report
from pgvamp_ofdm.reporting.validation import read_run, validate_merge


@pytest.fixture(scope="module")
def saved_report_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("report-evidence")
    config = load_config()
    config.values["data"].update(train_frames=1, val_frames=1)
    config.values["evaluation"].update(
        scenarios=["identity_awgn"],
        esn0_db=[25],
        frames_per_cell=1,
        timing_warmup=0,
        timing_repeats=2,
        batch_size=2,
        frame_cluster_bootstrap_repeats=20,
        label="smoke_system",
    )
    manifest = generate_dataset(config, root / "data")
    evaluate(config, manifest, root / "result", allow_untrained=True)
    return root / "result"


@pytest.fixture
def bundle(saved_report_run, tmp_path):
    return shutil.copytree(saved_report_run, tmp_path / "result")


def rehash(root):
    path = root / "bundle.json"
    metadata = json.loads(path.read_text())
    metadata["files"] = {name: file_hash(root / name) for name in metadata["files"]}
    path.write_text(json.dumps(metadata))


def test_report_persisted_only_complete_figures_and_zero_preservation(bundle, monkeypatch):
    from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector
    from pgvamp_ofdm.algorithms.pg_vamp.model import PGVAMPDetector
    from pgvamp_ofdm.data.dataset import EffectiveDataset

    def forbidden(*args, **kwargs):
        pytest.fail("report executed detection, data replay or training")

    for model in (MMSEDetector, VAMPDetector, PGVAMPDetector):
        monkeypatch.setattr(model, "detect", forbidden)
    monkeypatch.setattr(EffectiveDataset, "__getitem__", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    before = {name: (bundle / name).read_bytes() for name in load_bundle(bundle)["files"]}
    result = report([bundle])
    expected = {
        "ber_identity_awgn.png",
        "ser_identity_awgn.png",
        "nmse_identity_awgn.png",
        "latency.png",
        "stability.png",
        "learned_thresholds.png",
        "learned_mu.png",
        "effective_edges.png",
        "channel_ici_example.png",
        "waveform_and_sync_check.png",
    }
    assert set(result["figures"]) == expected
    assert all((bundle / "figures" / name).stat().st_size > 1000 for name in expected)
    assert all((bundle / name).read_bytes() == value for name, value in before.items())
    rows = csv_read(bundle / "aggregate_metrics.csv")
    assert any(float(r["ber"]) == 0 for r in rows)
    text = (bundle / "REPORT.md").read_text(encoding="utf-8")
    assert "未执行" in text and "区间不稳定" in text
    assert "Distinct trained seeds: 0" in text and result["seed_summary"] == []
    assert "A minus B" in text and "conditional-success BER" in text
    load_bundle(bundle)


def test_report_hash_tampering_rejected_before_output(bundle, tmp_path):
    (bundle / "aggregate_metrics.csv").write_text("modified")
    with pytest.raises(ValueError, match="hash"):
        report([bundle], tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("ber", 0.25),
        ("n_bits", 6399),
        ("error_energy", 123),
        ("hard_failures", 1),
        ("message_opportunities", 123),
    ],
)
def test_report_semantic_tampering_with_new_hash_rejected(bundle, field, value):
    path = bundle / "aggregate_metrics.csv"
    rows = csv_read(path)
    rows[-1][field] = value
    csv_write(path, rows)
    rehash(bundle)
    with pytest.raises(ValueError):
        read_run(bundle)


def test_lineage_missing_failure_and_example_tampering(bundle):
    path = bundle / "per_frame_metrics.csv"
    rows = csv_read(path)
    rows[0]["input_hashes"] = json.dumps(["bad"] * 8)
    csv_write(path, rows)
    rehash(bundle)
    with pytest.raises(ValueError, match="lineage"):
        read_run(bundle)


def _trained(run, seed):
    result = copy.deepcopy(run)
    result["bundle"]["run_id"] = f"seed-{seed}"
    result["checkpoint"].update(status="trained", train_seed=seed, checkpoint_hash=f"hash-{seed}")
    return result


def test_merge_requires_distinct_seed_compatible_lineage_and_baseline(bundle):
    a, b = _trained(read_run(bundle), 1), _trained(read_run(bundle), 2)
    validate_merge([a, b])
    bad = copy.deepcopy(b)
    bad["checkpoint"]["train_seed"] = 1
    with pytest.raises(ValueError, match="distinct"):
        validate_merge([a, bad])
    bad = copy.deepcopy(b)
    bad["bundle"]["input_lineage_hash"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        validate_merge([a, bad])
    bad = copy.deepcopy(b)
    bad["bundle"]["compatibility"]["runtime"]["cpu_threads"] = 17
    with pytest.raises(ValueError, match="incompatible"):
        validate_merge([a, bad])
    bad = copy.deepcopy(b)
    bad["aggregate"][0]["bit_errors"] = "99"
    with pytest.raises(ValueError, match="baseline"):
        validate_merge([a, bad])


def test_seed_summary_never_pseudoreplicates_baselines(bundle):
    from pgvamp_ofdm.reporting.report import _seed_summary

    a, b = _trained(read_run(bundle), 1), _trained(read_run(bundle), 2)
    b["aggregate"][-1]["ber"] = 0.2
    a["aggregate"][-1]["ber"] = 0.1
    summary = _seed_summary([a, b])
    ber = next(r for r in summary if r["metric"] == "ber")
    assert ber["mean"] == pytest.approx(0.15)
    assert ber["sample_std"] == pytest.approx(0.1 / 2**0.5)
    assert all(r["algorithm"].startswith("PG") and r["train_seeds"] == 2 for r in summary)
    assert _seed_summary([a]) == []


def test_incomplete_failure_report_retains_denominator(bundle, tmp_path):
    from pgvamp_ofdm.evaluation.metrics import rates

    for filename in ("per_frame_metrics.csv", "aggregate_metrics.csv"):
        path = bundle / filename
        rows = csv_read(path)
        row = rows[0]
        for key in (
            "n_frames",
            "n_blocks",
            "n_bits",
            "n_symbols",
            "bit_errors",
            "symbol_errors",
            "block_errors",
            "frame_errors",
            "successful_blocks",
            "hard_failures",
        ):
            row[key] = int(row[key])
        for key in ("error_energy", "target_energy"):
            row[key] = float(row[key])
        row.update(successful_blocks=7, hard_failures=1)
        row.update(rates(row))
        for key in ("ber_ci_low", "ber_ci_high", "ser_ci_low", "ser_ci_high"):
            row[key] = None
        csv_write(path, rows)
    path = bundle / "diagnostics.jsonl"
    diagnostics = [json.loads(line) for line in path.read_text().splitlines()]
    failure = diagnostics[0]
    failure.update(
        status="incomplete_or_failed",
        stage="detection",
        exception="FloatingPointError",
        reason="injected failure",
        message_opportunities=None,
        no_information_opportunities=None,
    )
    path.write_text("\n".join(json.dumps(r) for r in diagnostics) + "\n")
    (bundle / "failures.jsonl").write_text(json.dumps(failure) + "\n")
    path = bundle / "paired_comparisons.csv"
    csv_write(
        path,
        [
            r
            for r in csv_read(path)
            if not (r["algorithm_a"].startswith("MMSE") or r["algorithm_b"].startswith("MMSE"))
        ],
    )
    path = bundle / "bundle.json"
    metadata = json.loads(path.read_text())
    metadata["status"] = "incomplete_or_failed"
    path.write_text(json.dumps(metadata))
    rehash(bundle)
    result = report([bundle], tmp_path / "failed-report")
    assert result["status"] == "incomplete_or_failed"
    text = (tmp_path / "failed-report" / "REPORT.md").read_text(encoding="utf-8")
    assert "injected failure" in text and "6400" in text and "5600" in text


def test_safe_output_and_explicit_multi_run_destination(bundle, tmp_path):
    with pytest.raises(ValueError, match="explicit output"):
        report([bundle, tmp_path / "other"])
    with pytest.raises(ValueError, match="duplicate"):
        report([bundle, bundle], tmp_path / "merged")
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "user.txt").write_text("preserve")
    with pytest.raises(ValueError, match="exists"):
        report([bundle], target)
    assert (target / "user.txt").read_text() == "preserve"


def test_diagnostics_are_streamed_into_bounded_layer_summaries(bundle, monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def bounded(self, *args, **kwargs):
        if self.name == "diagnostics.jsonl":
            pytest.fail("large diagnostic JSONL must not be read into one string")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", bounded)
    run = read_run(bundle)
    assert len(run["diagnostics"]) == 1
    assert len(run["diagnostics"][0]["layer_summaries"]) == 8
    assert run["layer_ranges"]["effective_all_ratio"]["min"] >= 0
