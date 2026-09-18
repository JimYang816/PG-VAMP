"""Independent semantic checks at the persisted-results boundary."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..evaluation.artifacts import csv_read, load_bundle
from ..utils.random import stable_hash

COUNTS = (
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
)
ENERGIES = ("error_energy", "target_energy")


def number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite report value: {key}")
    return result


def integer(row: dict[str, Any], key: str) -> int:
    value = number(row, key)
    if value is None or value < 0 or value != int(value):
        raise ValueError(f"invalid count: {key}")
    return int(value)


def same(actual: Any, expected: float | None, key: str) -> None:
    value = number({key: actual}, key)
    if (value is None) != (expected is None) or (
        value is not None
        and expected is not None
        and not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise ValueError(f"inconsistent {key}")


def json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cell(row: dict[str, Any]) -> tuple[str, str, float]:
    return row["algorithm"], row["scenario"], float(row["esn0_db"])


def _stream_diagnostics(
    path: Path, samples: dict[str, dict[str, Any]], expected: set[tuple[str, str]]
) -> tuple[dict[Any, Any], list[dict[str, Any]], dict[str, Any], set[tuple[str, str]]]:
    """Retain O(cells * layers) summaries, never the large per-block JSONL payload."""
    totals: dict[Any, Any] = defaultdict(lambda: defaultdict(int))
    moments: dict[Any, Any] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0, math.inf, -math.inf])
    )
    seen, failed = set(), set()
    fields = ("effective_candidate_ratio", "effective_all_ratio", "safety_relative", "c")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pair = record["algorithm"], record["sample_id"]
            if pair in seen or pair not in expected:
                raise ValueError("invalid diagnostics sample identities")
            seen.add(pair)
            shared = samples[record["sample_id"]]
            if any(
                record[k] != shared[k]
                for k in ("input_hash", "frame_id", "scenario", "esn0_db", "block")
            ):
                raise ValueError("diagnostics lineage mismatch")
            if record["status"] == "incomplete_or_failed":
                failed.add(pair)
                continue
            if record["status"] != "complete":
                raise ValueError("invalid diagnostic status")
            target = totals[cell(record)]
            for key in ("message_opportunities", "no_information_opportunities"):
                target[key] += integer(record, key)
            for key in ("message_rejected", "precision_capped", "no_information"):
                target[key] += sum(record.get(key, [0]))
            for index, layer in enumerate(record.get("layer_summaries", [])):
                for field in fields:
                    values = layer[field] if isinstance(layer[field], list) else [layer[field]]
                    for value in values:
                        scalar = float(value)
                        # A finite unresolved contraction c may be slightly
                        # negative within its numerical tolerance (WP5).
                        if not math.isfinite(scalar) or (field != "c" and scalar < 0):
                            raise ValueError("invalid layer summary")
                        if field.startswith("effective_") and scalar > 1:
                            raise ValueError("invalid edge ratio")
                        stats = moments[index][field]
                        stats[0] += scalar
                        stats[1] += 1
                        stats[2], stats[3] = min(stats[2], scalar), max(stats[3], scalar)
    if seen != expected:
        raise ValueError("missing diagnostics")
    layers = [
        {field: [stats[0] / stats[1]] for field, stats in moments[index].items()}
        for index in sorted(moments)
    ]
    ranges = {}
    for field in fields:
        values = [moments[index][field] for index in moments]
        if values:
            ranges[field] = {
                "min": min(v[2] for v in values),
                "max": max(v[3] for v in values),
                "mean": sum(v[0] for v in values) / sum(v[1] for v in values),
            }
    summaries = [{"status": "complete", "layer_summaries": layers}] if layers else []
    return totals, summaries, ranges, failed


def _rates(row: dict[str, Any]) -> None:
    counts = {key: integer(row, key) for key in COUNTS}
    if not all(counts[k] > 0 for k in ("n_frames", "n_blocks", "n_bits", "n_symbols")):
        raise ValueError("empty metric population")
    if counts["successful_blocks"] + counts["hard_failures"] != counts["n_blocks"]:
        raise ValueError("failure denominator mismatch")
    if counts["n_blocks"] != 8 * counts["n_frames"] or counts["n_bits"] != 2 * counts["n_symbols"]:
        raise ValueError("incomplete physical frame denominator")
    symbols = counts["n_symbols"] // counts["n_blocks"]
    if counts["n_symbols"] != symbols * counts["n_blocks"]:
        raise ValueError("nonintegral symbol denominator")
    if not (
        counts["symbol_errors"] <= counts["bit_errors"] <= 2 * counts["symbol_errors"]
        and counts["symbol_errors"] <= counts["successful_blocks"] * symbols
        and counts["block_errors"] <= counts["successful_blocks"]
        and counts["frame_errors"] <= min(counts["n_frames"], counts["block_errors"])
        and counts["block_errors"] <= counts["symbol_errors"]
    ):
        raise ValueError("impossible error counts")
    complete = counts["hard_failures"] == 0
    if row["status"] != ("complete" if complete else "incomplete_or_failed"):
        raise ValueError("metric status conceals failures")
    for rate, numerator, denominator in (
        ("ber", "bit_errors", "n_bits"),
        ("ser", "symbol_errors", "n_symbols"),
        ("bler", "block_errors", "n_blocks"),
        ("fer", "frame_errors", "n_frames"),
    ):
        same(row.get(rate), counts[numerator] / counts[denominator] if complete else None, rate)
    error, target = (number(row, k) for k in ENERGIES)
    if error is None or target is None or error < 0 or target < 0:
        raise ValueError("invalid energy sums")
    if complete and target <= 0:
        raise ValueError("missing target energy")
    nmse = error / target if complete else None
    same(row.get("nmse_linear"), nmse, "nmse_linear")
    same(row.get("nmse_db"), 10 * math.log10(nmse) if nmse else None, "nmse_db")
    same(row.get("evm_pct"), 100 * math.sqrt(nmse) if nmse is not None else None, "evm_pct")
    same(
        row.get("goodput_bps"),
        6400 / 0.956 * (1 - counts["frame_errors"] / counts["n_frames"]) if complete else None,
        "goodput_bps",
    )
    successful_bits = counts["successful_blocks"] * 2 * symbols
    same(row.get("conditional_success_bits"), successful_bits, "conditional_success_bits")
    same(
        row.get("conditional_success_ber"),
        counts["bit_errors"] / successful_bits if successful_bits else None,
        "conditional_success_ber",
    )
    same(
        row.get("fer_zero_upper95"),
        1 - 0.05 ** (1 / counts["n_frames"]) if complete and counts["frame_errors"] == 0 else None,
        "fer_zero_upper95",
    )


def read_run(root: Path) -> dict[str, Any]:
    """Validate hashes, planned population, paired lineage and count arithmetic."""
    bundle = load_bundle(root)
    if bundle.get("kind") != "evaluation":
        raise ValueError("report requires an evaluation bundle")
    config = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    frames = csv_read(root / "per_frame_metrics.csv")
    aggregates = csv_read(root / "aggregate_metrics.csv")
    lineage = json_lines(root / "lineage.jsonl")
    failures = json_lines(root / "failures.jsonl")
    checkpoint = json.loads((root / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    if not frames or not aggregates or not lineage:
        raise ValueError("empty evaluation results")
    if stable_hash(lineage) != bundle.get("input_lineage_hash"):
        raise ValueError("input lineage digest mismatch")
    if len(lineage) != bundle["planned_samples"]:
        raise ValueError("planned sample count mismatch")
    samples = {r["sample_id"]: r for r in lineage}
    if len(samples) != len(lineage):
        raise ValueError("duplicate sample lineage")
    algorithms = bundle["algorithms"]
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("invalid algorithm list")
    for shared in lineage:
        if set(shared["algorithms"]) != set(algorithms):
            raise ValueError("unpaired algorithm inputs")
        if not re.fullmatch(r"[0-9a-f]{64}", shared["input_hash"]):
            raise ValueError("invalid input hash")
    for row in [*frames, *aggregates]:
        for field, expected in (
            ("run_id", bundle["run_id"]),
            ("manifest_hash", bundle["manifest_hash"]),
            ("test_seed", config["seed"]),
        ):
            if str(row[field]) != str(expected):
                raise ValueError(f"metric provenance mismatch: {field}")
        for field in ("train_seed", "checkpoint_hash"):
            expected = checkpoint.get(field) if row["algorithm"].startswith("PG") else None
            if row.get(field) != (str(expected) if expected is not None else ""):
                raise ValueError(f"checkpoint provenance mismatch: {field}")
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    frame_ids: set[tuple[str, str, float, str]] = set()
    for row in frames:
        _rates(row)
        frame_key = (*cell(row), row["frame_id"])
        if frame_key in frame_ids or integer(row, "n_frames") != 1:
            raise ValueError("duplicate/nonunit frame")
        frame_ids.add(frame_key)
        ids, hashes = json.loads(row["sample_ids"]), json.loads(row["input_hashes"])
        if len(ids) != 8 or len(hashes) != 8 or len(set(ids)) != 8:
            raise ValueError("missing frame block lineage")
        for block, (sample_id, checksum) in enumerate(zip(ids, hashes, strict=True)):
            entry = samples.get(sample_id)
            if entry is None or any(
                (
                    entry["input_hash"] != checksum,
                    entry["frame_id"] != row["frame_id"],
                    entry["scenario"] != row["scenario"],
                    float(entry["esn0_db"]) != float(row["esn0_db"]),
                    entry["block"] != block,
                    row["algorithm"] not in entry["algorithms"],
                )
            ):
                raise ValueError("frame/shared input lineage mismatch")
            pair = row["algorithm"], sample_id
            if pair in seen:
                raise ValueError("repeated algorithm sample")
            seen.add(pair)
        if integer(row, "n_symbols") != config["waveform"]["n_data"] * 8:
            raise ValueError("symbol allocation mismatch")
        groups[cell(row)].append(row)
    expected_pairs = {(a, s) for a in algorithms for s in samples}
    if seen != expected_pairs:
        raise ValueError("missing paired population")
    diagnostic_totals, diagnostics, layer_ranges, diagnostic_failed = _stream_diagnostics(
        root / "diagnostics.jsonl", samples, expected_pairs
    )
    for rows, label in ((failures, "failures"),):
        keys = [(r["algorithm"], r["sample_id"]) for r in rows]
        if len(set(keys)) != len(keys) or not set(keys) <= expected_pairs:
            raise ValueError(f"invalid {label} sample identities")
        for record in rows:
            shared = samples[record["sample_id"]]
            if any(
                record[field] != shared[field]
                for field in ("input_hash", "frame_id", "scenario", "esn0_db", "block")
            ):
                raise ValueError(f"{label} lineage mismatch")
    failed = {(r["algorithm"], r["sample_id"]) for r in failures}
    if failed != diagnostic_failed:
        raise ValueError("failure ledger mismatch")
    for row in frames:
        count = sum((row["algorithm"], s) in failed for s in json.loads(row["sample_ids"]))
        if count != integer(row, "hard_failures"):
            raise ValueError("frame failure count mismatch")
    aggregate_map = {cell(r): r for r in aggregates}
    if len(aggregate_map) != len(aggregates) or set(aggregate_map) != set(groups):
        raise ValueError("aggregate population mismatch")
    plan = config["evaluation"]
    expected_cells = {
        (a, s, float(db)) for a in algorithms for s in plan["scenarios"] for db in plan["esn0_db"]
    }
    if set(groups) != expected_cells:
        raise ValueError("fixed evaluation cells mismatch")
    for key, rows in groups.items():
        row = aggregate_map[key]
        _rates(row)
        if len(rows) != plan["frames_per_cell"]:
            raise ValueError("fixed frame population mismatch")
        for field in (*COUNTS, *ENERGIES):
            same(row[field], sum(float(r[field]) for r in rows), field)
        for field in ("message_opportunities", "no_information_opportunities"):
            same(row[field], diagnostic_totals[key][field], field)
        for field, rate, opportunities in (
            ("message_rejected", "message_reject_rate", "message_opportunities"),
            ("precision_capped", "precision_cap_rate", "message_opportunities"),
            ("no_information", "no_information_rate", "no_information_opportunities"),
        ):
            total = diagnostic_totals[key][field]
            denom = integer(row, opportunities)
            if total < 0 or total > denom:
                raise ValueError("impossible diagnostic rate")
            same(row[field], total, field)
            same(row[rate], total / denom if denom else None, rate)
        for metric in ("ber", "ser"):
            low, high = number(row, metric + "_ci_low"), number(row, metric + "_ci_high")
            if row["status"] != "complete":
                if low is not None or high is not None:
                    raise ValueError("failed population has confidence interval")
            elif low is None or high is None or not 0 <= low <= high <= 1:
                raise ValueError("invalid confidence interval")
    status = "incomplete_or_failed" if failed else "complete"
    if status != bundle["status"]:
        raise ValueError("bundle status mismatch")
    paired = csv_read(root / "paired_comparisons.csv")
    expected_comparisons = {
        (a, b, scenario, db, metric)
        for i, a in enumerate(algorithms)
        for b in algorithms[i + 1 :]
        for scenario in plan["scenarios"]
        for db in map(float, plan["esn0_db"])
        for metric in ("ber", "ser")
        if aggregate_map[a, scenario, db]["status"] == "complete"
        and aggregate_map[b, scenario, db]["status"] == "complete"
    }
    actual_comparisons = [
        (r["algorithm_a"], r["algorithm_b"], r["scenario"], float(r["esn0_db"]), r["metric"])
        for r in paired
    ]
    if len(set(actual_comparisons)) != len(actual_comparisons) or (
        set(actual_comparisons) != expected_comparisons
    ):
        raise ValueError("missing/duplicate paired comparison")
    for row in paired:
        a = aggregate_map[(row["algorithm_a"], row["scenario"], float(row["esn0_db"]))]
        b = aggregate_map[(row["algorithm_b"], row["scenario"], float(row["esn0_db"]))]
        if (
            a["status"] != "complete"
            or b["status"] != "complete"
            or row["metric"] not in ("ber", "ser")
        ):
            raise ValueError("invalid paired comparison")
        same(
            row["difference_a_minus_b"],
            float(a[row["metric"]]) - float(b[row["metric"]]),
            "paired difference",
        )
        low, high = number(row, "ci_low"), number(row, "ci_high")
        if low is None or high is None or not -1 <= low <= high <= 1:
            raise ValueError("invalid paired interval")
        if integer(row, "n_frames") != integer(a, "n_frames"):
            raise ValueError("paired frame denominator mismatch")
    return {
        "root": root,
        "bundle": bundle,
        "config": config,
        "frames": frames,
        "aggregate": aggregates,
        "lineage": lineage,
        "diagnostics": diagnostics,
        "layer_ranges": layer_ranges,
        "failures": failures,
        "paired": paired,
        "timing": csv_read(root / "timing.csv"),
        "checkpoint": checkpoint,
        "environment": json.loads((root / "environment.json").read_text(encoding="utf-8")),
    }


def validate_merge(runs: list[dict[str, Any]]) -> None:
    """Train seeds vary; fixed test lineage and timing protocol do not."""
    first = runs[0]
    if len({r["bundle"]["run_id"] for r in runs}) != len(runs):
        raise ValueError("duplicate evaluation run")
    seeds, checkpoints = set(), set()
    for run in runs:
        for field in ("compatibility", "input_lineage_hash", "manifest_hash", "algorithms"):
            if run["bundle"][field] != first["bundle"][field]:
                raise ValueError(f"incompatible merged results: {field}")
        checkpoint = run["checkpoint"]
        if len(runs) > 1:
            if checkpoint["status"] != "trained" or checkpoint["train_seed"] in seeds:
                raise ValueError("merge requires distinct trained seeds")
            if checkpoint["checkpoint_hash"] in checkpoints:
                raise ValueError("duplicate trained checkpoint")
        seeds.add(checkpoint.get("train_seed"))
        checkpoints.add(checkpoint.get("checkpoint_hash"))
        for row in run["aggregate"]:
            if not row["algorithm"].startswith("PG"):
                baseline = next(r for r in first["aggregate"] if cell(r) == cell(row))
                for key in (*COUNTS, *ENERGIES):
                    same(row[key], float(baseline[key]), "repeated baseline " + key)
