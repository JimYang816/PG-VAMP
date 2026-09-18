"""WP8 independent persisted evaluation audit, reused from archived WP7."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(root):
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["kind"] == "evaluation"
    for name, expected in bundle["files"].items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    frames = read_csv(root / "per_frame_metrics.csv")
    aggregates = read_csv(root / "aggregate_metrics.csv")
    lineage = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    identities = {row["sample_id"]: row for row in lineage}
    assert len(identities) == len(lineage) == bundle["planned_samples"]
    grouped = defaultdict(list)
    frame_inputs = {}
    count_keys = ("n_frames", "n_blocks", "n_bits", "n_symbols", "bit_errors",
                  "symbol_errors", "block_errors", "frame_errors", "successful_blocks",
                  "hard_failures")
    for frame in frames:
        key = (frame["algorithm"], frame["scenario"], frame["esn0_db"])
        grouped[key].append(frame)
        ids, hashes = json.loads(frame["sample_ids"]), json.loads(frame["input_hashes"])
        assert len(ids) == len(set(ids)) == len(hashes) == 8
        assert sorted(identities[s]["block"] for s in ids) == list(range(8))
        assert all(identities[s]["input_hash"] == h for s, h in zip(ids, hashes))
        assert all(frame["algorithm"] in identities[s]["algorithms"] for s in ids)
        pairing = (frame["scenario"], frame["esn0_db"], frame["frame_id"])
        assert frame_inputs.setdefault(pairing, (ids, hashes)) == (ids, hashes)
        assert int(frame["n_bits"]) == 6400 and int(frame["n_symbols"]) == 3200
        assert int(frame["n_blocks"]) == 8 and int(frame["n_frames"]) == 1
    summaries = []
    for row in aggregates:
        key = (row["algorithm"], row["scenario"], row["esn0_db"])
        selected = grouped.pop(key)
        for name in count_keys:
            assert int(row[name]) == sum(int(r[name]) for r in selected), (key, name)
        for name in ("error_energy", "target_energy"):
            assert math.isclose(float(row[name]), sum(float(r[name]) for r in selected),
                                rel_tol=1e-12, abs_tol=1e-12), (key, name)
        if int(row["hard_failures"]):
            assert row["status"] == "incomplete_or_failed"
            assert all(row[name] == "" for name in ("ber", "ser", "bler", "fer", "nmse_linear"))
        else:
            assert row["status"] == "complete"
            for metric, numerator, denominator in (
                ("ber", "bit_errors", "n_bits"), ("ser", "symbol_errors", "n_symbols"),
                ("bler", "block_errors", "n_blocks"), ("fer", "frame_errors", "n_frames"),
                ("nmse_linear", "error_energy", "target_energy")):
                assert math.isclose(float(row[metric]), float(row[numerator]) / float(row[denominator]),
                                    rel_tol=1e-12, abs_tol=1e-12), (key, metric)
            assert math.isclose(float(row["evm_pct"]), 100 * math.sqrt(float(row["nmse_linear"])), rel_tol=1e-12)
            assert math.isclose(float(row["goodput_bps"]), 6400 / .956 * (1-float(row["fer"])), rel_tol=1e-12, abs_tol=1e-12)
        summaries.append({k: row[k] for k in ("algorithm", "scenario", "esn0_db", "n_frames",
                                             "n_bits", "bit_errors", "status")})
    assert not grouped
    return {"root": str(root.resolve()), "status": "pass", "files_rehashed": len(bundle["files"]),
            "observed_block_ids": len(lineage), "frame_rows": len(frames), "cells": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [audit(root) for root in args.roots]
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
