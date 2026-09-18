"""Versioned atomic result publication and read-only integrity validation."""

import csv
import json
from pathlib import Path
from typing import Any

from ..data.manifest import file_hash

VERSION = 1


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] = ()) -> None:
    keys = list(dict.fromkeys([*fields, *(k for r in rows for k in r)]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def begin(output: Path) -> Path:
    pending = output.with_name(output.name + ".incomplete")
    if output.exists() or pending.exists():
        raise ValueError("result output/pending already exists; choose a fresh directory")
    pending.mkdir(parents=True)
    json_write(pending / "bundle.json", {"schema_version": VERSION, "status": "incomplete"})
    return pending


def publish(pending: Path, output: Path, metadata: dict[str, Any]) -> None:
    hashes = {
        p.relative_to(pending).as_posix(): file_hash(p)
        for p in sorted(pending.rglob("*"))
        if p.is_file() and p.name != "bundle.json"
    }
    json_write(pending / "bundle.json", {"schema_version": VERSION, **metadata, "files": hashes})
    if output.exists():
        raise ValueError("refusing to overwrite another run")
    pending.rename(output)


def load_bundle(root: Path) -> dict[str, Any]:
    value = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if value.get("schema_version") != VERSION or value.get("status") not in (
        "complete",
        "incomplete_or_failed",
    ):
        raise ValueError("unsupported or unpublished results bundle")
    required = {
        "resolved_config.yaml",
        "environment.json",
        "dataset_manifest_hashes.json",
        "checkpoint_metadata.json",
        "per_frame_metrics.csv",
        "aggregate_metrics.csv",
        "timing.csv",
        "diagnostics.jsonl",
        "lineage.jsonl",
        "failures.jsonl",
        "paired_comparisons.csv",
    }
    if not required <= value.get("files", {}).keys():
        raise ValueError("missing result artifacts")
    for name, checksum in value["files"].items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("unsafe artifact path")
        if not path.is_file() or file_hash(path) != checksum:
            raise ValueError(f"result artifact hash mismatch: {name}")
    return value
