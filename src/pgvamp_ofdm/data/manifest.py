"""Versioned dataset integrity, provenance, counts and split lineage."""

import json
import pickle
from pathlib import Path
from typing import Any

import torch

from ..config import Config, config_from_values
from ..utils.random import SEED_VERSION, stable_hash
from .records import (
    GENERATOR_VERSION,
    MODEL_VERSION,
    SPLITS,
    allocation_metadata,
    tensor_hash,
    validate_record,
    waveform_hash,
)


def file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_tensors(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_tensors(path: str | Path) -> Any:
    """Restricted CPU load, with an actionable error instead of unsafe pickle fallback."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
        raise ValueError(f"invalid or unsupported tensor artifact: {path}: {exc}") from exc


def counts(records: list[dict[str, Any]], blocks: int) -> dict[str, Any]:
    return {
        split: {
            "independent_frames": len({r["frame_id"] for r in records if r["split"] == split}),
            "independent_channels": len({r["channel_id"] for r in records if r["split"] == split}),
            "frame_copies": sum(r["split"] == split for r in records),
            "samples": blocks * sum(r["split"] == split for r in records),
        }
        for split in SPLITS
    }


def split_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        split: sorted({(r["frame_id"], r["channel_id"]) for r in records if r["split"] == split})
        for split in SPLITS
    }


def distribution(config: Config) -> dict[str, Any]:
    c = config.values
    return {
        "training_weights": c["data"]["train_scenario_weights"],
        "train_val_esn0_uniform_db": c["data"]["train_esn0_range_db"],
        "test_scenarios": c["evaluation"]["scenarios"],
        "test_esn0_db": c["evaluation"]["esn0_db"],
        "test_frames_per_cell": c["evaluation"]["frames_per_cell"],
        "snr_pairing": "same physical frame, bits and pilots; independent noise SNR subseeds",
        "split_rule": "global split-stream permutation before channel sampling or SNR expansion",
        "label": c["evaluation"]["label"],
    }


def validate_lineage(records: list[dict[str, Any]], config: Config) -> None:
    frame_split: dict[str, str] = {}
    channel_split: dict[str, str] = {}
    copies: set[tuple[str, str]] = set()
    physical: dict[str, str] = {}
    channel_owner: dict[str, str] = {}
    all_noise: set[int] = set()
    for record in records:
        split, frame, channel = record["split"], record["frame_id"], record["channel_id"]
        for key, table in ((frame, frame_split), (channel, channel_split)):
            if table.setdefault(key, split) != split:
                raise ValueError("frame/channel split leakage")
        if channel_owner.setdefault(channel, frame) != frame:
            raise ValueError("one channel instance cannot be renamed into multiple frames")
        key_copy = (frame, record["snr_copy"])
        if key_copy in copies:
            raise ValueError("duplicate frame/SNR copy")
        copies.add(key_copy)
        excluded = {"esn0_db", "noise_seed_real", "noise_seed_imag", "snr_copy"}
        fingerprint = tensor_hash({k: v for k, v in record.items() if k not in excluded})
        if physical.setdefault(frame, fingerprint) != fingerprint:
            raise ValueError("paired physical frame payload differs across SNR")
        seeds = record["noise_seed_real"].tolist() + record["noise_seed_imag"].tolist()
        if all_noise.intersection(seeds):
            raise ValueError("reused noise seed across frame/SNR copies")
        all_noise.update(seeds)
    c = config.values
    for split in ("train", "val"):
        subset = [r for r in records if r["split"] == split]
        if len(subset) != c["data"][f"{split}_frames"] or len(
            {r["frame_id"] for r in subset}
        ) != len(subset):
            raise ValueError(f"{split} frame count mismatch")
        low, high = c["data"]["train_esn0_range_db"]
        for r in subset:
            if c["data"]["train_scenario_weights"].get(r["scenario"], 0) <= 0:
                raise ValueError("train/val scenario outside configured mixture")
            if not bool(((r["esn0_db"] >= low) & (r["esn0_db"] < high)).all()):
                raise ValueError("train/val SNR outside configured range")
    test = [r for r in records if r["split"] == "test"]
    scenarios, snrs = c["evaluation"]["scenarios"], c["evaluation"]["esn0_db"]
    if len(test) != len(scenarios) * len(snrs) * c["evaluation"]["frames_per_cell"]:
        raise ValueError("test frame copy count mismatch")
    for scenario in scenarios:
        paired: set[str] | None = None
        for snr in snrs:
            selected = [
                r for r in test if r["scenario"] == scenario and float(r["esn0_db"][0]) == snr
            ]
            frames = {r["frame_id"] for r in selected}
            if len(selected) != c["evaluation"]["frames_per_cell"] or len(frames) != len(selected):
                raise ValueError("test scenario/SNR cell count mismatch")
            if paired is not None and frames != paired:
                raise ValueError("test SNR grid must reuse paired physical frames")
            paired = frames


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Config, list[dict[str, Any]]]:
    """Validate all shards on CPU with restricted unpickling before exposing any sample."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("unsupported manifest schema_version")
        config = config_from_values(manifest["resolved_config"])
        if manifest["config_sha256"] != stable_hash(config.values):
            raise ValueError("manifest config hash mismatch")
        expected = {
            "waveform_config_hash": waveform_hash(config),
            "allocation": allocation_metadata(config),
            "generator_version": GENERATOR_VERSION,
            "model_version": MODEL_VERSION,
            "seed_derivation": SEED_VERSION,
            "master_seed": config.values["seed"],
            "distribution": distribution(config),
            "snr_definition": "Es/N0 (dB), unit-energy data QPSK; sigma2=E|w|^2",
            "qpsk_mapping": "class=2*b_real+b_imag; x=((1-2*b_real)+j*(1-2*b_imag))/sqrt(2)",
        }
        for key, value in expected.items():
            if manifest[key] != value:
                raise ValueError(f"manifest {key} mismatch")
        if not isinstance(manifest["environment"], dict):
            raise ValueError("manifest environment missing")
        records: list[dict[str, Any]] = []
        visited: set[Path] = set()
        for shard in manifest["shards"]:
            target = (path.parent / shard["path"]).resolve()
            if not target.is_relative_to(path.parent.resolve()) or target in visited:
                raise ValueError("unsafe or duplicate shard path")
            visited.add(target)
            if file_hash(target) != shard["sha256"]:
                raise ValueError(f"shard checksum mismatch: {shard['path']}")
            loaded = load_tensors(target)
            if not isinstance(loaded, list) or len(loaded) != shard["frame_copies"]:
                raise ValueError("shard record count mismatch")
            for record in loaded:
                validate_record(record, config)
                if record["split"] != shard["split"]:
                    raise ValueError("shard split mismatch")
            records.extend(loaded)
        validate_lineage(records, config)
        if manifest["counts"] != counts(records, config.values["frame"]["n_ofdm_symbols"]):
            raise ValueError("manifest sample counts mismatch")
        if manifest["split_table"] != json.loads(json.dumps(split_table(records))):
            raise ValueError("manifest split table mismatch")
        return manifest, config, records
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"malformed dataset manifest: {exc}") from exc
