"""Budgeted explicit dense export and strict labeled/unlabeled input validation."""

import json
import re
from pathlib import Path
from typing import Any

import torch

from ..config import config_from_values
from ..modulation.qpsk import bits_to_symbols, classes_to_symbols, hard_decision
from ..utils.random import stable_hash
from .dataset import EffectiveDataset
from .manifest import file_hash, load_tensors, save_tensors
from .records import allocation_metadata


def _metadata_reserve(config: dict[str, Any], physical_frames: int) -> int:
    # Every export carries the complete split table, even when exporting a tiny split.
    # Two 64-character IDs per frame plus pickle/container allowance, and full config.
    return physical_frames * 1024 + 2 * len(json.dumps(config).encode("utf-8"))


def estimate_size(
    samples: int, dimension: int, dtype: str, *, labeled: bool = True
) -> dict[str, int]:
    """Exact tensor payload; explicit conservative reserve is not exact serialized size."""
    if any(type(v) is not int or v <= 0 for v in (samples, dimension)):
        raise ValueError("sample count and dimension must be positive integers")
    if dtype not in ("complex128", "complex64") or type(labeled) is not bool:
        raise ValueError("invalid dtype/labeled for size estimate")
    cb = 16 if dtype == "complex128" else 8
    matrix = samples * dimension * dimension * cb
    other = samples * (
        dimension * cb + cb // 2 + (dimension * cb + 2 * dimension if labeled else 0)
    )
    reserve = 1024 * 1024 + samples * 2048
    return {
        "matrix_payload_bytes": matrix,
        "other_tensor_payload_bytes": other,
        "tensor_payload_bytes": matrix + other,
        "serialization_metadata_reserve_bytes": reserve,
        "estimated_output_bytes": matrix + other + reserve,
    }


def validate_materialized(value: Any, *, purpose: str = "inference") -> None:
    """Validate square systems and all optional labels, even during inference."""
    if purpose not in ("inference", "train", "evaluation"):
        raise ValueError("purpose must be inference, train or evaluation")
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        raise ValueError("unsupported materialized schema_version")
    if value.keys() - {"schema_version", "H", "y", "sigma2", "x", "bits", "metadata"}:
        raise ValueError("unknown materialized fields")
    for key in ("H", "y", "sigma2"):
        if not isinstance(value.get(key), torch.Tensor):
            raise ValueError(f"materialized {key} must be tensor")
    h, y, variance = value["H"], value["y"], value["sigma2"]
    if h.ndim != 3 or h.shape[0] < 1 or h.shape[1] < 1 or h.shape[1] != h.shape[2]:
        raise ValueError("H must be nonempty [K,N,N] square systems")
    k, n, _ = h.shape
    if h.dtype not in (torch.complex128, torch.complex64):
        raise ValueError("H must have complex128/complex64 dtype")
    real = torch.float64 if h.dtype == torch.complex128 else torch.float32
    if y.shape != (k, n) or y.dtype != h.dtype or variance.shape != (k,) or variance.dtype != real:
        raise ValueError("y/sigma2 shape or dtype pairing mismatch")
    for tensor in (h, y, variance):
        if tensor.device != h.device or not bool(torch.isfinite(tensor).all()):
            raise ValueError("materialized tensors must be finite on one device")
    if not bool((variance > 0).all()):
        raise ValueError("sigma2 must be positive")
    if purpose != "inference" and not all(key in value for key in ("x", "bits")):
        raise ValueError(f"{purpose} requires x and bits labels")
    if "x" in value:
        x = value["x"]
        if (
            not isinstance(x, torch.Tensor)
            or x.shape != (k, n)
            or x.dtype != h.dtype
            or x.device != h.device
        ):
            raise ValueError("x shape/dtype/device mismatch")
        expected = classes_to_symbols(hard_decision(x), dtype=h.dtype)
        if not torch.allclose(
            x, expected, atol=1e-14 if h.dtype == torch.complex128 else 1e-6, rtol=0
        ):
            raise ValueError("x must contain unit QPSK symbols")
    if "bits" in value:
        bits = value["bits"]
        if (
            not isinstance(bits, torch.Tensor)
            or bits.shape != (k, n, 2)
            or bits.dtype != torch.uint8
            or bits.device != h.device
        ):
            raise ValueError("bits shape/dtype/device mismatch")
        labels = bits_to_symbols(bits, dtype=h.dtype)
        if "x" in value and not torch.equal(labels, value["x"]):
            raise ValueError("x and bits labels are inconsistent")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("materialized metadata required")
    try:
        config = config_from_values(metadata["resolved_config"])
        if metadata["config_sha256"] != stable_hash(config.values) or metadata[
            "allocation"
        ] != allocation_metadata(config):
            raise ValueError("materialized configuration or allocation mismatch")
        if (
            n != config.values["waveform"]["n_data"]
            or str(h.dtype) != "torch." + config.values["runtime"]["dtype"]
        ):
            raise ValueError("materialized dimension/dtype differs from config")
        if metadata["split"] not in ("train", "val", "test"):
            raise ValueError("invalid materialized split")
        for key in ("sample_ids", "frame_ids", "channel_ids"):
            ids = metadata[key]
            if (
                not isinstance(ids, list)
                or len(ids) != k
                or not all(isinstance(x, str) and x for x in ids)
            ):
                raise ValueError(f"materialized {key} count/type mismatch")
        if (
            len(set(metadata["sample_ids"])) != k
            or type(metadata["samples"]) is not int
            or metadata["samples"] != k
        ):
            raise ValueError("materialized duplicate sample IDs or sample count mismatch")
        table = metadata["split_table"]
        if not isinstance(table, dict) or set(table) != {"train", "val", "test"}:
            raise ValueError("materialized split table missing")
        frame_splits: dict[str, str] = {}
        channel_splits: dict[str, str] = {}
        selected_pairs: set[tuple[str, str]] = set()
        for split, pairs in table.items():
            if not isinstance(pairs, list):
                raise ValueError("invalid materialized split table")
            for pair in pairs:
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or not all(isinstance(v, str) and v for v in pair)
                ):
                    raise ValueError("invalid materialized frame/channel pair")
                if any(re.fullmatch(r"[0-9a-f]{64}", v) is None for v in pair):
                    raise ValueError("invalid materialized SHA-256 frame/channel identity")
                frame, channel = pair
                if frame in frame_splits or channel in channel_splits:
                    raise ValueError("materialized split overlap or duplicate lineage")
                frame_splits[frame], channel_splits[channel] = split, split
                if split == metadata["split"]:
                    selected_pairs.add((frame, channel))
        for sample, frame, channel in zip(
            metadata["sample_ids"], metadata["frame_ids"], metadata["channel_ids"]
        ):
            if (frame, channel) not in selected_pairs or not sample.startswith(frame + ":"):
                raise ValueError("materialized IDs disagree with split lineage")
            suffix = sample[len(frame) + 1 :].split(":")
            copies = (
                {float(snr).hex() for snr in config.values["evaluation"]["esn0_db"]}
                if metadata["split"] == "test"
                else {"continuous"}
            )
            if (
                len(suffix) != 2
                or suffix[0] not in copies
                or suffix[1]
                not in {str(b) for b in range(config.values["frame"]["n_ofdm_symbols"])}
            ):
                raise ValueError("materialized sample ID has invalid SNR copy or block")
        if (
            not isinstance(metadata["source_manifest_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", metadata["source_manifest_sha256"]) is None
        ):
            raise ValueError("invalid source manifest hash")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed materialized metadata: {exc}") from exc


def load_materialized(path: str | Path, *, purpose: str = "inference") -> dict[str, Any]:
    value: dict[str, Any] = load_tensors(path)
    validate_materialized(value, purpose=purpose)
    return value


def materialize(
    manifest: str | Path,
    split: str,
    output: str | Path,
    *,
    max_output_bytes: int | None = None,
    allow_large_output: bool = False,
    labeled: bool = True,
) -> dict[str, int]:
    dataset = EffectiveDataset(manifest, split, cache_entries=0)
    budget = (
        dataset.config.values["data"]["max_output_bytes"]
        if max_output_bytes is None
        else max_output_bytes
    )
    if type(budget) is not int or budget <= 0:
        raise ValueError("max_output_bytes must be positive integer")
    estimate = estimate_size(
        len(dataset), 400, dataset.config.values["runtime"]["dtype"], labeled=labeled
    )
    extra = _metadata_reserve(
        dataset.config.values, sum(len(pairs) for pairs in dataset.manifest["split_table"].values())
    )
    extra = max(
        extra,
        2 * len(json.dumps(dataset.manifest["split_table"]).encode("utf-8"))
        + 2 * len(json.dumps(dataset.config.values).encode("utf-8")),
    )
    estimate["serialization_metadata_reserve_bytes"] += extra
    estimate["estimated_output_bytes"] += extra
    # This guard must precede __getitem__, allocation of dense tensors and output creation.
    if estimate["estimated_output_bytes"] > budget and not allow_large_output:
        raise ValueError(f"output estimate {estimate} exceeds {budget}; use --allow-large-output")
    output = Path(output)
    if output.exists():
        raise ValueError("materialized output already exists")
    k, cd, rd = len(dataset), dataset.runtime.complex_dtype, dataset.runtime.real_dtype
    value: dict[str, Any] = {
        "schema_version": 1,
        "H": torch.empty((k, 400, 400), dtype=cd),
        "y": torch.empty((k, 400), dtype=cd),
        "sigma2": torch.empty(k, dtype=rd),
    }
    if labeled:
        value.update(
            x=torch.empty((k, 400), dtype=cd), bits=torch.empty((k, 400, 2), dtype=torch.uint8)
        )
    metadata = {
        "samples": k,
        "sample_ids": [],
        "frame_ids": [],
        "channel_ids": [],
        "split": split,
        "source_manifest_sha256": file_hash(Path(manifest)),
        "resolved_config": dataset.config.values,
        "config_sha256": dataset.manifest["config_sha256"],
        "allocation": dataset.manifest["allocation"],
        "split_table": dataset.manifest["split_table"],
        "capacity_estimate": estimate,
    }
    value["metadata"] = metadata
    for index in range(k):
        sample = dataset[index]
        for key in ("H", "y", "sigma2", "x", "bits"):
            if key in value:
                value[key][index] = sample[key]
        for key in ("sample", "frame", "channel"):
            metadata[f"{key}_ids"].append(sample[f"{key}_id"])
    validate_materialized(value, purpose="evaluation" if labeled else "inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tensors(output, value)
    return {**estimate, "actual_file_bytes": output.stat().st_size}
