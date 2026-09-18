"""Restricted, versioned checkpoints and complete CPU-safe random state."""

import copy
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..algorithms import PGVAMPDetector
from ..config import Config, config_from_values
from ..data.manifest import load_tensors, save_tensors
from ..data.records import allocation_metadata
from ..utils.device import Runtime
from ..utils.random import stable_hash

QPSK_MAPPING = "class=2*b_real+b_imag; x=((1-2*b_real)+j*(1-2*b_imag))/sqrt(2)"
MODEL_KEYS = (
    "depth",
    "rho_hi_db",
    "rho_lo_db",
    "min_gap_db",
    "temperature_db",
    "init_mu",
    "jitter",
    "mask_mode",
)


def model_for(config: Config, runtime: Runtime) -> PGVAMPDetector:
    settings = config.values["pg_vamp"]
    return PGVAMPDetector(
        **{k: settings[k] for k in MODEL_KEYS}, dtype=runtime.real_dtype, device=runtime.device
    )


def physical_contract(config: Config) -> dict[str, Any]:
    return {k: config.values[k] for k in ("waveform", "frame", "receiver", "noise")}


def resume_contract(config: Config) -> dict[str, Any]:
    values = config.values
    return {
        "physics": physical_contract(config),
        "pg_vamp": values["pg_vamp"],
        "training": {k: v for k, v in values["training"].items() if k != "max_steps"},
        "dtype": values["runtime"]["dtype"],
        "seed": values["seed"],
        "deterministic": values["runtime"]["deterministic"],
        "num_workers": values["runtime"]["num_workers"],
    }


def rng_state(device: torch.device) -> dict[str, Any]:
    ns: Any = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.random.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _safe(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ValueError("checkpoint contains nonfinite tensor")
    elif isinstance(value, dict):
        for k, v in value.items():
            if type(k) not in (str, int):
                raise ValueError("checkpoint keys must be strings/integers")
            _safe(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _safe(v)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("checkpoint contains unsupported type")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("checkpoint contains nonfinite number")


def validate_checkpoint(value: Any) -> dict[str, Any]:
    """Validate before mutating model, optimizer or global RNG state."""
    required = {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "resolved_config",
        "algorithm_config",
        "waveform_config",
        "subcarrier_mapping",
        "qpsk_mapping",
        "step",
        "epoch",
        "sampler_position",
        "sampler",
        "random_seed",
        "rng_state",
        "best_validation_metric",
        "best_step",
        "bad_validation_events",
        "last_validation_step",
        "training_manifest_hash",
        "validation_manifest_hash",
        "dtype",
        "device_used_for_training",
        "mask_mode",
        "environment",
        "run_id",
        "contract_hash",
        "resume_validation_state",
        "execution_kind",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("checkpoint missing/unknown fields")
    _safe(value)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema_version")
    if value["execution_kind"] not in ("physical", "algebra_fixture"):
        raise ValueError("invalid execution kind")
    config = config_from_values(value["resolved_config"])
    checks = {
        "algorithm_config": config.values["pg_vamp"],
        "waveform_config": physical_contract(config),
        "subcarrier_mapping": allocation_metadata(config),
        "qpsk_mapping": QPSK_MAPPING,
        "dtype": config.values["runtime"]["dtype"],
        "mask_mode": "soft",
        "random_seed": config.values["seed"],
        "contract_hash": stable_hash(resume_contract(config)),
    }
    for key, expected in checks.items():
        if value[key] != expected:
            raise ValueError(f"checkpoint {key} mismatch")
    for key in ("training_manifest_hash", "validation_manifest_hash", "run_id", "contract_hash"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValueError(f"invalid checkpoint {key}")
    for key in (
        "step",
        "epoch",
        "sampler_position",
        "best_step",
        "bad_validation_events",
        "last_validation_step",
    ):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"invalid checkpoint {key}")
    if not 0 <= value["best_step"] <= value["last_validation_step"] <= value["step"]:
        raise ValueError("checkpoint validation progress mismatch")
    resume_state = value["resume_validation_state"]
    if not isinstance(resume_state, dict) or set(resume_state) != {
        "best_validation_metric",
        "best_step",
        "bad_validation_events",
        "last_validation_step",
    }:
        raise ValueError("invalid resume validation state")
    for key in ("best_step", "bad_validation_events", "last_validation_step"):
        if type(resume_state[key]) is not int or resume_state[key] < 0:
            raise ValueError("invalid resume validation counter")
    if not 0 <= resume_state["best_step"] <= resume_state["last_validation_step"] <= value["step"]:
        raise ValueError("invalid resume validation progress")
    if resume_state["best_validation_metric"] is not None and (
        type(resume_state["best_validation_metric"]) not in (float, int)
        or resume_state["best_validation_metric"] < 0
    ):
        raise ValueError("invalid resume validation metric")
    metric = value["best_validation_metric"]
    if metric is not None and (type(metric) not in (float, int) or metric < 0):
        raise ValueError("invalid best validation metric")
    if not isinstance(value["environment"], dict) or not all(
        k in value["environment"]
        for k in ("python", "torch", "numpy", "package_source_sha256", "engineering_spec_sha256")
    ):
        raise ValueError("checkpoint environment missing")
    if (
        not isinstance(value["device_used_for_training"], str)
        or re.fullmatch(r"cpu|cuda(?::\d+)?", value["device_used_for_training"]) is None
    ):
        raise ValueError("invalid training device")
    rd = torch.float64 if value["dtype"] == "complex128" else torch.float32
    depth = config.values["pg_vamp"]["depth"]
    state = value["model_state_dict"]
    if not isinstance(state, dict) or set(state) != {"raw_gaps", "raw_mu"}:
        raise ValueError("checkpoint must contain exactly 2T parameters")
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (depth,) or tensor.dtype != rd:
            raise ValueError("checkpoint parameter shape/dtype mismatch")
    sampler = value["sampler"]
    if not isinstance(sampler, dict) or set(sampler) != {"permutation", "generator_state", "size"}:
        raise ValueError("invalid sampler fields")
    size, permutation = sampler["size"], sampler["permutation"]
    if (
        type(size) is not int
        or size < 1
        or not isinstance(permutation, torch.Tensor)
        or permutation.dtype != torch.int64
        or permutation.shape != (size,)
        or not torch.equal(permutation.sort().values, torch.arange(size))
    ):
        raise ValueError("invalid sampler permutation")
    if not 0 <= value["sampler_position"] <= size:
        raise ValueError("invalid sampler cursor")
    try:
        torch.Generator().set_state(sampler["generator_state"])
        rs = value["rng_state"]
        if set(rs) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("invalid RNG fields")
        random.Random().setstate(rs["python"])
        ns = rs["numpy"]
        np.random.RandomState().set_state(
            (ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
        )
        torch.Generator().set_state(rs["torch"])
        if not isinstance(rs["cuda"], list) or any(
            not isinstance(x, torch.Tensor) or x.dtype != torch.uint8 or x.ndim != 1
            for x in rs["cuda"]
        ):
            raise ValueError("invalid CUDA RNG states")
    except (KeyError, TypeError, RuntimeError, IndexError) as exc:
        raise ValueError(f"invalid RNG state: {exc}") from exc
    optimizer = value["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("invalid optimizer fields")
    groups = optimizer["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get("params") != [0, 1]:
        raise ValueError("invalid optimizer parameter groups")
    training = config.values["training"]
    if (
        groups[0].get("lr") != training["learning_rate"]
        or groups[0].get("weight_decay") != training["weight_decay"]
        or groups[0].get("betas") != (0.9, 0.999)
        or groups[0].get("eps") != 1e-8
    ):
        raise ValueError("optimizer settings mismatch")
    if set(optimizer["state"]) != ({0, 1} if value["step"] else set()):
        raise ValueError("optimizer state missing")
    for key, expected in {
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }.items():
        if groups[0].get(key) != expected:
            raise ValueError(f"optimizer {key} mismatch")
    if groups[0].get("decoupled_weight_decay", False) is not False:
        raise ValueError("optimizer decoupled_weight_decay mismatch")
    for item in optimizer["state"].values():
        if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid Adam state")
        for key in ("exp_avg", "exp_avg_sq"):
            t = item[key]
            if not isinstance(t, torch.Tensor) or t.shape != (depth,) or t.dtype != rd:
                raise ValueError("optimizer tensor shape/dtype mismatch")
        if bool((item["exp_avg_sq"] < 0).any()):
            raise ValueError("negative Adam second moment")
        if (
            not isinstance(item["step"], torch.Tensor)
            or item["step"].numel() != 1
            or item["step"].item() != value["step"]
        ):
            raise ValueError("optimizer step mismatch")
    return value


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return validate_checkpoint(load_tensors(path))
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"malformed checkpoint: {exc}") from exc


def save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    validate_checkpoint(value)

    # Saving CPU tensors also permits safe loading on hosts without CUDA.
    def cpu(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().clone()
        if isinstance(v, dict):
            return {k: cpu(x) for k, x in v.items()}
        if isinstance(v, list):
            return [cpu(x) for x in v]
        if isinstance(v, tuple):
            return tuple(cpu(x) for x in v)
        return copy.deepcopy(v)

    save_tensors(path, cpu(value))
