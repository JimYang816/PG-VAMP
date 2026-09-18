"""Strict compact record schema, physical validation and tensor fingerprints."""

import copy
import hashlib
import re
from dataclasses import fields
from typing import Any

import torch

from ..channel.parameters import PathParameters
from ..channel.validity import validate_cp_support
from ..config import Config
from ..modulation.allocation import build_allocation
from ..modulation.qpsk import bits_to_symbols, classes_to_symbols, hard_decision
from ..utils.device import Runtime
from ..utils.random import stable_hash
from ..waveform.frame import FrameWaveforms, build_frame

SCHEMA_VERSION = 1
GENERATOR_VERSION = "compact-physical-v1"
MODEL_VERSION = "affine-grid-8192-v1"
SPLITS = ("train", "val", "test")


def waveform_hash(config: Config) -> str:
    return stable_hash({key: config.values[key] for key in ("waveform", "frame", "receiver")})


def allocation_metadata(config: Config) -> dict[str, Any]:
    allocation = build_allocation(config)
    return {
        field.name: value.tolist() if isinstance(value, torch.Tensor) else value
        for field in fields(allocation)
        for value in [getattr(allocation, field.name)]
    }


def tensor_hash(values: dict[str, Any]) -> str:
    """Hash actual tensor payloads, shape, dtype and basic metadata, independent of .pt ZIP."""
    digest = hashlib.sha256()
    for key in sorted(values):
        value = values[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(stable_hash(value).encode())
    return digest.hexdigest()


def record_paths(record: dict[str, Any], device: torch.device | str = "cpu") -> PathParameters:
    return PathParameters(
        record["path_gain_complex"].to(device),
        record["path_delay_s"].to(device),
        record["path_epsilon"].to(device),
        record["scenario"],
    )


def record_frame(record: dict[str, Any], config: Config, runtime: Runtime) -> FrameWaveforms:
    # Physical generator precision and replay/audit device are explicit runtime choices.
    settings = copy.deepcopy(config.values)
    settings["runtime"]["device"] = str(runtime.device)
    settings["runtime"]["dtype"] = str(runtime.complex_dtype).removeprefix("torch.")
    return build_frame(
        record["data_bits"].to(runtime.device).unsqueeze(0),
        record["pilot_symbols"].to(runtime.device, runtime.complex_dtype).unsqueeze(0),
        Config(settings),
        runtime,
    )


def validate_record(record: Any, config: Config) -> None:
    """Reject malformed tensors/metadata before physical reconstruction; CPU loading only."""
    required = {
        "schema_version",
        "frame_id",
        "channel_id",
        "split",
        "scenario",
        "path_count",
        "path_gain_complex",
        "path_delay_s",
        "path_epsilon",
        "data_bits",
        "pilot_symbols",
        "esn0_db",
        "noise_seed_real",
        "noise_seed_imag",
        "arrival_offset_samples",
        "waveform_config_hash",
        "generator_version",
        "cp_valid",
        "generation_rejections",
        "snr_copy",
    }
    if not isinstance(record, dict) or record.keys() != required:
        raise ValueError("compact record has missing or unknown fields")
    if type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported record schema_version")
    if record["generator_version"] != GENERATOR_VERSION:
        raise ValueError("unsupported generator_version")
    for key in ("frame_id", "channel_id", "snr_copy"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"record {key} must be a nonempty string")
    for key in ("frame_id", "channel_id"):
        if re.fullmatch(r"[0-9a-f]{64}", record[key]) is None:
            raise ValueError(f"record {key} must be a canonical SHA-256 identity")
    if record["split"] not in SPLITS or record["cp_valid"] is not True:
        raise ValueError("invalid split or cp_valid")
    if record["waveform_config_hash"] != waveform_hash(config):
        raise ValueError("record waveform_config_hash mismatch")
    for key in ("path_count", "arrival_offset_samples", "generation_rejections"):
        if type(record[key]) is not int or record[key] < (1 if key == "path_count" else 0):
            raise ValueError(f"invalid record {key}")
    if record["arrival_offset_samples"] > config.values["frame"]["arrival_offset_max_samples"]:
        raise ValueError("arrival offset exceeds configured range")
    if record["generation_rejections"] >= config.values["data"]["max_channel_attempts"]:
        raise ValueError("generation rejection count exceeds retry budget")
    m = config.values["frame"]["n_ofdm_symbols"]
    specs = {
        "path_gain_complex": ((record["path_count"],), torch.complex128),
        "path_delay_s": ((record["path_count"],), torch.float64),
        "path_epsilon": ((record["path_count"],), torch.float64),
        "data_bits": ((m, 400, 2), torch.uint8),
        "pilot_symbols": ((m, 64), torch.complex128),
        "esn0_db": ((m,), torch.float64),
        "noise_seed_real": ((m,), torch.int64),
        "noise_seed_imag": ((m,), torch.int64),
    }
    for key, (shape, dtype) in specs.items():
        value = record[key]
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"record {key} requires {dtype} {shape}")
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"record {key} must be finite on CPU")
    bits_to_symbols(record["data_bits"])
    pilots = record["pilot_symbols"]
    if not torch.allclose(pilots, classes_to_symbols(hard_decision(pilots)), atol=1e-14, rtol=0):
        raise ValueError("pilot symbols must be unit QPSK")
    for key in ("noise_seed_real", "noise_seed_imag"):
        if bool((record[key] < 0).any()):
            raise ValueError("noise seeds must be nonnegative")
    seeds = torch.cat((record["noise_seed_real"], record["noise_seed_imag"]))
    if seeds.unique().numel() != 2 * m:
        raise ValueError("noise seeds must be independent per block and quadrature")
    if record["split"] == "test" and not bool((record["esn0_db"] == record["esn0_db"][0]).all()):
        raise ValueError("test frame must have one SNR for all blocks")
    expected_copy = float(record["esn0_db"][0]).hex() if record["split"] == "test" else "continuous"
    if record["snr_copy"] != expected_copy:
        raise ValueError("record SNR copy identity disagrees with split/SNR")
    from ..channel.noise import sigma2_from_esn0

    for snr in record["esn0_db"]:
        sigma2_from_esn0(float(snr))
    runtime = Runtime(torch.device("cpu"), torch.complex128, torch.float64, torch.get_num_threads())
    frame = record_frame(record, config, runtime)
    validate_cp_support(record_paths(record), frame.layout, config)
