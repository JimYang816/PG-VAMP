"""Strict WP0 configuration and static summaries (no physical simulation)."""

import copy
import math
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A configuration violates the explicit source contract."""


class _Loader(yaml.SafeLoader):
    pass


# YAML 1.2-style exponent numbers also accept the source's `1.0e8` spelling.
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def _mapping(loader: _Loader, node: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ConfigError(f"configuration keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_Loader.add_constructor("tag:yaml.org,2002:map", _mapping)


def _parse(text: str) -> dict[str, Any]:
    try:
        value = yaml.load(text, Loader=_Loader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration must be a mapping")
    return value


_DEFAULT = _parse(files("pgvamp_ofdm").joinpath("base.yaml").read_text(encoding="utf-8"))
_SCENARIOS = (
    "identity_awgn",
    "static_multipath",
    "affine_doppler_mild",
    "affine_doppler_moderate",
    "affine_doppler_strong",
)
# Strings not listed here have a single supported contract value in base.yaml.
_ENUMS: dict[str, tuple[str, ...]] = {
    "runtime.dtype": ("complex128", "complex64"),
    "receiver.sync_mode": ("oracle_timing", "lfm_detect"),
    "data.backend": ("effective_fast", "waveform_reference"),
    "data.storage": ("compact_frame_records", "materialized"),
    "evaluation.label": ("development_only", "main_simulation", "smoke_system"),
    "evaluation.timing_mode": ("per_observation_cold_H", "same_H_amortized"),
}
_ZERO_OK = {
    "seed",
    "runtime.num_workers",
    "waveform.n_guard_left",
    "waveform.n_guard_right",
    "frame.leading_silence_samples",
    "frame.sync_guard_samples",
    "frame.trailing_silence_samples",
    "frame.arrival_offset_max_samples",
    "data.audit_waveform_frames",
    "data.matrix_cache_entries",
    "pg_vamp.jitter",
    "training.weight_decay",
    "evaluation.timing_warmup",
    "channel.first_delay_s",
    "frame.lfm_tukey_alpha",
}
_SIGNED = {"pg_vamp.rho_hi_db", "pg_vamp.rho_lo_db"}
_FIXED = {
    "schema_version": 1,
    "runtime.amp": False,
    "waveform.dc_null": True,
    "waveform.symbol_energy": 1.0,
    "waveform.pilot_energy": 1.0,
    "channel.forbid_interblock_interference": True,
    "channel.validate_cp_support": True,
    "noise.use_measured_per_frame_signal_power": False,
    "pg_vamp.precision_min": 1e-10,
    "pg_vamp.precision_max": 1e8,
    "pg_vamp.alpha_margin": 1e-6,
    "vamp.precision_min": 1e-10,
    "vamp.precision_max": 1e8,
    "vamp.alpha_margin": 1e-6,
    "vamp.learned_parameters": False,
    "vamp.adaptive_stopping": False,
    "mmse.qpsk_post_denoiser": False,
    "evaluation.shared_samples": True,
    "evaluation.save_per_frame_counts": True,
}


def _check(value: Any, default: Any, path: str) -> None:
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path or 'root'} must be a mapping")
        for key, item in value.items():
            if key not in default:
                raise ConfigError(f"unknown configuration key: {path + '.' if path else ''}{key}")
            _check(item, default[key], f"{path}.{key}" if path else key)
        return
    if isinstance(default, list):
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{path} must be a nonempty list")
        if path == "evaluation.algorithms":
            valid = all(v in ("mmse", "vamp", "pg_vamp") for v in value)
        elif path == "evaluation.scenarios":
            valid = all(v in _SCENARIOS for v in value)
        else:
            valid = all(type(v) in (int, float) and math.isfinite(v) for v in value)
        if not valid or any(value[i] == value[j] for i in range(len(value)) for j in range(i)):
            raise ConfigError(f"{path} contains invalid or duplicate entries")
        return
    if type(default) is bool:
        valid = type(value) is bool
    elif type(default) is int:
        valid = type(value) is int
    elif type(default) is float:
        valid = type(value) in (int, float)
    else:
        valid = isinstance(value, str)
    if not valid:
        raise ConfigError(f"{path} requires {type(default).__name__}")
    if path in _FIXED and value != _FIXED[path]:
        raise ConfigError(f"{path} must be {_FIXED[path]!r}")
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ConfigError(f"{path} must be finite")
        if path in _SIGNED:
            return
        if path.startswith(("channel.epsilon_max.", "data.train_scenario_weights.")):
            if not 0 <= value <= 1:
                raise ConfigError(f"{path} must be in [0,1]")
        elif value < 0 or (value == 0 and path not in _ZERO_OK):
            raise ConfigError(f"{path} must be {'nonnegative' if path in _ZERO_OK else 'positive'}")
    elif isinstance(value, str):
        if path == "runtime.device":
            if not re.fullmatch(r"cpu|cuda(?::\d+)?", value):
                raise ConfigError("runtime.device must be cpu or cuda[:index]")
        elif value not in _ENUMS.get(path, (default,)):
            raise ConfigError(f"unsupported {path}: {value!r}")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) else value
    return result


@dataclass(frozen=True)
class Config:
    """Validated serialized settings; mode is explicit, never inferred from dimensions."""

    values: dict[str, Any]

    @property
    def mode(self) -> str:
        """Return physical or algebra_fixture."""
        return "algebra_fixture" if self.values.get("profile") == "smoke_math" else "physical"

    def save(self, path: str | Path) -> None:
        """Save a complete resolved config as safe YAML, without generating data."""
        Path(path).write_text(yaml.safe_dump(self.values, sort_keys=False), encoding="utf-8")


def load_config(
    path: str | Path | None = None,
    *,
    device: str | None = None,
    dtype: str | None = None,
) -> Config:
    """Merge a strict profile over bundled §20 defaults; reject unknown nested keys."""
    given = _parse(Path(path).read_text(encoding="utf-8")) if path is not None else {}
    profile = given.pop("profile", "physical")
    if profile not in ("physical", "smoke_math"):
        raise ConfigError("profile must be physical or smoke_math")
    if profile == "smoke_math":
        defaults = {
            "schema_version": 1,
            "seed": _DEFAULT["seed"],
            "runtime": _DEFAULT["runtime"],
            "algebra": {"dimension": 32, "depth": 2, "updates": 2},
        }
        _check(given, defaults, "")
        resolved = _merge(defaults, given)
        if resolved["algebra"] != defaults["algebra"]:
            raise ConfigError("smoke_math requires algebra N=32, T=2, updates=2")
    else:
        defaults = _DEFAULT
        _check(given, defaults, "")
        resolved = _merge(defaults, given)
    # Validate user input before CLI overrides so an invalid value is never hidden.
    if device is not None:
        _check(device, "cpu", "runtime.device")
        resolved["runtime"]["device"] = device
    if dtype is not None:
        _check(dtype, "complex128", "runtime.dtype")
        resolved["runtime"]["dtype"] = dtype
    if profile == "physical":
        _physical(resolved)
    resolved["profile"] = profile
    return Config(resolved)


def _physical(c: dict[str, Any]) -> None:
    w, f, ch, pg = c["waveform"], c["frame"], c["channel"], c["pg_vamp"]
    band = w["band_hz"]
    if len(band) != 2 or not 0 < band[0] < band[1] < w["sample_rate_hz"] / 2:
        raise ConfigError("waveform.band_hz requires two increasing positive sub-Nyquist edges")
    spacing = (band[1] - band[0]) / w["n_grid"]
    if not math.isclose(w["n_fft_wave"] * spacing, w["sample_rate_hz"], rel_tol=1e-12):
        raise ConfigError("n_fft_wave * subcarrier_spacing_hz must equal sample_rate_hz")
    if not math.isclose(
        w["carrier_hz"] / spacing, round(w["carrier_hz"] / spacing), abs_tol=1e-10, rel_tol=0
    ):
        raise ConfigError("carrier_hz must lie on the FFT grid")
    if w["carrier_hz"] != sum(band) / 2:
        raise ConfigError("carrier_hz must equal the design band center")
    allocation = (
        w["n_grid"],
        w["n_data"],
        w["n_pilots"],
        w["n_guard_left"],
        w["n_guard_right"],
        w["dc_null"],
    )
    if allocation != (512, 400, 64, 24, 23, True):
        raise ConfigError("default_400_64_47_1 requires 512 grid and 400/64/24/23/1 counts")
    if not 0 < w["cp_samples"] <= w["n_fft_wave"]:
        raise ConfigError("cp_samples must be in (0,n_fft_wave]")
    if not ch["first_delay_s"] <= ch["max_delay_s"] <= w["cp_samples"] / w["sample_rate_hz"]:
        raise ConfigError("initial channel delays must be ordered and within CP duration")
    if ch["min_paths"] > ch["max_paths"]:
        raise ConfigError("channel.min_paths must not exceed max_paths")
    if not 0 <= f["lfm_tukey_alpha"] <= 1 or not 0 < c["receiver"]["lfm_threshold"] <= 1:
        raise ConfigError("LFM window alpha and receiver threshold must be in [0,1] / (0,1]")
    if [f["lfm_start_hz"], f["lfm_stop_hz"]] != band:
        raise ConfigError("LFM sweep endpoints must match design band")
    if pg["rho_hi_db"] - pg["rho_lo_db"] - pg["depth"] * pg["min_gap_db"] <= 0:
        raise ConfigError("rho_hi - rho_lo - depth*min_gap must be positive")
    if not 0 < pg["init_mu"] < 1:
        raise ConfigError("pg_vamp.init_mu must be in (0,1)")
    snr = c["data"]["train_esn0_range_db"]
    if len(snr) != 2 or snr[0] >= snr[1]:
        raise ConfigError("train_esn0_range_db requires increasing endpoints")
    if not math.isclose(sum(c["data"]["train_scenario_weights"].values()), 1, abs_tol=1e-12):
        raise ConfigError("train_scenario_weights must sum to one")


def config_from_values(values: dict[str, Any]) -> Config:
    """Validate a complete persisted physical configuration without temporary files."""
    if not isinstance(values, dict) or values.get("profile") != "physical":
        raise ConfigError("persisted dataset requires a physical resolved config")
    given = copy.deepcopy(values)
    given.pop("profile")
    _check(given, _DEFAULT, "")

    def complete(value: dict[str, Any], default: dict[str, Any]) -> None:
        if value.keys() != default.keys():
            raise ConfigError("persisted resolved config has missing keys")
        for key, item in default.items():
            if isinstance(item, dict):
                complete(value[key], item)

    complete(given, _DEFAULT)
    _physical(given)
    given["profile"] = "physical"
    return Config(given)


def summarize(config: Config) -> dict[str, Any]:
    """Return static dimensions and byte estimates; no CP paths are generated or certified."""
    c = config.values
    summary: dict[str, Any] = {"mode": config.mode, **c["runtime"]}
    if config.mode == "algebra_fixture":
        summary.update(c["algebra"])
        summary["physical_parameters"] = "not applicable: algebra fixture only"
        return summary
    w, f, d = c["waveform"], c["frame"], c["data"]
    fs, nw, n, m = w["sample_rate_hz"], w["n_fft_wave"], w["n_data"], f["n_ofdm_symbols"]
    df = (w["band_hz"][1] - w["band_hz"][0]) / w["n_grid"]
    count = sum(
        f[k]
        for k in (
            "leading_silence_samples",
            "lfm_samples",
            "sync_guard_samples",
            "trailing_silence_samples",
        )
    )
    count += m * (nw + w["cp_samples"])
    cb = 16 if c["runtime"]["dtype"] == "complex128" else 8
    rb = cb // 2
    frames = d["train_frames"] + d["val_frames"]
    blocks = frames * m
    dense = blocks * (n * n * cb + 2 * n * cb + rb + 2 * n)
    # Compact tensor payload uses generator precision (complex128/float64),
    # uint8 bits, explicit pilot symbols, per-block EsN0 and two 64-bit noise seeds.
    compact = frames * (c["channel"]["max_paths"] * 32 + m * n * 2 + m * (w["n_pilots"] * 16 + 24))
    summary.update(
        {
            "sample_rate_hz": fs,
            "n_fft_wave": nw,
            "n_grid": w["n_grid"],
            "n_data": n,
            "n_pilots": w["n_pilots"],
            "n_guard": w["n_guard_left"] + w["n_guard_right"],
            "n_dc": 1,
            "design_band_hz": w["band_hz"],
            "carrier_hz": w["carrier_hz"],
            "subcarrier_spacing_hz": df,
            "grid_endpoints_hz": [w["carrier_hz"] - 256 * df, w["carrier_hz"] + 255 * df],
            "active_endpoints_hz": [w["carrier_hz"] - 232 * df, w["carrier_hz"] + 232 * df],
            "cp_samples": w["cp_samples"],
            "cp_duration_ms": 1000 * w["cp_samples"] / fs,
            "useful_duration_ms": 1000 * nw / fs,
            "frame_samples": count,
            "frame_duration_s": count / fs,
            "data_bits_per_frame": m * n * 2,
            "cp_path_support": "not evaluated: requires WP2 physical paths",
            "capacity_estimate": {
                "scope": "train + validation; excludes evaluation and audits",
                "frames": frames,
                "blocks": blocks,
                "single_H_bytes": n * n * cb,
                "dense_H_only_bytes": blocks * n * n * cb,
                "dense_labeled_payload_bytes": dense,
                "compact_numeric_payload_bytes": compact,
                "compact_assumptions": (
                    "max_paths, float64 paths, uint8 bits, complex128 pilots, per-block seeds"
                ),
                "exclusions": (
                    "variable metadata, IDs, hashes, serialization and filesystem overhead"
                ),
            },
            "scenario_degeneracies": {
                "identity_awgn": "identity H; no delayed/scaled paths",
                "static_multipath": "epsilon=0 for every path",
            },
        }
    )
    return summary
