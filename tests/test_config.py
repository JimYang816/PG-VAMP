from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from pgvamp_ofdm.config import ConfigError, load_config, summarize


def test_source_defaults_and_packaged_copy():
    assert (
        Path("configs/base.yaml").read_bytes()
        == files("pgvamp_ofdm").joinpath("base.yaml").read_bytes()
    )
    default = load_config("configs/base.yaml").values
    assert default["pg_vamp"]["precision_max"] == 1e8
    assert default["vamp"]["precision_min"] == 1e-10
    assert default == load_config("configs/cpu_dev.yaml").values


def test_cpu_summary():
    s = summarize(load_config("configs/cpu_dev.yaml"))
    assert (s["device"], s["dtype"], s["sample_rate_hz"], s["n_fft_wave"]) == (
        "cpu",
        "complex128",
        96000,
        8192,
    )
    assert [s[k] for k in ("n_grid", "n_data", "n_pilots", "n_guard", "n_dc")] == [
        512,
        400,
        64,
        47,
        1,
    ]
    assert s["subcarrier_spacing_hz"] == 11.71875
    assert s["grid_endpoints_hz"] == [21000, 26988.28125]
    assert s["active_endpoints_hz"] == [21281.25, 26718.75]
    assert s["cp_samples"] == 2048
    assert s["cp_duration_ms"] == pytest.approx(21.3333333333333)
    assert (s["frame_samples"], s["frame_duration_s"], s["data_bits_per_frame"]) == (
        91776,
        0.956,
        6400,
    )
    estimate = s["capacity_estimate"]
    assert estimate["single_H_bytes"] == 400 * 400 * 16
    assert estimate["dense_H_only_bytes"] == 160 * 8 * 400 * 400 * 16
    assert estimate["compact_numeric_payload_bytes"] < estimate["dense_H_only_bytes"]


@pytest.mark.parametrize(
    "name", ["base", "cpu_dev", "main", "smoke_system", "smoke_math", "cuda_example"]
)
def test_profiles_roundtrip(name, tmp_path):
    original = load_config(f"configs/{name}.yaml")
    dest = tmp_path / "resolved.yaml"
    original.save(dest)
    assert load_config(dest).values == original.values


def test_profile_scales():
    main = load_config("configs/main.yaml").values
    assert main["runtime"]["device"] == "cpu"
    assert main["data"]["train_frames"] == 1024
    assert main["data"]["val_frames"] == 128
    assert main["training"]["max_steps"] == 5000
    assert main["training"]["validation_max_blocks"] == 1024
    assert len(main["evaluation"]["scenarios"]) == 5
    assert main["evaluation"]["esn0_db"] == [-5, 0, 5, 10, 15, 20, 25]
    assert main["evaluation"]["frames_per_cell"] == 256
    math = load_config("configs/smoke_math.yaml")
    assert math.mode == "algebra_fixture"
    assert "waveform" not in math.values
    assert math.values["algebra"] == {"dimension": 32, "depth": 2, "updates": 2}
    system = load_config("configs/smoke_system.yaml").values
    assert system["waveform"]["n_data"] == 400
    assert system["pg_vamp"]["depth"] == 8
    assert system["training"]["max_steps"] == 2


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"typo": 1}, "unknown"),
        ({"runtime": {"typo": 1}}, "runtime.typo"),
        ({"channel": {"epsilon_max": {"typo": 1}}}, "epsilon_max.typo"),
        ({"runtime": {"dtype": "float16"}}, "dtype"),
        ({"runtime": {"device": "mps"}}, "device"),
        ({"runtime": {"cpu_threads": True}}, "cpu_threads"),
        ({"runtime": {"amp": True}}, "amp"),
        ({"waveform": {"coding": "ldpc"}}, "coding"),
        ({"receiver": {"csi_mode": "estimated"}}, "csi_mode"),
        ({"waveform": {"n_fft_wave": 512}}, "subcarrier_spacing"),
        ({"waveform": {"n_data": 32}}, "400/64"),
        ({"waveform": {"carrier_hz": 24001}}, "FFT grid"),
        ({"waveform": {"cp_samples": 10}}, "CP duration"),
        ({"pg_vamp": {"depth": 120}}, "min_gap"),
        ({"pg_vamp": {"precision_min": 1e-12}}, "precision_min"),
        ({"pg_vamp": {"jitter": -1}}, "jitter"),
        ({"training": {"learning_rate": float("nan")}}, "finite"),
        ({"evaluation": {"algorithms": [{}]}}, "invalid"),
        ({"evaluation": {"esn0_db": [[]]}}, "invalid"),
        ({"data": {"train_scenario_weights": {"static_multipath": 0.8}}}, "sum to one"),
        ({"profile": "smoke_math", "waveform": {"n_data": 32}}, "unknown"),
        ({"profile": "smoke_math", "algebra": {"dimension": 8}}, "N=32"),
    ],
)
def test_invalid_configs(bad, match, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_duplicate_and_invalid_override(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unique"):
        load_config(path)
    path.write_text("runtime: {dtype: float16}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="dtype"):
        load_config(path, dtype="complex128")


def test_tukey_zero_and_cli_override(tmp_path):
    path = tmp_path / "alpha.yaml"
    path.write_text("frame: {lfm_tukey_alpha: 0}\n", encoding="utf-8")
    c = load_config(path, dtype="complex64", device="cuda:2")
    assert c.values["frame"]["lfm_tukey_alpha"] == 0
    assert c.values["runtime"]["dtype"] == "complex64"
