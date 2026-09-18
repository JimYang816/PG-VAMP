import copy

import pytest
import torch

from pgvamp_ofdm.channel.parameters import PathParameters
from pgvamp_ofdm.data.generate import generate_records
from pgvamp_ofdm.utils.random import derive_seed, generator


def test_stream_separation():
    streams = ["split", "channel", "bits", "pilots", "noise", "arrival_offset"]
    seeds = [derive_seed(12, stream, "frame", 0) for stream in streams]
    assert len(set(seeds)) == len(streams)
    expected = torch.rand(10, generator=generator(seeds[2]))
    torch.rand(999, generator=generator(seeds[1]))
    assert torch.equal(expected, torch.rand(10, generator=generator(seeds[2])))
    assert derive_seed(12, "noise", "frame", 0, "real") != derive_seed(
        12, "noise", "frame", 0, "imag"
    )


def test_retry_isolation_and_exhaustion(data_config, monkeypatch):
    import pgvamp_ofdm.data.generate as module

    original = module.sample_paths
    baseline = generate_records(data_config)
    calls = 0

    def first_invalid(config, scenario, *, generator):
        nonlocal calls
        calls += 1
        paths = original(config, scenario, generator=generator)
        if calls % 2:
            return PathParameters(
                paths.gain, torch.ones_like(paths.delay_s), paths.epsilon, scenario
            )
        return paths

    monkeypatch.setattr(module, "sample_paths", first_invalid)
    changed = generate_records(data_config)
    for before, after in zip(baseline, changed):
        assert after["generation_rejections"] == 1
        for key in ("data_bits", "pilot_symbols", "noise_seed_real", "noise_seed_imag", "esn0_db"):
            assert torch.equal(before[key], after[key])
        assert before["arrival_offset_samples"] == after["arrival_offset_samples"]
    data_config.values["data"]["max_channel_attempts"] = 1
    calls = 0
    with pytest.raises(ValueError, match="retries exhausted"):
        generate_records(data_config)


def test_backend_dtype_do_not_change_physical_identity(data_config):
    base = generate_records(data_config)
    config = copy.deepcopy(data_config)
    config.values["runtime"]["dtype"] = "complex64"
    config.values["data"]["backend"] = "waveform_reference"
    other = generate_records(config)
    for left, right in zip(base, other):
        assert left["frame_id"] == right["frame_id"]
        for key in ("data_bits", "path_gain_complex", "noise_seed_real"):
            assert torch.equal(left[key], right[key])


def test_training_mixture_and_continuous_snr(data_config):
    data_config.values["data"]["train_frames"] = 400
    records = [r for r in generate_records(data_config) if r["split"] == "train"]
    for scenario, weight in data_config.values["data"]["train_scenario_weights"].items():
        actual = sum(r["scenario"] == scenario for r in records)
        assert abs(actual - 400 * weight) < 6 * (400 * weight * (1 - weight)) ** 0.5
    snrs = torch.cat([r["esn0_db"] for r in records])
    assert abs(float(snrs.mean()) - 10) < 6 * 30 / (12 * snrs.numel()) ** 0.5
