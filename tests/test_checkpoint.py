"""Safe checkpoint corruption and incompatible state rejection."""

import copy

import pytest
import torch
from test_training import run

from pgvamp_ofdm.training.checkpoint import load_checkpoint, validate_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    run(tmp_path / "run", 2)
    return load_checkpoint(tmp_path / "run" / "last.pt")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.pop("rng_state"),
        lambda c: c.update(schema_version=2),
        lambda c: c.update(qpsk_mapping="wrong"),
        lambda c: c.update(mask_mode="hard"),
        lambda c: c.update(dtype="complex64"),
        lambda c: c.update(training_manifest_hash="bad"),
        lambda c: c["model_state_dict"].update(raw_mu=torch.ones(3)),
        lambda c: c["model_state_dict"]["raw_mu"].fill_(float("nan")),
        lambda c: c["sampler"]["permutation"].fill_(0),
        lambda c: c.update(sampler_position=999),
        lambda c: c["optimizer_state_dict"]["state"].clear(),
        lambda c: c["optimizer_state_dict"]["param_groups"][0].update(maximize=True),
        lambda c: c["optimizer_state_dict"]["state"][0]["exp_avg_sq"].fill_(-1),
        lambda c: c["rng_state"].update(torch=torch.zeros(3)),
    ],
)
def test_malformed_checkpoint(checkpoint, mutation):
    broken = copy.deepcopy(checkpoint)
    mutation(broken)
    with pytest.raises(ValueError):
        validate_checkpoint(broken)


def test_restricted_load_corruption(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"this is not a checkpoint")
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_resume_manifest_and_depth(tmp_path, checkpoint):
    from test_training import fixture_config

    from pgvamp_ofdm.smoke import math_samples
    from pgvamp_ofdm.training.trainer import train_samples

    path = tmp_path / "copy.pt"
    torch.save(checkpoint, path)
    data = math_samples(27)
    with pytest.raises(ValueError, match="manifest"):
        train_samples(fixture_config(), data, data, tmp_path / "bad", "a" * 64, resume=path)
    config = fixture_config()
    config.values["pg_vamp"]["depth"] = 3
    with pytest.raises(ValueError, match="contract"):
        train_samples(config, data, data, tmp_path / "bad", "a" * 64, resume=path)


def test_prediction_roundtrip_from_live_model(tmp_path, checkpoint):
    from pgvamp_ofdm.config import config_from_values
    from pgvamp_ofdm.smoke import math_samples
    from pgvamp_ofdm.training.checkpoint import model_for, save_checkpoint
    from pgvamp_ofdm.utils.device import resolve_runtime

    runtime = resolve_runtime()
    config = config_from_values(checkpoint["resolved_config"])
    original = model_for(config, runtime)
    original.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        original.raw_mu.add_(0.1)
    checkpoint["model_state_dict"] = original.state_dict()
    sample = math_samples(13)[0]
    inputs = {k: sample[k][None] for k in ("H", "y", "sigma2")}
    expected = original(**inputs).x_soft.detach()
    save_checkpoint(tmp_path / "roundtrip.pt", checkpoint)
    restored = model_for(config, runtime)
    restored.load_state_dict(load_checkpoint(tmp_path / "roundtrip.pt")["model_state_dict"])
    torch.testing.assert_close(expected, restored(**inputs).x_soft, atol=0, rtol=0)
