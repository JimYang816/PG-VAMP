"""Inference label isolation and explicit dtype conversion on physical WP3 inputs."""

import pytest
import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.data.manifest import load_tensors, save_tensors
from pgvamp_ofdm.data.materialize import materialize
from pgvamp_ofdm.inference import infer
from pgvamp_ofdm.modulation.qpsk import bits_to_symbols
from pgvamp_ofdm.training.trainer import train


@pytest.fixture(scope="module")
def artifacts(data_manifest, tmp_path_factory):
    path = tmp_path_factory.mktemp("inference")
    config = load_config("configs/wp3_smoke.yaml")
    config.values["pg_vamp"]["depth"] = 2
    config.values["training"].update(max_steps=2, validation_max_blocks=1, validation_every_steps=1)
    train(config, data_manifest, path / "training")
    materialize(data_manifest, "val", path / "labeled.pt")
    value = load_tensors(path / "labeled.pt")
    value.pop("x")
    value.pop("bits")
    save_tensors(path / "unlabeled.pt", value)
    return path


def test_label_free_changed_labels_dtype_cpu(artifacts, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU called CUDA")

    for key in (
        "is_available",
        "get_rng_state_all",
        "set_rng_state_all",
        "manual_seed_all",
        "device_count",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, key, forbidden)
    ck = artifacts / "training" / "best.pt"
    for name in ("labeled", "unlabeled"):
        infer(ck, artifacts / f"{name}.pt", tmp_path / f"{name}.pt")
    a, b = (load_tensors(tmp_path / f"{name}.pt") for name in ("labeled", "unlabeled"))
    assert torch.equal(a["x_soft"], b["x_soft"])
    changed = load_tensors(artifacts / "labeled.pt")
    changed["bits"] = 1 - changed["bits"]
    changed["x"] = bits_to_symbols(changed["bits"])
    save_tensors(tmp_path / "changed.pt", changed)
    infer(ck, tmp_path / "changed.pt", tmp_path / "changed-output.pt")
    assert torch.equal(a["x_soft"], load_tensors(tmp_path / "changed-output.pt")["x_soft"])
    infer(ck, artifacts / "unlabeled.pt", tmp_path / "single.pt", dtype="complex64")
    single = load_tensors(tmp_path / "single.pt")
    assert single["explicit_dtype_conversion"] and single["x_soft"].dtype == torch.complex64
    # Same well-conditioned fixture, allowance for float32 factorization/accumulation.
    torch.testing.assert_close(
        single["x_soft"].to(torch.complex128), a["x_soft"], atol=2e-5, rtol=2e-4
    )
    assert single["metadata"]["sample_ids"] == a["metadata"]["sample_ids"]


def test_inference_physics_mismatch(artifacts, tmp_path):
    from pgvamp_ofdm.config import config_from_values
    from pgvamp_ofdm.utils.random import stable_hash

    data = load_tensors(artifacts / "unlabeled.pt")
    data["metadata"]["resolved_config"]["receiver"]["sync_mode"] = "lfm_detect"
    config = config_from_values(data["metadata"]["resolved_config"])
    data["metadata"]["config_sha256"] = stable_hash(config.values)
    save_tensors(tmp_path / "bad.pt", data)
    with pytest.raises(ValueError, match="physics"):
        infer(artifacts / "training" / "best.pt", tmp_path / "bad.pt", tmp_path / "output.pt")


def test_inference_legacy_data_config(artifacts, tmp_path):
    from pgvamp_ofdm.utils.random import stable_hash

    data = load_tensors(artifacts / "unlabeled.pt")
    config = data["metadata"]["resolved_config"]
    for key in ("early_stopping_patience", "early_stopping_min_delta"):
        config["training"].pop(key)
    data["metadata"]["config_sha256"] = stable_hash(config)
    save_tensors(tmp_path / "legacy.pt", data)
    ck = artifacts / "training" / "best.pt"
    infer(ck, tmp_path / "legacy.pt", tmp_path / "legacy-output.pt")
    infer(ck, artifacts / "unlabeled.pt", tmp_path / "current-output.pt")
    assert torch.equal(
        load_tensors(tmp_path / "legacy-output.pt")["x_soft"],
        load_tensors(tmp_path / "current-output.pt")["x_soft"],
    )


def test_explicit_cuda_unavailable(artifacts, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        infer(
            artifacts / "training" / "best.pt",
            artifacts / "unlabeled.pt",
            tmp_path / "output.pt",
            device="cuda:0",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_inference_agreement(artifacts, tmp_path):
    ck = artifacts / "training" / "best.pt"
    for device, name in (("cpu", "cpu"), ("cuda:0", "cuda")):
        infer(ck, artifacts / "unlabeled.pt", tmp_path / f"{name}.pt", device=device)
    torch.testing.assert_close(
        load_tensors(tmp_path / "cpu.pt")["x_soft"],
        load_tensors(tmp_path / "cuda.pt")["x_soft"],
        atol=1e-9,
        rtol=1e-8,
    )
