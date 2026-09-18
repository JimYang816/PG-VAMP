"""Independent regression checks for persisted compatibility and failure boundaries."""

import copy
import json
import shutil

import pytest
import torch
from test_training import equal_tree, run

from pgvamp_ofdm.config import config_from_values, load_config
from pgvamp_ofdm.data.manifest import load_manifest, save_tensors
from pgvamp_ofdm.data.materialize import load_materialized, materialize
from pgvamp_ofdm.training.checkpoint import load_checkpoint, validate_checkpoint
from pgvamp_ofdm.utils.random import stable_hash


def legacy(values):
    result = copy.deepcopy(values)
    for key in ("early_stopping_patience", "early_stopping_min_delta"):
        result["training"].pop(key)
    return result


def test_legacy_config_is_exact_and_not_default_filled():
    values = legacy(load_config().values)
    result = config_from_values(values, allow_legacy_data=True)
    assert result.values == values
    with pytest.raises(ValueError, match="missing keys"):
        config_from_values(values)
    for key in ("batch_size", "early_stopping", "learning_rate"):
        bad = copy.deepcopy(values)
        bad["training"].pop(key)
        with pytest.raises(ValueError, match="missing keys"):
            config_from_values(bad, allow_legacy_data=True)
    values["training"]["early_stopping_patience"] = 10
    with pytest.raises(ValueError, match="missing keys"):
        config_from_values(values, allow_legacy_data=True)


def test_legacy_manifest_and_materialized_hashes(data_manifest, tmp_path):
    root = tmp_path / "legacy"
    shutil.copytree(data_manifest.parent, root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["resolved_config"] = legacy(manifest["resolved_config"])
    manifest["config_sha256"] = stable_hash(manifest["resolved_config"])
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    loaded, config, _ = load_manifest(path)
    assert config.values == manifest["resolved_config"]
    assert loaded["config_sha256"] == stable_hash(config.values)
    materialize(path, "val", tmp_path / "dense.pt", labeled=False)
    dense = load_materialized(tmp_path / "dense.pt")
    assert dense["metadata"]["config_sha256"] == manifest["config_sha256"]
    assert path.read_bytes() == before
    dense["metadata"]["resolved_config"]["seed"] += 1
    save_tensors(tmp_path / "tampered.pt", dense)
    with pytest.raises(ValueError, match="configuration"):
        load_materialized(tmp_path / "tampered.pt")


@pytest.mark.parametrize(
    "flag",
    [
        "amsgrad",
        "maximize",
        "capturable",
        "differentiable",
        "foreach",
        "fused",
        "decoupled_weight_decay",
    ],
)
def test_adam_behavior_flags_rejected(tmp_path, flag):
    run(tmp_path / "run", 2)
    state = load_checkpoint(tmp_path / "run" / "last.pt")
    state["optimizer_state_dict"]["param_groups"][0][flag] = True
    with pytest.raises(ValueError, match=flag):
        validate_checkpoint(state)


@pytest.mark.parametrize("failed_name", ["last.pt", "scheduled-best.pt"])
def test_interrupted_checkpoint_write_replays_committed_boundary(
    tmp_path, monkeypatch, failed_name
):
    import pgvamp_ofdm.training.trainer as trainer

    run(tmp_path / "whole", 4)
    original = trainer.save_checkpoint

    def fail(path, state):
        if path.name == failed_name and state["step"] == 2:
            raise OSError("injected checkpoint write failure")
        return original(path, state)

    monkeypatch.setattr(trainer, "save_checkpoint", fail)
    with pytest.raises(OSError):
        run(tmp_path / "split", 4)
    committed = load_checkpoint(tmp_path / "split" / "last.pt")
    assert committed["step"] == (1 if failed_name == "last.pt" else 2)
    monkeypatch.setattr(trainer, "save_checkpoint", original)
    run(tmp_path / "split", 4, tmp_path / "split" / "last.pt")
    a, b = (load_checkpoint(tmp_path / name / "last.pt") for name in ("whole", "split"))
    for key in ("model_state_dict", "optimizer_state_dict", "sampler", "rng_state"):
        equal_tree(a[key], b[key])
    events = [
        json.loads(s) for s in (tmp_path / "split" / "training.jsonl").read_text().splitlines()
    ]
    resume = next(e for e in events if e["event"] == "resume")
    assert resume["invalidates_prior_events_after_step"] == committed["step"]


def test_inference_detector_failure_has_ids_dtype_norm(tmp_path, monkeypatch):
    from pgvamp_ofdm.algorithms import PGVAMPDetector
    from pgvamp_ofdm.inference import infer

    # Independent acceptance artifacts are not test fixtures: use the normal
    # tiny trainer, and patch validated input loading only for this boundary test.
    run(tmp_path / "training", 2)
    h = torch.eye(400, dtype=torch.complex128)[None]
    value = {
        "H": h,
        "y": h[:, 0],
        "sigma2": torch.ones(1, dtype=torch.float64),
        "metadata": {"resolved_config": load_config().values, "sample_ids": ["failure-id"]},
    }
    monkeypatch.setattr("pgvamp_ofdm.inference.load_materialized", lambda *a, **k: value)

    def fail(*args, **kwargs):
        raise FloatingPointError("PG-VAMP Cholesky; layer=1")

    monkeypatch.setattr(PGVAMPDetector, "forward", fail)
    with pytest.raises(FloatingPointError, match="failure-id.*dtype=.*H_norm="):
        infer(tmp_path / "training" / "last.pt", tmp_path / "input.pt", tmp_path / "out.pt")
    assert not (tmp_path / "out.pt").exists()


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("waveform", "n_fft_wave", 4096),
        ("waveform", "cp_samples", 1024),
        ("frame", "n_ofdm_symbols", 1),
    ],
)
def test_system_smoke_rejects_reduced_physical_dimensions(tmp_path, section, key, value):
    from pgvamp_ofdm.smoke import smoke

    config = load_config("configs/smoke_system.yaml")
    config.values[section][key] = value
    with pytest.raises(ValueError, match="8192/2048/8"):
        smoke(config, tmp_path / "smoke")
