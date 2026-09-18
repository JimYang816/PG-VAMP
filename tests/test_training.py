"""WP6 independent loss, deterministic resume, logging and failure acceptance."""

import copy
import json
import random

import numpy as np
import pytest
import torch

from pgvamp_ofdm.algorithms import PGVAMPDetector
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.smoke import math_samples
from pgvamp_ofdm.training.checkpoint import load_checkpoint, restore_rng, rng_state
from pgvamp_ofdm.training.losses import layer_loss
from pgvamp_ofdm.training.trainer import train_samples
from pgvamp_ofdm.utils.random import stable_hash


def fixture_config(steps=4, dtype="complex128"):
    config = load_config(dtype=dtype)
    config.values["pg_vamp"]["depth"] = 2
    config.values["training"].update(max_steps=steps, batch_size=2, validation_every_steps=1)
    return config


def run(path, steps=4, resume=None, dtype="complex128"):
    config = fixture_config(steps, dtype)
    data = math_samples(27, getattr(torch, dtype))
    return train_samples(config, data, data, path, stable_hash("fixture"), resume=resume)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_loss_independent_and_layer_connectivity():
    target = torch.tensor([[1 + 2j, -2 + 1j]], dtype=torch.complex128)
    a = torch.tensor([[2 + 4j, 1 + 1j]], dtype=torch.complex128, requires_grad=True)
    b = torch.tensor([[1 + 1j, -2 + 3j]], dtype=torch.complex128, requires_grad=True)
    loss = layer_loss([a, b], target)
    assert loss.item() == pytest.approx((5 + 9) / 2 / 3 + (1 + 4) / 2 * 2 / 3)
    loss.backward()
    assert a.grad is not None and b.grad is not None
    sample = math_samples(13)[0]
    model = PGVAMPDetector(2)
    inputs = {k: sample[k].unsqueeze(0) for k in ("H", "y", "sigma2")}
    small = model(**inputs, return_layer_outputs=True)
    detailed = model(**inputs, return_diagnostics=True)
    assert "layers" not in small.diagnostics
    assert sum(p.numel() for p in model.parameters()) == 4
    for x, layer in zip(small.diagnostics["layer_outputs"], detailed.diagnostics["layers"]):
        torch.testing.assert_close(x, layer["xhat1"], atol=0, rtol=0)
    gs = torch.autograd.grad(
        layer_loss(small.diagnostics["layer_outputs"], sample["x"][None]), tuple(model.parameters())
    )
    gd = torch.autograd.grad(
        layer_loss([s["xhat1"] for s in detailed.diagnostics["layers"]], sample["x"][None]),
        tuple(model.parameters()),
    )
    for a, b in zip(gs, gd):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert bool(torch.isfinite(a).all()) and bool((a != 0).all())


def test_strict_resume_epoch_tail_optimizer_rng(tmp_path):
    run(tmp_path / "continuous")
    run(tmp_path / "resumed", 2)
    run(tmp_path / "resumed", 4, tmp_path / "resumed" / "last.pt")
    a, b = (load_checkpoint(tmp_path / name / "last.pt") for name in ("continuous", "resumed"))
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "sampler",
        "epoch",
        "sampler_position",
        "best_step",
        "bad_validation_events",
    ):
        equal_tree(a[key], b[key])
    assert a["epoch"] == 1 and a["sampler_position"] == 3
    sequences = []
    for name in ("continuous", "resumed"):
        events = [
            json.loads(line)
            for line in (tmp_path / name / "training.jsonl").read_text().splitlines()
        ]
        sequences.append([e["sample_ids"] for e in events if e["event"] == "update"])
    assert sequences[0] == sequences[1]

    def draw():
        return random.random(), np.random.rand(), torch.rand(1)

    restore_rng(a["rng_state"], torch.device("cpu"))
    da = draw()
    restore_rng(b["rng_state"], torch.device("cpu"))
    db = draw()
    equal_tree(da, db)


def test_cpu_training_resume_never_cuda(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU called CUDA")

    for key in (
        "is_available",
        "manual_seed_all",
        "get_rng_state_all",
        "set_rng_state_all",
        "device_count",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, key, forbidden)
    run(tmp_path / "run", 2)
    run(tmp_path / "run", 4, tmp_path / "run" / "last.pt")


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_logs_and_updates(tmp_path, dtype):
    run(tmp_path / "run", 2, dtype=dtype)
    events = [json.loads(x) for x in (tmp_path / "run" / "training.jsonl").read_text().splitlines()]
    updates = [e for e in events if e["event"] == "update"]
    assert len(updates) == 2
    for event in updates:
        assert event["loss"] >= 0 and event["gradient_norm_before_clip"] > 0
        assert event["message_opportunities"] == len(event["sample_ids"])
        for layer in event["diagnostics"]["layer_summaries"]:
            assert {
                "rho",
                "mu",
                "d_min",
                "ell_max",
                "alpha1",
                "alpha2",
                "gamma1",
                "gamma2",
                "candidate_edges",
                "message_opportunities",
            } <= layer.keys()
        assert event["diagnostics"]["layer_summaries"][-1]["message_opportunities"] == [0] * len(
            event["sample_ids"]
        )
    initial = PGVAMPDetector(2, dtype=torch.float64 if dtype == "complex128" else torch.float32)
    state = load_checkpoint(tmp_path / "run" / "last.pt")
    assert all(
        not torch.equal(state["model_state_dict"][k], v) for k, v in initial.state_dict().items()
    )


@pytest.mark.parametrize("failure", ["loss", "gradient", "cholesky"])
def test_failure_is_contextual_and_not_committed(tmp_path, monkeypatch, failure):
    import pgvamp_ofdm.training.trainer as trainer

    if failure == "loss":

        def fail(*args):
            raise FloatingPointError("nonfinite training loss")

        monkeypatch.setattr(trainer, "layer_loss", fail)
    elif failure == "gradient":
        original = PGVAMPDetector.forward

        def forward(self, *args, **kwargs):
            self.raw_mu.register_hook(lambda g: g * float("nan"))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(PGVAMPDetector, "forward", forward)
    else:
        original_cholesky = torch.linalg.cholesky_ex

        def fail_cholesky(a, *args, **kwargs):
            factor, info = original_cholesky(a, *args, **kwargs)
            return factor, torch.ones_like(info)

        monkeypatch.setattr(torch.linalg, "cholesky_ex", fail_cholesky)
    with pytest.raises((FloatingPointError, RuntimeError)):
        run(tmp_path / "run", 2)
    assert not (tmp_path / "run" / "last.pt").exists()
    event = json.loads((tmp_path / "run" / "run.json").read_text())
    assert event["status"] == "failed" and event["committed_step"] == 0
    assert event["sample_ids"] and event["H_norm"] > 0 and event["dtype"] == "complex128"


def test_resume_rejects_changed_contract_and_collision(tmp_path):
    run(tmp_path / "run", 2)
    with pytest.raises(ValueError, match="empty"):
        run(tmp_path / "run", 2)
    before = (tmp_path / "run" / "run.json").read_bytes()
    with pytest.raises(ValueError, match="contract"):
        run(tmp_path / "run", 4, tmp_path / "run" / "last.pt", dtype="complex64")
    assert before == (tmp_path / "run" / "run.json").read_bytes()


def test_rng_safe_roundtrip():
    state = copy.deepcopy(rng_state(torch.device("cpu")))
    restore_rng(state, torch.device("cpu"))
    equal_tree(state, rng_state(torch.device("cpu")))


def test_off_cadence_terminal_validation_does_not_change_resume(tmp_path):
    config = fixture_config(4)
    config.values["training"].update(
        validation_every_steps=3,
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_min_delta=1.0,
    )
    data = math_samples(27)
    train_samples(config, data, data, tmp_path / "whole", stable_hash("fixture"))
    partial = copy.deepcopy(config)
    partial.values["training"]["max_steps"] = 2
    train_samples(partial, data, data, tmp_path / "split", stable_hash("fixture"))
    train_samples(
        config,
        data,
        data,
        tmp_path / "split",
        stable_hash("fixture"),
        resume=tmp_path / "split" / "last.pt",
    )
    a, b = (load_checkpoint(tmp_path / name / "last.pt") for name in ("whole", "split"))
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "resume_validation_state",
        "best_validation_metric",
        "best_step",
        "bad_validation_events",
    ):
        equal_tree(a[key], b[key])
    assert (tmp_path / "split" / "terminal-best-before-resume-2.pt").exists()


def test_resume_new_output_copies_matching_best_and_rejects_rewind(tmp_path):
    run(tmp_path / "original", 2)
    import shutil

    shutil.copy2(tmp_path / "original" / "last.pt", tmp_path / "old.pt")
    run(tmp_path / "new", 4, tmp_path / "original" / "last.pt")
    assert (tmp_path / "new" / "best.pt").exists()
    run(tmp_path / "original", 4, tmp_path / "original" / "last.pt")
    with pytest.raises(ValueError, match="rewind"):
        run(tmp_path / "original", 4, tmp_path / "old.pt")


def test_skipped_validation_and_early_stop_state(tmp_path):
    config = fixture_config(6)
    config.values["training"].update(
        validation_every_steps=2,
        early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0,
    )
    data = math_samples(27)
    result = train_samples(config, data, data, tmp_path / "run", stable_hash("fixture"))
    assert result["step"] == 4
    events = [json.loads(s) for s in (tmp_path / "run" / "training.jsonl").read_text().splitlines()]
    updates = [e for e in events if e["event"] == "update"]
    assert updates[0]["validation_nmse"] is None and updates[0]["last_validation_step"] == 0
    assert updates[2]["validation_nmse"] is None and updates[2]["last_validation_step"] == 2


def test_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import pgvamp_ofdm.training.trainer as trainer

    original = trainer.layer_loss
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("injected second update failure")
        return original(*args)

    monkeypatch.setattr(trainer, "layer_loss", fail_second)
    with pytest.raises(FloatingPointError):
        run(tmp_path / "run", 2)
    assert load_checkpoint(tmp_path / "run" / "last.pt")["step"] == 1
    assert json.loads((tmp_path / "run" / "run.json").read_text())["committed_step"] == 1


def test_best_write_failure_can_resume_last(tmp_path, monkeypatch):
    import pgvamp_ofdm.training.trainer as trainer

    original = trainer.save_checkpoint

    def fail_best(path, value):
        if path.name == "best.pt":
            raise OSError("injected best artifact write failure")
        return original(path, value)

    monkeypatch.setattr(trainer, "save_checkpoint", fail_best)
    with pytest.raises(OSError):
        run(tmp_path / "run", 2)
    assert load_checkpoint(tmp_path / "run" / "last.pt")["step"] == 1
    monkeypatch.setattr(trainer, "save_checkpoint", original)
    run(tmp_path / "run", 4, tmp_path / "run" / "last.pt")
    assert load_checkpoint(tmp_path / "run" / "last.pt")["step"] == 4
    assert (tmp_path / "run" / "best.pt").exists()


def test_early_stopped_fresh_output_keeps_last(tmp_path):
    config = fixture_config(6)
    config.values["training"].update(
        validation_every_steps=1,
        early_stopping=True,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0,
    )
    data = math_samples(27)
    train_samples(config, data, data, tmp_path / "original", stable_hash("fixture"))
    state = load_checkpoint(tmp_path / "original" / "last.pt")
    assert state["step"] == 2
    result = train_samples(
        config,
        data,
        data,
        tmp_path / "new",
        stable_hash("fixture"),
        resume=tmp_path / "original" / "last.pt",
    )
    assert result["step"] == 2 and (tmp_path / "new" / "last.pt").exists()
