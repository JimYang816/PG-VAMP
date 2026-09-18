"""Deterministic sampling, supervised PG updates and complete strict resume."""

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import torch

from ..config import Config
from ..data.dataset import EffectiveDataset
from ..data.manifest import file_hash, save_json
from ..data.records import allocation_metadata
from ..utils.device import Runtime, resolve_runtime
from ..utils.random import generator, seed_runtime, stable_hash
from .checkpoint import (
    QPSK_MAPPING,
    load_checkpoint,
    model_for,
    physical_contract,
    restore_rng,
    resume_contract,
    rng_state,
    save_checkpoint,
)
from .losses import layer_loss
from .optimizer import DeviceExplicitAdam


class Samples(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Any]: ...


def batch_samples(dataset: Samples, indices: list[int], runtime: Runtime) -> dict[str, Any]:
    samples = [dataset[i] for i in indices]
    if not samples or any("x" not in s or "bits" not in s for s in samples):
        raise ValueError("training/validation requires nonempty labeled samples")
    result: dict[str, Any] = {
        "sample_ids": [s["sample_id"] for s in samples],
        "source_dtype": str(samples[0]["H"].dtype),
    }
    for key in ("H", "y", "sigma2", "x"):
        result[key] = torch.stack([s[key] for s in samples]).to(
            device=runtime.device,
            dtype=runtime.real_dtype if key == "sigma2" else runtime.complex_dtype,
        )
    return result


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items() if k != "layer_outputs"}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def append_log(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(event), allow_nan=False) + "\n")
        handle.flush()


def train_samples(
    config: Config,
    train_data: Samples,
    val_data: Samples,
    output: Path,
    manifest_hash: str,
    *,
    resume: Path | None = None,
    execution_kind: str = "physical",
) -> dict[str, Any]:
    """Internal shared engine; physical entry point below validates manifests first."""
    from ..cli import _provenance

    if len(train_data) < 1 or len(val_data) < 1:
        raise ValueError("train and validation splits must be nonempty")
    c, rt = config.values, config.values["runtime"]
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    settings = c["training"]
    checkpoint = load_checkpoint(resume) if resume is not None else None
    resume_best = None
    if checkpoint is not None:
        if checkpoint["execution_kind"] != execution_kind:
            raise ValueError("resume execution kind mismatch")
        if output.exists() and (output / "last.pt").exists():
            existing = load_checkpoint(output / "last.pt")
            if existing["step"] > checkpoint["step"]:
                raise ValueError("cannot rewind an existing run; use a fresh output")
        rv = checkpoint["resume_validation_state"]
        if rv["best_step"]:
            if rv["best_step"] == checkpoint["step"]:
                resume_best = checkpoint
            else:
                assert resume is not None
                resume_best = load_checkpoint(resume.parent / "scheduled-best.pt")
            if (
                resume_best["run_id"] != checkpoint["run_id"]
                or resume_best["step"] != rv["best_step"]
                or resume_best["best_validation_metric"] != rv["best_validation_metric"]
            ):
                raise ValueError("matching scheduled best checkpoint required for strict resume")
        if checkpoint["contract_hash"] != stable_hash(resume_contract(config)):
            raise ValueError("strict resume configuration/dtype contract mismatch")
        if any(
            checkpoint[k] != manifest_hash
            for k in ("training_manifest_hash", "validation_manifest_hash")
        ):
            raise ValueError("strict resume manifest identity mismatch")
        if (
            checkpoint["sampler"]["size"] != len(train_data)
            or settings["max_steps"] <= checkpoint["step"]
            or settings["max_steps"] < checkpoint["resolved_config"]["training"]["max_steps"]
        ):
            raise ValueError("strict resume requires matching dataset size and a higher max_steps")
        if output.exists() and any(output.iterdir()):
            identity = output / "run.json"
            if (
                not identity.exists()
                or json.loads(identity.read_text())["run_id"] != checkpoint["run_id"]
            ):
                raise ValueError("resume output belongs to another run")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("new training output must be empty")
    seed_runtime(c["seed"], runtime.device)
    model = model_for(config, runtime)
    optimizer = DeviceExplicitAdam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    sampler = generator(c["seed"])
    permutation = torch.randperm(len(train_data), generator=sampler)
    step = epoch = cursor = bad = best_step = last_val = 0
    best: float | None = None
    run_id = stable_hash(uuid.uuid4().hex)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        permutation = checkpoint["sampler"]["permutation"]
        sampler.set_state(checkpoint["sampler"]["generator_state"])
        step, epoch, cursor = (checkpoint[k] for k in ("step", "epoch", "sampler_position"))
        best, best_step, bad, last_val = (
            checkpoint["resume_validation_state"][k]
            for k in (
                "best_validation_metric",
                "best_step",
                "bad_validation_events",
                "last_validation_step",
            )
        )
        run_id = checkpoint["run_id"]
        restore_rng(checkpoint["rng_state"], runtime.device)
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        if not (output / "last.pt").exists():
            save_checkpoint(output / "last.pt", checkpoint)
        # Terminal off-cadence validation is a real measurement for this run's
        # best artifact, but not a scheduled early-stop event after resuming.
        if (output / "best.pt").exists() and checkpoint["last_validation_step"] != checkpoint[
            "resume_validation_state"
        ]["last_validation_step"]:
            archive = output / f"terminal-best-before-resume-{checkpoint['step']}.pt"
            if archive.exists():
                raise ValueError("terminal best archive already exists; use fresh output")
            (output / "best.pt").replace(archive)
        if resume_best is not None:
            save_checkpoint(output / "best.pt", resume_best)
            save_checkpoint(output / "scheduled-best.pt", resume_best)
    config.save(output / "resolved_config.yaml")
    environment = _provenance(runtime)
    environment["torch"] = str(environment["torch"])
    save_json(output / "environment.json", environment)
    save_json(output / "run.json", {"run_id": run_id, "status": "running", "committed_step": step})
    log = output / "training.jsonl"
    append_log(
        log,
        {
            "event": "resume" if checkpoint else "start",
            "run_id": run_id,
            "step": step,
            "invalidates_prior_events_after_step": step if checkpoint else None,
            "runtime": rt,
            "previous_device": checkpoint["device_used_for_training"] if checkpoint else None,
        },
    )
    batch: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    try:
        while step < settings["max_steps"] and not (
            settings["early_stopping"] and bad >= settings["early_stopping_patience"]
        ):
            started = time.perf_counter()
            if cursor == len(train_data):
                epoch += 1
                cursor = 0
                permutation = torch.randperm(len(train_data), generator=sampler)
            indices = permutation[cursor : cursor + settings["batch_size"]].tolist()
            batch = batch_samples(train_data, indices, runtime)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["H"], batch["y"], batch["sigma2"], return_layer_outputs=True)
            diagnostics = result.diagnostics
            loss = layer_loss(diagnostics["layer_outputs"], batch["x"])
            loss.backward()
            if any(
                p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()
            ):
                raise FloatingPointError("nonfinite or missing parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("nonfinite updated parameter")
            next_step = step + 1
            metric: float | None = None
            improved = False
            scheduled = next_step % settings["validation_every_steps"] == 0
            resume_validation = {
                "best_validation_metric": best,
                "best_step": best_step,
                "bad_validation_events": bad,
                "last_validation_step": last_val,
            }
            if (
                next_step % settings["validation_every_steps"] == 0
                or next_step == settings["max_steps"]
            ):
                model.eval()
                error = energy = 0.0
                with torch.inference_mode():
                    for start in range(
                        0,
                        min(len(val_data), settings["validation_max_blocks"]),
                        settings["batch_size"],
                    ):
                        stop = min(
                            start + settings["batch_size"],
                            len(val_data),
                            settings["validation_max_blocks"],
                        )
                        vb = batch_samples(val_data, list(range(start, stop)), runtime)
                        try:
                            prediction = model(vb["H"], vb["y"], vb["sigma2"])
                        except (ValueError, RuntimeError, FloatingPointError):
                            batch = vb
                            raise
                        ve = (prediction.x_soft - vb["x"]).abs().square().sum()
                        if not bool(torch.isfinite(ve)):
                            batch = vb
                            raise FloatingPointError("nonfinite validation output")
                        error += float(ve)
                        energy += float(vb["x"].abs().square().sum())
                if energy <= 0:
                    raise ValueError("validation target energy must be positive")
                metric = error / energy
                last_val = next_step
                improved = best is None or metric < best
                significant = best is None or metric < best - settings["early_stopping_min_delta"]
                bad = 0 if significant else bad + 1
                if improved:
                    best, best_step = metric, next_step
            if scheduled:
                resume_validation = {
                    "best_validation_metric": best,
                    "best_step": best_step,
                    "bad_validation_events": bad,
                    "last_validation_step": last_val,
                }
            cursor += len(indices)
            saved_rng = rng_state(runtime.device)
            if checkpoint is not None and runtime.device.type == "cpu":
                saved_rng["cuda"] = checkpoint["rng_state"]["cuda"]
            state = {
                "schema_version": 1,
                "execution_kind": execution_kind,
                "resume_validation_state": resume_validation,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "resolved_config": c,
                "algorithm_config": c["pg_vamp"],
                "waveform_config": physical_contract(config),
                "subcarrier_mapping": allocation_metadata(config),
                "qpsk_mapping": QPSK_MAPPING,
                "step": next_step,
                "epoch": epoch,
                "sampler_position": cursor,
                "sampler": {
                    "permutation": permutation,
                    "generator_state": sampler.get_state(),
                    "size": len(train_data),
                },
                "random_seed": c["seed"],
                "rng_state": saved_rng,
                "best_validation_metric": best,
                "best_step": best_step,
                "bad_validation_events": bad,
                "last_validation_step": last_val,
                "training_manifest_hash": manifest_hash,
                "validation_manifest_hash": manifest_hash,
                "dtype": rt["dtype"],
                "device_used_for_training": str(runtime.device),
                "mask_mode": "soft",
                "environment": environment,
                "run_id": run_id,
                "contract_hash": stable_hash(resume_contract(config)),
            }
            opportunities = len(indices) * max(model.depth - 1, 0)
            append_log(
                log,
                {
                    "event": "update",
                    "run_id": run_id,
                    "step": next_step,
                    "epoch": epoch,
                    "sample_ids": batch["sample_ids"],
                    "source_dtype": batch["source_dtype"],
                    "training_dtype": rt["dtype"],
                    "loss": float(loss.detach()),
                    "validation_nmse": metric,
                    "last_validation_step": last_val,
                    "best_step": best_step,
                    "gradient_norm_before_clip": float(norm),
                    "diagnostics": diagnostics,
                    "message_opportunities": opportunities,
                    "rejection_rate": int(diagnostics["message_rejected"].sum())
                    / max(opportunities, 1),
                    "precision_cap_rate": int(diagnostics["precision_capped"].sum())
                    / max(opportunities, 1),
                    "no_information_opportunities": len(indices) * model.depth,
                    "no_information_rate": int(diagnostics["no_information"].sum())
                    / (len(indices) * model.depth),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            # Commit the self-contained update before replacing best artifacts.
            # If a later best write fails, resume can rebuild that best from last.
            save_checkpoint(output / "last.pt", state)
            step = next_step
            if improved:
                save_checkpoint(output / "best.pt", state)
                if scheduled:
                    save_checkpoint(output / "scheduled-best.pt", state)
            save_json(
                output / "run.json", {"run_id": run_id, "status": "running", "committed_step": step}
            )
        summary = {
            "run_id": run_id,
            "status": "complete",
            "step": step,
            "best_step": best_step,
            "best_validation_nmse": best,
            "last_checkpoint": str(output / "last.pt"),
        }
        save_json(output / "run.json", summary)
        return summary
    except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
        failure = {
            "event": "failure",
            "run_id": run_id,
            "status": "failed",
            "committed_step": step,
            "sample_ids": batch.get("sample_ids", []),
            "dtype": rt["dtype"],
            "device": str(runtime.device),
            "H_max_abs": batch["H"].abs().amax().item() if "H" in batch else None,
            "H_norm": torch.linalg.vector_norm(batch["H"]).item() if "H" in batch else None,
            "diagnostics": diagnostics,
            "error": str(exc),
        }
        append_log(log, failure)
        save_json(output / "run.json", json_value(failure))
        raise


def train(
    config: Config, manifest: Path, output: Path, *, resume: Path | None = None
) -> dict[str, Any]:
    if config.mode != "physical":
        raise ValueError("train requires physical data; use smoke for algebra fixtures")
    train_data, val_data = EffectiveDataset(manifest, "train"), EffectiveDataset(manifest, "val")
    if physical_contract(config) != physical_contract(train_data.config):
        raise ValueError("training physics/mapping differs from manifest")
    return train_samples(config, train_data, val_data, output, file_hash(manifest), resume=resume)
