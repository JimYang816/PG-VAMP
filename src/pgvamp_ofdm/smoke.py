"""Acceptance-only smoke orchestration and integer counts, not performance evaluation."""

import copy
from pathlib import Path
from typing import Any

import torch

from .algorithms import MMSEDetector, VAMPDetector
from .config import Config, load_config
from .data.audit import audit_dataset
from .data.dataset import EffectiveDataset
from .data.generate import generate_dataset
from .data.manifest import file_hash, load_tensors, save_json, save_tensors
from .data.materialize import materialize
from .data.records import tensor_hash
from .inference import infer
from .modulation.qpsk import bits_to_symbols
from .training.checkpoint import load_checkpoint, model_for
from .training.trainer import batch_samples, train, train_samples
from .utils.device import resolve_runtime
from .utils.random import generator, stable_hash


def error_counts(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor,
    estimate: torch.Tensor,
    target: torch.Tensor,
    frame_ids: list[str],
    blocks: list[int],
    blocks_per_frame: int,
) -> dict[str, Any]:
    """Aggregate actual counts; only complete unique frames contribute to FER."""
    if (
        predicted_bits.shape != target_bits.shape
        or target_bits.ndim != 3
        or target_bits.shape[-1] != 2
    ):
        raise ValueError("counts require equal [B,N,2] bits")
    if (
        estimate.shape != target.shape
        or estimate.shape != target_bits.shape[:2]
        or not bool(torch.isfinite(estimate).all())
    ):
        raise ValueError("invalid estimates for counts")
    if len(frame_ids) != len(blocks) or len(blocks) != len(target) or blocks_per_frame < 1:
        raise ValueError("invalid frame count metadata")
    errors = predicted_bits != target_bits
    symbol = errors.any(-1)
    block_errors = symbol.any(-1)
    frames: dict[str, dict[int, bool]] = {}
    for i, (frame, block) in enumerate(zip(frame_ids, blocks)):
        group = frames.setdefault(frame, {})
        if block in group or not 0 <= block < blocks_per_frame:
            raise ValueError("duplicate/invalid frame block")
        group[block] = bool(block_errors[i])
    complete = [v for v in frames.values() if len(v) == blocks_per_frame]
    return {
        "bit_errors": int(errors.sum()),
        "bits": errors.numel(),
        "symbol_errors": int(symbol.sum()),
        "symbols": symbol.numel(),
        "block_errors": int(block_errors.sum()),
        "blocks": len(blocks),
        "frame_errors": sum(any(v.values()) for v in complete),
        "frames": len(complete),
        "incomplete_frames": len(frames) - len(complete),
        "squared_error_energy": float((estimate - target).abs().square().sum()),
        "target_energy": float(target.abs().square().sum()),
    }


def math_samples(seed: int, dtype: torch.dtype = torch.complex128) -> list[dict[str, Any]]:
    gen = generator(seed)
    rd = torch.float64 if dtype == torch.complex128 else torch.float32
    samples = []
    for index in range(3):
        h = torch.eye(32, dtype=dtype) + 0.08 * torch.randn(32, 32, dtype=dtype, generator=gen)
        bits = torch.randint(0, 2, (32, 2), dtype=torch.uint8, generator=gen)
        x = bits_to_symbols(bits, dtype=dtype)
        y = h @ x + 0.3 * torch.randn(32, dtype=dtype, generator=gen)
        samples.append(
            {
                "H": h,
                "y": y,
                "sigma2": torch.tensor(0.09, dtype=rd),
                "x": x,
                "bits": bits,
                "sample_id": f"algebra_fixture:{index}",
            }
        )
    return samples


def smoke(config: Config, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("smoke output already exists; choose a fresh path")
    output.mkdir(parents=True)
    config.save(output / "requested_config.yaml")
    rt = config.values["runtime"]
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    if config.mode == "algebra_fixture":
        samples = math_samples(config.values["seed"], runtime.complex_dtype)
        # The detector has no physical dimension parameter; this internal trainer
        # contract supplies its settings only. The fixture is explicitly recorded.
        working = load_config()
        working.values["runtime"] = copy.deepcopy(rt)
        working.values["seed"] = config.values["seed"]
        working.values["pg_vamp"]["depth"] = 2
        working.values["training"].update(max_steps=2, validation_every_steps=1)
        receipt = train_samples(
            working,
            samples,
            samples,
            output / "training",
            stable_hash({"algebra_fixture": config.values}),
            execution_kind="algebra_fixture",
        )
        batch = batch_samples(samples, [0], runtime)
        before = model_for(working, runtime)
        after = model_for(working, runtime)
        state = load_checkpoint(output / "training" / "last.pt")
        after.load_state_dict(state["model_state_dict"])
        if not all(not torch.equal(a, b) for a, b in zip(before.parameters(), after.parameters())):
            raise AssertionError("algebra updates did not change both parameter vectors")
        reloaded = model_for(working, runtime)
        reloaded.load_state_dict(
            load_checkpoint(output / "training" / "last.pt")["model_state_dict"]
        )
        with torch.inference_mode():
            a = after(batch["H"], batch["y"], batch["sigma2"]).x_soft
            b = reloaded(batch["H"], batch["y"], batch["sigma2"]).x_soft
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        report = {
            "scope": "algebra_fixture only; no physical/performance claim",
            "dimension": 32,
            "depth": 2,
            "updates": receipt["step"],
            "trainable_scalars": 4,
            "checkpoint_roundtrip_max_abs": float((a - b).abs().max()),
        }
    else:
        c = config.values
        if (
            c["waveform"]["n_grid"] != 512
            or c["waveform"]["n_data"] != 400
            or c["waveform"]["n_fft_wave"] != 8192
            or c["waveform"]["cp_samples"] != 2048
            or c["frame"]["n_ofdm_symbols"] != 8
            or c["pg_vamp"]["depth"] != 8
            or c["training"]["max_steps"] != 2
        ):
            raise ValueError("system smoke requires 512/400/8192/2048/8/T8/two updates")
        manifest = generate_dataset(config, output / "data")
        audit = audit_dataset(manifest, output / "audit", waveform_frames=1)
        audit_artifact = load_tensors(output / "audit" / "frame-000.pt")
        epsilon = audit_artifact["record"]["path_epsilon"]
        if not bool((epsilon != 0).any()) or torch.unique(epsilon).numel() < 2:
            raise AssertionError("system smoke needs distinct nonzero path time scaling")
        receipt = train(config, manifest, output / "training")
        dataset = EffectiveDataset(manifest, "val")
        batch = batch_samples(dataset, list(range(c["frame"]["n_ofdm_symbols"])), runtime)
        pg = model_for(config, runtime)
        pg.load_state_dict(load_checkpoint(output / "training" / "last.pt")["model_state_dict"])
        shared_hash = tensor_hash({k: batch[k].cpu() for k in ("H", "y", "sigma2")})
        counts: dict[str, Any] = {}
        saved: dict[str, Any] = {
            "sample_ids": batch["sample_ids"],
            "shared_input_hash": shared_hash,
            **{k: batch[k].cpu() for k in ("H", "y", "sigma2", "x")},
        }
        target_bits = torch.stack([dataset[i]["bits"] for i in range(len(batch["sample_ids"]))])
        frame_ids = [dataset[i]["frame_id"] for i in range(len(batch["sample_ids"]))]
        with torch.inference_mode():
            for name, detector in (
                ("mmse", MMSEDetector()),
                ("vamp", VAMPDetector(iterations=c["vamp"]["iterations"])),
                ("pg_vamp", pg),
            ):
                result = detector.detect(batch["H"], batch["y"], batch["sigma2"])
                counts[name] = {
                    "input_hash": shared_hash,
                    **error_counts(
                        result.bits_hat.cpu(),
                        target_bits,
                        result.x_soft.cpu(),
                        batch["x"].cpu(),
                        frame_ids,
                        list(range(len(frame_ids))),
                        c["frame"]["n_ofdm_symbols"],
                    ),
                }
                saved[name] = {"x_soft": result.x_soft.cpu(), "bits_hat": result.bits_hat.cpu()}
        saved.update(target_bits=target_bits, frame_ids=frame_ids)
        save_tensors(output / "shared_predictions.pt", saved)
        materialize(manifest, "val", output / "unlabeled.pt", labeled=False)
        infer(
            output / "training" / "last.pt",
            output / "unlabeled.pt",
            output / "inference.pt",
            device=str(runtime.device),
        )
        prediction = load_tensors(output / "inference.pt")
        torch.testing.assert_close(
            prediction["x_soft"][: len(frame_ids)],
            saved["pg_vamp"]["x_soft"],
            atol=1e-9 if rt["dtype"] == "complex128" else 2e-5,
            rtol=1e-8 if rt["dtype"] == "complex128" else 2e-4,
        )
        report = {
            "scope": "smoke_system; no performance/convergence claim",
            "n_grid": 512,
            "n_data": 400,
            "n_fft_wave": c["waveform"]["n_fft_wave"],
            "cp_samples": c["waveform"]["cp_samples"],
            "blocks_per_frame": c["frame"]["n_ofdm_symbols"],
            "depth": 8,
            "updates": receipt["step"],
            "audit": audit,
            "audit_path_epsilon": epsilon.tolist(),
            "counts": counts,
            "checkpoint_roundtrip": "passed",
        }
    report.update(
        status="passed",
        device=str(runtime.device),
        dtype=rt["dtype"],
        checkpoint_sha256=file_hash(output / "training" / "last.pt"),
    )
    save_json(output / "smoke.json", report)
    return report
