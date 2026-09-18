"""Fixed complete-frame evaluation from one shared physical replay."""

import json
import math
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from ..algorithms import MMSEDetector, VAMPDetector
from ..algorithms.prepared import Detector
from ..channel.affine import affine_waveform
from ..channel.effective_matrix import effective_matrix
from ..channel.noise import complex_awgn
from ..config import Config, config_from_values
from ..data.dataset import EffectiveDataset, detection_inputs
from ..data.manifest import file_hash, save_tensors
from ..data.records import record_frame, record_paths, tensor_hash
from ..modulation.qpsk import bits_to_symbols
from ..receiver.fft_receiver import fft_receive
from ..receiver.synchronization import normalized_correlation
from ..training.checkpoint import load_checkpoint, model_for, physical_contract
from ..utils.device import Runtime, resolve_runtime
from ..utils.random import derive_seed, generator, stable_hash
from .artifacts import begin, csv_write, json_write, publish
from .metrics import aggregate, block_counts, frame_counts
from .statistics import bootstrap
from .timing import MODES, measure_detector, synchronize


def setup(
    config: Config,
    manifest: Path,
    checkpoint: Path | None,
    algorithms: list[str],
    allow_untrained: bool,
) -> tuple[EffectiveDataset, Runtime, dict[str, Detector], dict[str, Any]]:
    if config.mode != "physical" or len(set(algorithms)) != len(algorithms) or not algorithms:
        raise ValueError("evaluation needs physical config and distinct algorithms")
    if not set(algorithms) <= {"mmse", "vamp", "pg_vamp"}:
        raise ValueError("unknown evaluation algorithm")
    c, rt = config.values, config.values["runtime"]
    if c["evaluation"]["label"] == "main_simulation" and (
        ("pg_vamp" in algorithms and c["pg_vamp"]["depth"] != 8)
        or ("vamp" in algorithms and c["vamp"]["iterations"] != 8)
    ):
        raise ValueError("main_simulation requires PG depth 8 and VAMP iterations 8")
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    data = EffectiveDataset(manifest, "test", device=str(runtime.device))
    if physical_contract(config) != physical_contract(data.config):
        raise ValueError("evaluation physics/mapping differs from manifest")
    for key in ("scenarios", "esn0_db", "frames_per_cell"):
        if c["evaluation"][key] != data.config.values["evaluation"][key]:
            raise ValueError(f"fixed evaluation plan differs from manifest: {key}")
    if c["seed"] != data.config.values["seed"]:
        raise ValueError("test seed differs from manifest; generate that test population first")
    if not data.records or data.blocks != 8:
        raise ValueError("evaluation requires nonempty eight-block frames")
    expected = {
        (s, float(db)) for s in c["evaluation"]["scenarios"] for db in c["evaluation"]["esn0_db"]
    }
    cells: dict[tuple[str, float], set[str]] = defaultdict(set)
    for r in data.records:
        snr = r["esn0_db"].tolist()
        if len(snr) != 8 or len(set(snr)) != 1:
            raise ValueError("frame SNR must be constant across eight blocks")
        cell = (r["scenario"], float(snr[0]))
        if r["frame_id"] in cells[cell]:
            raise ValueError("duplicate frame in evaluation cell")
        cells[cell].add(r["frame_id"])
    if set(cells) != expected or any(
        len(v) != c["evaluation"]["frames_per_cell"] for v in cells.values()
    ):
        raise ValueError("missing/unexpected frames in fixed evaluation plan")
    # Dataset storage dtype remains part of its provenance. Explicit conversion
    # happens once, in the shared caller projection for all detectors.
    models: dict[str, Detector] = {}
    metadata: dict[str, Any] = {
        "status": "not_requested",
        "checkpoint_hash": None,
        "train_seed": None,
        "training_steps": None,
    }
    if "mmse" in algorithms:
        models["MMSE (linear)"] = MMSEDetector().eval()
    if "vamp" in algorithms:
        models[f"VAMP-{c['vamp']['iterations']}"] = VAMPDetector(c["vamp"]["iterations"]).eval()
    if "pg_vamp" in algorithms:
        model = model_for(config, runtime).eval()
        initial = [v.detach().cpu().tolist() for v in model.thresholds()]
        name = "PG-VAMP-VC"
        if checkpoint is None:
            if not allow_untrained:
                raise ValueError("PG-VAMP requires --checkpoint or explicit --allow-untrained")
            name = "PG-VAMP-untrained"
            metadata["status"] = "untrained"
        else:
            state = load_checkpoint(checkpoint)
            saved = config_from_values(state["resolved_config"])
            if state["execution_kind"] != "physical" or physical_contract(
                saved
            ) != physical_contract(config):
                raise ValueError("checkpoint physics/execution kind mismatch")
            if saved.values["pg_vamp"] != c["pg_vamp"]:
                raise ValueError("checkpoint PG settings mismatch")
            if state["dtype"] != rt["dtype"]:
                raise ValueError(
                    "evaluation checkpoint dtype mismatch; keep precision experiments separate"
                )
            model.load_state_dict(state["model_state_dict"], strict=True)
            metadata.update(
                status="trained",
                checkpoint_hash=file_hash(checkpoint),
                train_seed=state["random_seed"],
                training_steps=state["step"],
                checkpoint_run_id=state["run_id"],
                best_step=state["best_step"],
                training_scenarios=saved.values["data"]["train_scenario_weights"],
                training_manifest_hash=state["training_manifest_hash"],
            )
        metadata.update(
            initial_rho=initial[0],
            initial_mu=initial[1],
            learned_rho=model.thresholds()[0].detach().cpu().tolist(),
            learned_mu=model.thresholds()[1].detach().cpu().tolist(),
        )
        models[name] = model
    # EffectiveDataset resolves its storage runtime; reset requested compute threads.
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    return data, runtime, models, metadata


def shared_inputs(sample: dict[str, Any], runtime: Runtime) -> dict[str, torch.Tensor]:
    return {
        k: v.to(
            runtime.device, runtime.real_dtype if k == "sigma2" else runtime.complex_dtype
        ).unsqueeze(0)
        for k, v in detection_inputs(sample).items()
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def ici(H: torch.Tensor) -> float:
    energy = H.abs().square()
    off = energy.clone()
    off.diagonal(dim1=-2, dim2=-1).zero_()
    return float(off.sum() / energy.sum().clamp_min(torch.finfo(energy.dtype).eps))


def physical_example(
    data: EffectiveDataset,
    record: dict[str, Any],
    sample: dict[str, Any],
    destination: Path,
    input_hash: str,
) -> dict[str, Any]:
    """One real physical record; bounded persisted examples, no detector reruns."""
    frame = record_frame(record, data.config, data.runtime)
    paths = record_paths(record, data.runtime.device)
    h = effective_matrix(
        paths, frame.layout, data.config, sample["block"], dtype=data.runtime.complex_dtype
    )
    arrival = record["arrival_offset_samples"]
    times = (
        torch.arange(
            frame.layout.frame_samples + arrival, dtype=torch.float64, device=data.runtime.device
        )
        / frame.layout.sample_rate_hz
    )
    received = affine_waveform(frame, paths, times, data.config, arrival_offset_samples=arrival)[0]
    received = received.to(data.runtime.complex_dtype)
    template_times = (
        torch.arange(frame.lfm.analytic.numel(), dtype=torch.float64, device=data.runtime.device)
        / frame.layout.sample_rate_hz
    )
    template = (
        frame.lfm.analytic
        * torch.exp(-2j * math.pi * data.config.values["waveform"]["carrier_hz"] * template_times)
    ).to(data.runtime.complex_dtype)
    sync = normalized_correlation(
        received, template, threshold=data.config.values["receiver"]["lfm_threshold"]
    )
    save_tensors(
        destination,
        {
            "sample_id": sample["sample_id"],
            "input_hash": input_hash,
            "h_grid": h.cpu(),
            "H": sample["H"].cpu(),
            "transmit_real": frame.real[0].cpu(),
            "received_iq": received.cpu(),
            "sync_scores": sync.scores.cpu(),
            "sync_peak": int(sync.peak_start),
            "sync_detected": bool(sync.detected),
            "arrival_offset_samples": arrival,
            "expected_template_start": arrival + frame.layout.lfm[0],
            "sync_threshold": sync.threshold,
            "sync_signal_domain": "ideal_complex_iq_baseband",
            "sample_rate_hz": frame.layout.sample_rate_hz,
            "sync_role": "diagnostic matched-correlation only; evaluation uses oracle_timing",
        },
    )
    return {
        "sample_id": sample["sample_id"],
        "grid_ici_ratio": ici(h),
        "effective_ici_ratio": ici(sample["H"]),
    }


def timing_observations(
    data: EffectiveDataset,
    record: dict[str, Any],
    sample: dict[str, Any],
    runtime: Runtime,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Independent ideal-I/Q waveform-noise draws, followed by actual FFT/data rows."""
    frame = record_frame(record, data.config, data.runtime)
    inp = shared_inputs(sample, runtime)
    target = bits_to_symbols(
        record["data_bits"][sample["block"]].to(runtime.device), dtype=runtime.complex_dtype
    )
    clean = inp["H"][0] @ target
    ys = []
    for i in range(count):
        noise = complex_awgn(
            (data.config.values["waveform"]["n_fft_wave"],),
            float(sample["sigma2"]),
            gen_r=generator(
                derive_seed(seed, "benchmark_noise_real", sample["sample_id"], i), runtime.device
            ),
            gen_i=generator(
                derive_seed(seed, "benchmark_noise_imag", sample["sample_id"], i), runtime.device
            ),
            dtype=runtime.complex_dtype,
        )
        observed = fft_receive(noise, frame.allocation)
        ys.append(clean + observed[frame.allocation.data_grid_index])
    return torch.stack(ys)


def _benchmark_sample(
    data: EffectiveDataset,
    models: dict[str, Detector],
    sample: dict[str, Any],
    record: dict[str, Any],
    runtime: Runtime,
    settings: dict[str, Any],
    modes: tuple[str, ...],
    seed: int,
) -> list[dict[str, Any]]:
    inp = shared_inputs(sample, runtime)
    count = max(settings["batch_size"], 2) * settings["timing_repeats"]
    ys = timing_observations(data, record, sample, runtime, count, seed)
    rows = []
    for name, model in models.items():
        for mode in modes:
            try:
                measured = measure_detector(
                    model,
                    inp["H"],
                    ys,
                    inp["sigma2"],
                    mode=mode,
                    batch_size=settings["batch_size"],
                    warmup=settings["timing_warmup"],
                    repeats=settings["timing_repeats"],
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                measured = [
                    {"status": "incomplete_or_failed", "reason": str(exc), "timing_mode": mode}
                ]
            for row in measured:
                row.update(
                    algorithm=name,
                    sample_id=sample["sample_id"],
                    H_hash=tensor_hash({"H": inp["H"]}),
                    noise_observations_hash=tensor_hash({"y": ys}),
                    noise_seed=seed,
                    scenario=record["scenario"],
                    esn0_db=float(record["esn0_db"][0]),
                )
                rows.append(row)
    return rows


def evaluate(
    config: Config,
    manifest: Path,
    output: Path,
    *,
    checkpoint: Path | None = None,
    algorithms: list[str] | None = None,
    allow_untrained: bool = False,
    bootstrap_seed: int | None = None,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Publish counts/diagnostics/timing, retaining every predeclared block opportunity."""
    from ..cli import _provenance

    c, settings = config.values, config.values["evaluation"]
    data, runtime, models, checkpoint_meta = setup(
        config, manifest, checkpoint, algorithms or settings["algorithms"], allow_untrained
    )
    seed = (
        derive_seed(c["seed"], "frame_cluster_bootstrap")
        if bootstrap_seed is None
        else bootstrap_seed
    )
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be nonnegative")
    pending = begin(output)
    run_id, manifest_hash = uuid.uuid4().hex, file_hash(manifest)
    config.save(pending / "resolved_config.yaml")
    json_write(pending / "environment.json", {**_provenance(runtime), "argv": argv or sys.argv})
    json_write(
        pending / "dataset_manifest_hashes.json",
        {"manifest_hash": manifest_hash, "manifest": data.manifest},
    )
    json_write(pending / "checkpoint_metadata.json", checkpoint_meta)
    per_frame: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, float], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    diagnostics_by_cell: dict[tuple[str, str, float], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    lineage_digest = []
    measured_cells: set[tuple[str, float]] = set()
    examples: list[dict[str, Any]] = []
    common = {
        "run_id": run_id,
        "device": str(runtime.device),
        "dtype": str(runtime.complex_dtype),
        "test_seed": data.config.values["seed"],
        "manifest_hash": manifest_hash,
        "cpu_threads": runtime.cpu_threads,
        "experiment_label": settings["label"],
    }
    with (
        (pending / "lineage.jsonl").open("w", encoding="utf-8") as lineage,
        (pending / "diagnostics.jsonl").open("w", encoding="utf-8") as diagnostics,
        (pending / "failures.jsonl").open("w", encoding="utf-8") as failures,
        torch.inference_mode(),
    ):
        for record_index, record in enumerate(data.records):
            scenario, db = record["scenario"], float(record["esn0_db"][0])
            counts: dict[str, list[dict[str, Any] | None]] = {name: [] for name in models}
            sample_ids, hashes = [], []
            for block in range(8):
                synchronize(runtime.device)
                start = perf_counter()
                sample = data[record_index * 8 + block]
                inp = shared_inputs(sample, runtime)
                synchronize(runtime.device)
                replay_ms = (perf_counter() - start) * 1000
                identity = {
                    "sample_id": sample["sample_id"],
                    "frame_id": record["frame_id"],
                    "scenario": scenario,
                    "esn0_db": db,
                    "block": block,
                    "input_hash": tensor_hash(inp),
                    "algorithms": list(models),
                }
                lineage.write(json.dumps(identity) + "\n")
                lineage_digest.append(identity)
                sample_ids.append(sample["sample_id"])
                hashes.append(identity["input_hash"])
                for name, model in models.items():
                    context = {
                        **identity,
                        "algorithm": name,
                        "dtype": common["dtype"],
                        "device": common["device"],
                    }
                    try:
                        owned = {k: v.clone() for k, v in inp.items()}
                        result = model.detect(owned["H"], owned["y"], owned["sigma2"])
                        if tensor_hash(owned) != identity["input_hash"]:
                            raise RuntimeError("detector mutated shared inputs")
                        for field in (
                            result.x_soft,
                            result.bits_hat,
                            result.class_hat,
                            result.probabilities,
                        ):
                            if field is not None and not bool(torch.isfinite(field).all()):
                                raise FloatingPointError("final nonfinite output")
                        measured_counts = block_counts(
                            result.bits_hat[0],
                            result.x_soft[0],
                            sample["bits"],
                            sample["x"].to(runtime.complex_dtype),
                        )
                        diag = jsonable(result.diagnostics)
                        depth = getattr(model, "depth", getattr(model, "iterations", 0))
                        diag.update(
                            context,
                            status="complete",
                            message_opportunities=max(depth - 1, 0),
                            no_information_opportunities=depth,
                            effective_ici_ratio=ici(inp["H"]),
                        )
                        diagnostics.write(json.dumps(diag, allow_nan=False) + "\n")
                        sums = diagnostics_by_cell[name, scenario, db]
                        for counter in ("message_rejected", "precision_capped", "no_information"):
                            sums[counter] += sum(diag.get(counter, [0]))
                        sums["message_opportunities"] += diag["message_opportunities"]
                        sums["no_information_opportunities"] += diag["no_information_opportunities"]
                        counts[name].append(measured_counts)
                    except (FloatingPointError, RuntimeError, ValueError) as exc:
                        counts[name].append(None)
                        category = (
                            "final_nonfinite_outputs"
                            if "final nonfinite" in str(exc)
                            else "decomposition_failures"
                            if any(
                                word in str(exc).lower()
                                for word in ("cholesky", "svd", "factorization")
                            )
                            else "other_hard_failures"
                        )
                        diagnostics_by_cell[name, scenario, db][category] += 1
                        failure = {
                            **context,
                            "stage": "detection",
                            "exception": type(exc).__name__,
                            "reason": str(exc),
                            "category": category,
                            "H_norm": float(torch.linalg.vector_norm(inp["H"])),
                            "status": "incomplete_or_failed",
                            "partial_diagnostics": "unavailable",
                            "message_opportunities": None,
                            "no_information_opportunities": None,
                        }
                        failures.write(json.dumps(failure, allow_nan=False) + "\n")
                        diagnostics.write(json.dumps(failure, allow_nan=False) + "\n")
                timing_rows.append(
                    {
                        **common,
                        "algorithm": "shared_preprocessing",
                        "scenario": scenario,
                        "esn0_db": db,
                        "sample_id": sample["sample_id"],
                        "batch_size": 1,
                        "mean_latency_ms": replay_ms,
                        "status": "complete",
                        "timing_mode": "shared_replay_preprocessing",
                        "backend": data.config.values["data"]["backend"],
                        "includes": "compact_replay,H_build_or_cache,noise_FFT,"
                        "pilot_elimination,shared_conversion",
                        "sync": "oracle_timing; synchronization search not executed",
                    }
                )
                if block == 0 and (scenario, db) not in measured_cells:
                    timing_rows.extend(
                        {**common, **r}
                        for r in _benchmark_sample(
                            data, models, sample, record, runtime, settings, MODES, seed
                        )
                    )
                    measured_cells.add((scenario, db))
                if not examples:
                    examples.append(
                        physical_example(
                            data, record, sample, pending / "example.pt", identity["input_hash"]
                        )
                    )
            for name in models:
                row = {
                    **common,
                    "algorithm": name,
                    "scenario": scenario,
                    "esn0_db": db,
                    "train_seed": checkpoint_meta["train_seed"] if name.startswith("PG") else None,
                    "checkpoint_hash": checkpoint_meta["checkpoint_hash"]
                    if name.startswith("PG")
                    else None,
                    "frame_id": record["frame_id"],
                    "sample_ids": json.dumps(sample_ids),
                    "input_hashes": json.dumps(hashes),
                    **frame_counts(counts[name], c["waveform"]["n_data"]),
                }
                per_frame.append(row)
                groups[scenario, db][name].append(row)
    for (scenario, db), algorithms_frames in groups.items():
        intervals, paired = bootstrap(
            algorithms_frames,
            seed=derive_seed(seed, "cell", scenario, db),
            repeats=settings["frame_cluster_bootstrap_repeats"],
        )
        paired_rows.extend({**common, "scenario": scenario, "esn0_db": db, **r} for r in paired)
        for name, frames in algorithms_frames.items():
            row = {
                k: frames[0][k]
                for k in (
                    *common.keys(),
                    "algorithm",
                    "scenario",
                    "esn0_db",
                    "train_seed",
                    "checkpoint_hash",
                )
            }
            row.update(aggregate(frames), **intervals[name])
            ds = diagnostics_by_cell[name, scenario, db]
            row.update(
                {
                    key: ds[key]
                    for key in (
                        "final_nonfinite_outputs",
                        "decomposition_failures",
                        "other_hard_failures",
                    )
                }
            )
            row["unavailable_samples"] = row["hard_failures"]
            opportunities = ds["message_opportunities"]
            info_opportunities = ds["no_information_opportunities"]
            for key, rate in (
                ("message_rejected", "message_reject_rate"),
                ("precision_capped", "precision_cap_rate"),
                ("no_information", "no_information_rate"),
            ):
                total = ds[key]
                denom = info_opportunities if key == "no_information" else opportunities
                row[key] = total
                row[rate] = total / denom if denom else None
            row.update(
                message_opportunities=opportunities,
                no_information_opportunities=info_opportunities,
                diagnostic_missing_blocks=row["hard_failures"],
                diagnostics_scope="successful_blocks_only"
                if row["hard_failures"]
                else "all_blocks",
            )
            timed = [
                t
                for t in timing_rows
                if t["algorithm"] == name
                and t["scenario"] == scenario
                and t["esn0_db"] == db
                and t.get("batch_size") == 1
                and t.get("timing_mode") == settings["timing_mode"]
                and t["status"] == "complete"
            ]
            row.update(
                {
                    k: timed[0][k] if timed else None
                    for k in (
                        "mean_latency_ms",
                        "median_latency_ms",
                        "p95_latency_ms",
                        "batch_size",
                        "timing_mode",
                    )
                }
            )
            aggregate_rows.append(row)
    csv_write(pending / "per_frame_metrics.csv", per_frame)
    csv_write(pending / "aggregate_metrics.csv", aggregate_rows)
    csv_write(pending / "timing.csv", timing_rows)
    csv_write(
        pending / "paired_comparisons.csv", paired_rows, ("algorithm_a", "algorithm_b", "metric")
    )
    json_write(pending / "examples.json", examples)
    metadata = {
        "run_id": run_id,
        "kind": "evaluation",
        "status": "incomplete_or_failed"
        if any(r["hard_failures"] for r in aggregate_rows)
        else "complete",
        "input_lineage_hash": stable_hash(lineage_digest),
        "bootstrap_seed": seed,
        "planned_samples": len(data),
        "algorithms": list(models),
        "manifest_hash": manifest_hash,
        "compatibility": {
            "physics": physical_contract(config),
            "evaluation": settings,
            "runtime": c["runtime"],
            "test_seed": c["seed"],
            "pg_vamp": c["pg_vamp"],
            "vamp": c["vamp"],
        },
    }
    publish(pending, output, metadata)
    return {
        "output": str(output),
        "status": metadata["status"],
        "samples": len(data),
        "run_id": run_id,
    }


def benchmark(
    config: Config,
    manifest: Path,
    output: Path,
    *,
    checkpoint: Path | None = None,
    algorithms: list[str] | None = None,
    allow_untrained: bool = False,
    modes: tuple[str, ...] = MODES,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Standalone bounded timing over one predeclared first block per cell."""
    from ..cli import _provenance

    data, runtime, models, metadata = setup(
        config,
        manifest,
        checkpoint,
        algorithms or config.values["evaluation"]["algorithms"],
        allow_untrained,
    )
    pending = begin(output)
    config.save(pending / "resolved_config.yaml")
    json_write(pending / "environment.json", {**_provenance(runtime), "argv": argv or sys.argv})
    json_write(pending / "checkpoint_metadata.json", metadata)
    rows, seen = [], set()
    with torch.inference_mode():
        for i, record in enumerate(data.records):
            cell = (record["scenario"], float(record["esn0_db"][0]))
            if cell in seen:
                continue
            seen.add(cell)
            synchronize(runtime.device)
            start = perf_counter()
            sample = data[i * 8]
            shared_inputs(sample, runtime)
            synchronize(runtime.device)
            rows.append(
                {
                    "algorithm": "shared_preprocessing",
                    "scenario": cell[0],
                    "esn0_db": cell[1],
                    "sample_id": sample["sample_id"],
                    "timing_mode": "shared_replay_preprocessing",
                    "batch_size": 1,
                    "mean_latency_ms": (perf_counter() - start) * 1000,
                    "backend": data.config.values["data"]["backend"],
                    "includes": "compact_replay,H_build_or_cache,noise_FFT,"
                    "pilot_elimination,shared_conversion",
                    "sync": "oracle_timing; synchronization search not executed",
                    "device": str(runtime.device),
                    "dtype": str(runtime.complex_dtype),
                    "cpu_threads": runtime.cpu_threads,
                    "status": "complete",
                }
            )
            rows.extend(
                _benchmark_sample(
                    data,
                    models,
                    sample,
                    record,
                    runtime,
                    config.values["evaluation"],
                    modes,
                    derive_seed(config.values["seed"], "benchmark"),
                )
            )
    csv_write(pending / "timing.csv", rows)
    status = "complete" if all(r["status"] == "complete" for r in rows) else "incomplete_or_failed"
    publish(
        pending,
        output,
        {"kind": "benchmark", "status": status, "manifest_hash": file_hash(manifest)},
    )
    return {"output": str(output), "status": status, "rows": len(rows)}
