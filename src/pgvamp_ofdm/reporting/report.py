"""Read-only reporting: no training, detection, parameter selection or replay."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .plots import make_plots
from .validation import cell, number, read_run, validate_merge


def _display(value: Any) -> str:
    if value is None or value == "":
        return "未执行 / unavailable"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "未执行：没有持久记录。\n"
    return (
        "| "
        + " | ".join(fields)
        + " |\n| "
        + " | ".join("---" for _ in fields)
        + " |\n"
        + "\n".join("| " + " | ".join(_display(row.get(k)) for k in fields) + " |" for row in rows)
        + "\n"
    )


def _seed_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(runs) < 2:
        return rows
    for first in runs[0]["aggregate"]:
        if not first["algorithm"].startswith("PG"):
            continue
        group = [r for run in runs for r in run["aggregate"] if cell(r) == cell(first)]
        for metric in ("ber", "ser", "nmse_linear", "fer"):
            values = [number(r, metric) for r in group]
            available = [v for v in values if v is not None]
            complete = len(available) == len(values)
            rows.append(
                {
                    "algorithm": first["algorithm"],
                    "scenario": first["scenario"],
                    "esn0_db": first["esn0_db"],
                    "metric": metric,
                    "train_seeds": len(runs),
                    "mean": float(np.mean(available)) if complete else None,
                    "sample_std": float(np.std(available, ddof=1)) if complete else None,
                    "status": "complete" if complete else "incomplete_or_failed",
                }
            )
    return rows


def _markdown(runs: list[dict[str, Any]], figures: list[str], summary: list[dict[str, Any]]) -> str:
    first = runs[0]
    config = first["config"]
    text = [
        "# Paired evaluation report\n",
        "本报告仅读取已保存且通过文件哈希、输入 lineage 与计数复算检查的评测产物。",
        "## Signal and evaluation contract\n",
        "512 grid / 400 data / 64 pilots / 47 guards / 1 DC null; uncoded "
        "QPSK, eight OFDM blocks per frame. "
        "Ideal complex I/Q, oracle timing, perfect CSI and known noise "
        "variance. Full-H MMSE (linear), "
        "fixed-iteration SVD VAMP, and checkpoint PG-VAMP use the same samples "
        "and preprocessing. "
        "This is a simulated affine time-scaling channel; real-hardware, "
        "synchronization-error and coded-link performance are 未执行.",
        "```json\n"
        + json.dumps(
            {k: config[k] for k in ("waveform", "frame", "receiver") if k in config}, indent=2
        )
        + "\n```",
        "## Evidence and environment\n",
    ]
    for run in runs:
        bundle, checkpoint = run["bundle"], run["checkpoint"]
        environment = {
            key: value
            for key, value in run["environment"].items()
            if key not in ("package_source_hashes", "spec_hashes")
        }
        concise_checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if not key.startswith(("initial_", "learned_"))
        }
        text += [
            f"Run `{bundle['run_id']}` — status `{bundle['status']}`; source `{run['root']}`.",
            f"Planned blocks: {bundle['planned_samples']}; "
            f"label: {run['config']['evaluation'].get('label')}; "
            f"bootstrap seed: {bundle['bootstrap_seed']}; "
            f"test manifest hash: `{bundle['manifest_hash']}`.",
            "Checkpoint metadata (actual steps, seed and hash; parameters plotted below):",
            "```json\n" + json.dumps(concise_checkpoint, indent=2, ensure_ascii=False) + "\n```",
            "Actual environment and invocation:",
            "```json\n" + json.dumps(environment, indent=2, ensure_ascii=False) + "\n```",
            "Full code/spec hashes: "
            f"[environment.json]({run['root'].as_posix()}/environment.json).",
        ]
    seeds = {
        r["checkpoint"].get("train_seed") for r in runs if r["checkpoint"]["status"] == "trained"
    }
    text += [
        f"Distinct trained seeds: {len(seeds)}.",
        "A single seed does not establish across-training-seed uncertainty. "
        "Repeated baseline predictions are shown once for performance and are "
        "not independent training seeds. "
        "Timing repetitions remain separately labeled run measurements.",
        "## Counts, performance and uncertainty\n",
        "Rates use summed counts/energy, with no target-dependent alignment. "
        "Missing full-population metrics remain unavailable. "
        "Intervals resample independent frames as clusters; within-frame bits "
        "and cross-SNR copies are not independent trials. "
        "Fewer than 20 frames: 区间不稳定 (unstable interval). Zero errors remain 0 "
        "/ measured denominator. "
        "An all-zero bootstrap interval does not prove zero true BER; FER "
        "upper bounds concern independent frame events only.",
    ]
    rows = [
        r
        for i, run in enumerate(runs)
        for r in run["aggregate"]
        if not i or r["algorithm"].startswith("PG")
    ]
    text.append(
        _table(
            rows,
            (
                "run_id",
                "algorithm",
                "scenario",
                "esn0_db",
                "train_seed",
                "n_frames",
                "n_blocks",
                "bit_errors",
                "n_bits",
                "symbol_errors",
                "n_symbols",
                "ber",
                "ber_ci_low",
                "ber_ci_high",
                "ser",
                "ser_ci_low",
                "ser_ci_high",
                "nmse_linear",
                "nmse_db",
                "evm_pct",
                "bler",
                "fer",
                "fer_zero_upper95",
                "goodput_bps",
                "status",
            ),
        )
    )
    text += [
        "### Paired differences (A minus B)\n",
        "Each interval uses the same resampled frame indices for both "
        "algorithms. No superiority claim is made from a small smoke "
        "population.",
    ]
    for run in runs:
        text += [
            f"Run `{run['bundle']['run_id']}`:",
            _table(
                run["paired"],
                (
                    "scenario",
                    "esn0_db",
                    "algorithm_a",
                    "algorithm_b",
                    "metric",
                    "difference_a_minus_b",
                    "ci_low",
                    "ci_high",
                    "n_frames",
                    "ci_unstable_few_frames",
                ),
            ),
        ]
    text += [
        "### Across training seeds\n",
        _table(
            summary,
            (
                "algorithm",
                "scenario",
                "esn0_db",
                "metric",
                "train_seeds",
                "mean",
                "sample_std",
                "status",
            ),
        ),
        "Seed mean/std above, when available, condition on the same fixed test "
        "set; they are not extra independent test frames.",
        "## Timing and memory\n",
        "Cold-H and same-H amortized timing are separate scopes. B=1 is online "
        "block latency; batch latency and "
        "amortized throughput are distinct. Preparation is listed separately. "
        "Shared replay preprocessing is outside detector timing. "
        "CPU RSS is process-lifetime high-water memory and is not attributable "
        "to a single algorithm. Matrix bytes are estimates. "
        "Compute bits/s is not physical-link goodput.",
    ]
    for run in runs:
        text += [
            f"Run `{run['bundle']['run_id']}`:",
            _table(
                run["timing"],
                (
                    "algorithm",
                    "scenario",
                    "esn0_db",
                    "timing_mode",
                    "batch_size",
                    "device",
                    "dtype",
                    "cpu_threads",
                    "warmup",
                    "repeats",
                    "mean_latency_ms",
                    "median_latency_ms",
                    "p95_latency_ms",
                    "prepare_ms",
                    "total_amortized_per_block_ms",
                    "blocks_per_second",
                    "data_bits_per_second",
                    "cpu_process_peak_bytes",
                    "cuda_peak_allocated_bytes",
                    "matrix_work_bytes_estimate",
                    "trainable_parameters",
                    "includes",
                    "excludes",
                    "status",
                ),
            ),
        ]
    text += [
        "## Stability, failure ledger and model diagnostics\n",
        _table(
            rows,
            (
                "algorithm",
                "scenario",
                "esn0_db",
                "hard_failures",
                "successful_blocks",
                "message_rejected",
                "message_opportunities",
                "message_reject_rate",
                "precision_capped",
                "precision_cap_rate",
                "no_information",
                "no_information_opportunities",
                "no_information_rate",
                "diagnostics_scope",
                "conditional_success_bits",
                "conditional_success_ber",
            ),
        ),
        "Legal message rejection retains the previous valid message. Hard "
        "failures preserve planned denominators; "
        "conditional-success BER is not full-population BER. Diagnostics with "
        "failures cover successful blocks only.",
    ]
    for run in runs:
        checkpoint = run["checkpoint"]
        for key in ("rho", "mu"):
            initial, learned = checkpoint.get("initial_" + key), checkpoint.get("learned_" + key)
            if initial is not None and learned is not None:
                delta = float(np.max(np.abs(np.asarray(initial) - np.asarray(learned))))
                text.append(
                    f"Run `{run['bundle']['run_id']}`: "
                    f"max |saved {key} - initial {key}| = {delta:.12g}; "
                    + ("parameters changed." if delta > 0 else "no observed parameter change.")
                )
        layers = [layer for d in run["diagnostics"] for layer in d.get("layer_summaries", [])]
        for key in ("effective_candidate_ratio", "effective_all_ratio", "safety_relative", "c"):
            stats = run["layer_ranges"].get(key)
            if stats:
                text.append(
                    f"{key}: min={stats['min']:.8g}, mean={stats['mean']:.8g}, "
                    f"max={stats['max']:.8g} "
                    "over saved successful block/layer diagnostics."
                )
        if layers:
            dense = run["layer_ranges"]["effective_all_ratio"]["min"] >= 0.95
            text.append(
                "At least 95% of all off-diagonal positions are effective in every "
                "recorded layer: " + str(dense) + "."
            )
            text.append(
                "Safety relative size is reported against its baseline norm. Whether "
                "it is excessively conservative "
                "requires a controlled ablation; that ablation is 未执行. Zero-baseline "
                "flags remain in diagnostics.jsonl."
            )
        text += [
            _table(run["failures"], ("algorithm", "sample_id", "stage", "exception", "reason"))
            if run["failures"]
            else "Hard failures in this run: 0."
        ]
        training_scenarios = checkpoint.get("training_scenarios", {})
        ood = sorted(
            set(config["evaluation"]["scenarios"])
            - {k for k, v in training_scenarios.items() if v > 0}
        )
        text.append(
            "Scenarios absent from recorded training support: "
            + (
                ", ".join(ood)
                if training_scenarios and ood
                else "none identified / training support unavailable"
            )
            + ". Distribution-shift degradation cannot be isolated from these counts "
            "without a matched experiment; 未执行."
        )
    text += [
        "## Limits and unexecuted work\n",
        "Only the listed populations, checkpoints and timing measurements are "
        "supported. Main training, a sufficiently powered "
        "full SNR sweep, CUDA hardware validation and real-world performance "
        "are not established by a development/smoke report; "
        "evidence outside this bundle is 未执行. PG-VAMP may show no BER gain or "
        "may be slower. "
        "No target-BER SNR gain is interpolated or extrapolated here; "
        "sufficient nonzero error counts and coverage would be required.",
        "## Figures\n",
    ]
    text += [f"![{name}](figures/{name})" for name in figures]
    return "\n\n".join(text) + "\n"


def report(results: list[Path], output: Path | None = None) -> dict[str, Any]:
    """Read validated evaluation directories and create only report artifacts.

    Single-run output defaults to that bundle. Multi-run reports require a fresh
    explicit output directory. Existing non-report files are never overwritten.
    """
    if not results:
        raise ValueError("at least one result directory is required")
    roots = [Path(root).resolve() for root in results]
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate result directory")
    if len(roots) > 1 and output is None:
        raise ValueError("multi-run report requires explicit output")
    runs = [read_run(root) for root in roots]
    validate_merge(runs)
    destination = Path(output).resolve() if output is not None else roots[0]
    if destination in roots and (len(roots) > 1 or destination != roots[0]):
        raise ValueError("merged report cannot overwrite a source bundle")
    in_place = len(roots) == 1 and destination == roots[0]
    if not in_place and destination.exists():
        raise ValueError("report output exists; use a fresh directory")
    if in_place:
        immutable = runs[0]["bundle"]["files"]
        if any(name.startswith("figures/") for name in immutable):
            raise ValueError("report figures overlap immutable evidence")
        if (destination / "REPORT.md").exists() and not (
            destination / "report_metadata.json"
        ).is_file():
            raise ValueError("existing report lacks ownership metadata; use a fresh output")
        for filename in ("REPORT.md", "figures", "report_metadata.json"):
            path = destination / filename
            if path.is_symlink() or filename in runs[0]["bundle"]["files"]:
                raise ValueError("report destination overlaps immutable evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = _seed_summary(runs)
    with tempfile.TemporaryDirectory(prefix=".report-", dir=destination.parent) as temporary:
        pending = Path(temporary)
        figures = make_plots(runs, pending / "figures")
        (pending / "REPORT.md").write_text(_markdown(runs, figures, summary), encoding="utf-8")
        metadata = {
            "source_runs": [r["bundle"]["run_id"] for r in runs],
            "figures": figures,
            "seed_summary": summary,
            "status": "incomplete_or_failed" if any(r["failures"] for r in runs) else "complete",
        }
        (pending / "report_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        destination.mkdir(exist_ok=in_place)
        figure_dir = destination / "figures"
        figure_dir.mkdir(exist_ok=True)
        for path in (pending / "figures").iterdir():
            if (figure_dir / path.name).is_symlink():
                raise ValueError("unsafe figure destination")
            os.replace(path, figure_dir / path.name)
        os.replace(pending / "report_metadata.json", destination / "report_metadata.json")
        os.replace(pending / "REPORT.md", destination / "REPORT.md")
    return {"output": str(destination), **metadata}
