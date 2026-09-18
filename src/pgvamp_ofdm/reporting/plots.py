"""Headless plots; zero-rate display markers never modify source tables."""

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .validation import number  # noqa: E402


def _finish(fig: Any, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _missing(ax: Any, message: str = "Not executed / no persisted evidence") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def series(runs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for index, run in enumerate(runs):
        for row in run["aggregate"]:
            if index and not row["algorithm"].startswith("PG"):
                continue
            label = row["algorithm"]
            if row["algorithm"].startswith("PG"):
                label += " seed=" + str(run["checkpoint"].get("train_seed"))
            result.append((label, row))
    return result


def _curves(runs: list[dict[str, Any]], destination: Path) -> list[str]:
    names = []
    entries = series(runs)
    for scenario in sorted({row["scenario"] for _, row in entries}):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario):
            raise ValueError("unsafe scenario figure name")
        for metric in ("ber", "ser", "nmse"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            key = "nmse_linear" if metric == "nmse" else metric
            plotted = False
            for label in sorted({label for label, _ in entries}):
                rows = sorted(
                    (r for name, r in entries if name == label and r["scenario"] == scenario),
                    key=lambda r: float(r["esn0_db"]),
                )
                complete = [r for r in rows if r["status"] == "complete"]
                if not complete:
                    continue
                x = [float(r["esn0_db"]) for r in rows]
                y = [number(r, key) if r["status"] == "complete" else None for r in rows]
                positive = [v for v in y if v is not None and v > 0]
                floor = min(positive) * 0.3 if positive else 1e-6
                # NaN gaps preserve missing/failed SNR cells instead of connecting across them.
                display = [float("nan") if v is None else (floor if v == 0 else v) for v in y]
                (line,) = ax.plot(x, display, ".-", label=label)
                for db, value, shown, row in zip(x, y, display, rows, strict=True):
                    if value == 0:
                        ax.scatter([db], [shown], marker="v", color=line.get_color(), zorder=4)
                        denominator = row["n_symbols"] if metric == "ser" else row["n_bits"]
                        text = "zero energy error" if metric == "nmse" else f"0/{denominator}"
                        ax.annotate(
                            text, (db, shown), xytext=(3, 5), textcoords="offset points", fontsize=7
                        )
                    elif value is not None and metric in ("ber", "ser"):
                        low, high = (
                            number(row, metric + "_ci_low"),
                            number(row, metric + "_ci_high"),
                        )
                        if low is not None and high is not None and high > 0:
                            ax.vlines(db, max(low, floor), high, color=line.get_color(), alpha=0.5)
                plotted = True
            if plotted:
                ax.set_yscale("log")
                ax.legend(fontsize=7)
            else:
                _missing(ax, "Incomplete or failed: full-population metric unavailable")
            frames = sorted({int(r["n_frames"]) for _, r in entries if r["scenario"] == scenario})
            labels = sorted({r["config"]["evaluation"].get("label", "unlabeled") for r in runs})
            evidence = f"{', '.join(labels)}; F={','.join(map(str, frames))} per SNR"
            if min(frames) < 20:
                evidence += "; few frames: unstable inference"
            ax.set(
                xlabel="Es/N0 (dB)", ylabel=key, title=f"{scenario}: {metric.upper()}\n{evidence}"
            )
            ax.grid(alpha=0.25)
            fig.text(
                0.01,
                0.005,
                "Triangles: measured zero; vertical placement is display only.\n"
                + (
                    "Frame-cluster 95% intervals."
                    if metric != "nmse"
                    else "NMSE: summed error energy / summed target energy; no CI computed."
                ),
                fontsize=7,
            )
            filename = f"{metric}_{scenario}.png"
            _finish(fig, destination / filename)
            names.append(filename)
    return names


def _diagnostic_plots(runs: list[dict[str, Any]], destination: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    settings: set[tuple[str, str, str, str, str]] = set()
    for index, mode in enumerate(("per_observation_cold_H", "same_H_amortized")):
        rows = [
            (f"{r['algorithm']}\n{r['scenario']} {r['esn0_db']} dB\nrun {i + 1}", r)
            for i, run in enumerate(runs)
            for r in run["timing"]
            if r.get("timing_mode") == mode
            and r.get("batch_size") == "1"
            and r["status"] == "complete"
        ]
        ax = axes[index]
        if rows:
            settings.update(
                (r["device"], r["dtype"], r["cpu_threads"], r["warmup"], r["repeats"])
                for _, r in rows
            )
            x = np.arange(len(rows))
            ax.bar(x, [float(r["mean_latency_ms"]) for _, r in rows], label="mean")
            ax.scatter(
                x,
                [float(r["p95_latency_ms"]) for _, r in rows],
                marker="x",
                color="black",
                label="p95",
            )
            ax.set_xticks(x, [label for label, _ in rows], rotation=80, fontsize=6)
            ax.legend()
        else:
            _missing(ax)
        ax.set(title=mode, ylabel="Online B=1 latency (ms)")
    caption = " | ".join(
        f"{device}, {dtype}, threads={threads}, warmup={warmup}, repeats={repeats}"
        for device, dtype, threads, warmup, repeats in sorted(settings)
    )
    fig.text(0.02, 0.015, caption or "Timing not executed", fontsize=8)
    _finish(fig, destination / "latency.png")
    entries = series(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(entries))
    labels = [f"{label}\n{r['scenario']} {r['esn0_db']} dB" for label, r in entries]
    axes[0].bar(x, [int(r["hard_failures"]) for _, r in entries])
    axes[0].set_ylim(0, max(1, max(int(r["hard_failures"]) for _, r in entries) * 1.2))
    for position, (_, row) in zip(x, entries, strict=True):
        axes[0].annotate(
            f"{row['hard_failures']}/{row['n_blocks']}",
            (position, int(row["hard_failures"])),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axes[0].set(title="Hard failures (planned blocks retained)", ylabel="Blocks")
    for key in ("message_reject_rate", "precision_cap_rate", "no_information_rate"):
        axes[1].plot(
            x,
            [number(r, key) if number(r, key) is not None else np.nan for _, r in entries],
            ".-",
            label=key,
        )
    axes[1].set(title="Measured diagnostic opportunities", ylabel="Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=80, fontsize=6)
    _finish(fig, destination / "stability.png")
    for key, filename in (("rho", "learned_thresholds.png"), ("mu", "learned_mu.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for run in runs:
            metadata = run["checkpoint"]
            initial, learned = metadata.get("initial_" + key), metadata.get("learned_" + key)
            if initial is not None and learned is not None:
                layer = np.arange(1, len(learned) + 1)
                seed = metadata.get("train_seed")
                ax.plot(layer, initial, "--", label=f"initial seed={seed}")
                ax.plot(layer, learned, ".-", label=f"saved seed={seed}")
                plotted = True
        if plotted:
            ax.legend()
        else:
            _missing(ax)
        ax.set(
            xlabel="Layer",
            ylabel="rho (dB)" if key == "rho" else key,
            title="Persisted initial and evaluated parameters",
        )
        _finish(fig, destination / filename)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for run in runs:
        layers = [
            d["layer_summaries"]
            for d in run["diagnostics"]
            if d.get("layer_summaries") and d["status"] == "complete"
        ]
        if not layers:
            continue
        for key, ax in (
            ("effective_candidate_ratio", axes[0]),
            ("effective_all_ratio", axes[0]),
            ("safety_relative", axes[1]),
            ("c", axes[2]),
        ):
            values = np.asarray(
                [[np.asarray(layer[key]).mean() for layer in block] for block in layers],
                dtype=float,
            )
            means = values.mean(axis=0)
            ax.plot(
                np.arange(1, len(means) + 1),
                means,
                ".-",
                label=f"{key}, seed={run['checkpoint'].get('train_seed')}",
            )
    for ax, title in zip(
        axes,
        ("Effective edges: both denominators", "Safety / baseline norm", "Linear information c"),
        strict=True,
    ):
        if ax.lines:
            ax.legend(fontsize=6)
        else:
            _missing(ax)
        ax.set(title=title, xlabel="Layer; mean over successful blocks")
    _finish(fig, destination / "effective_edges.png")
    return [
        "latency.png",
        "stability.png",
        "learned_thresholds.png",
        "learned_mu.png",
        "effective_edges.png",
    ]


def make_plots(runs: list[dict[str, Any]], destination: Path) -> list[str]:
    """Generate required figures, annotating missing evidence without invented data."""
    destination.mkdir()
    names = _curves(runs, destination) + _diagnostic_plots(runs, destination)
    run = runs[0]
    example = None
    if "example.pt" in run["bundle"]["files"]:
        example = torch.load(run["root"] / "example.pt", map_location="cpu", weights_only=True)
        samples = {r["sample_id"]: r["input_hash"] for r in run["lineage"]}
        if example.get("sample_id") not in samples or (
            example.get("input_hash") != samples[example["sample_id"]]
        ):
            raise ValueError("example identity absent from shared lineage")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(
        axes, ("h_grid", "H"), ("Full grid H_g", "Effective data H"), strict=True
    ):
        if example is not None and key in example:
            h = example[key]
            energy = h.abs().square()
            off = energy.clone()
            off.diagonal().zero_()
            ratio = float(off.sum() / energy.sum().clamp_min(torch.finfo(energy.dtype).eps))
            title += f"\nICI energy ratio={ratio:.6g}"
            image = ax.imshow(h.abs().numpy(), origin="lower", aspect="auto")
            fig.colorbar(image, ax=ax, label="Magnitude")
        else:
            _missing(ax)
        ax.set(title=title, xlabel="Input subcarrier", ylabel="Output subcarrier")
    _finish(fig, destination / "channel_ici_example.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7))
    if example is not None:
        rate = example["sample_rate_hz"]
        for ax, key in zip(axes[:2], ("transmit_real", "received_iq"), strict=True):
            value = example[key]
            ax.plot(np.arange(value.numel()) / rate, value.real.numpy(), linewidth=0.5)
            ax.set(title=key, xlabel="Time (s)")
        value = example["sync_scores"]
        axes[2].plot(np.arange(value.numel()), value.numpy())
        axes[2].axvline(example["sync_peak"], linestyle="--", label="measured peak")
        axes[2].axvline(
            example.get(
                "expected_template_start",
                example["arrival_offset_samples"]
                + run["config"]["frame"]["leading_silence_samples"],
            ),
            color="black",
            linestyle=":",
            label="expected LFM start (arrival + leading silence)",
        )
        if "sync_threshold" in example:
            axes[2].axhline(
                example["sync_threshold"], linestyle=":", color="gray", label="detection threshold"
            )
        axes[2].legend()
        axes[2].set(
            title=f"Diagnostic correlation; detected={example['sync_detected']}. "
            "Evaluation uses oracle timing.",
            xlabel="Sample index",
        )
    else:
        for ax in axes:
            _missing(ax)
    _finish(fig, destination / "waveform_and_sync_check.png")
    return [*names, "channel_ici_example.png", "waveform_and_sync_check.png"]
