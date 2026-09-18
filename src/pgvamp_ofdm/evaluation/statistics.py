"""Shared frame-cluster resampling; no independent-bit approximation."""

from typing import Any

import numpy as np


def bootstrap(
    rows: dict[str, list[dict[str, Any]]], *, seed: int, repeats: int = 2000
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 2 or not rows:
        raise ValueError("bootstrap requires nonnegative seed, repeats >= 2, and frames")
    ordered = {a: sorted(v, key=lambda x: x["frame_id"]) for a, v in rows.items()}
    first = next(iter(ordered.values()))
    ids = [r["frame_id"] for r in first]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("bootstrap requires unique independent frames")
    if any([r["frame_id"] for r in v] != ids for v in ordered.values()):
        raise ValueError("paired frame identities disagree")
    rng = np.random.default_rng(seed)
    # O(F) index storage, regardless of bootstrap repeat count.
    distributions: dict[str, dict[str, list[float]]] = {
        a: {m: [] for m in ("ber", "ser")}
        for a, v in ordered.items()
        if all(r["status"] == "complete" for r in v)
    }
    arrays = {
        a: {
            key: np.asarray([r[key] for r in ordered[a]], dtype=np.float64)
            for key in ("bit_errors", "symbol_errors", "n_bits", "n_symbols")
        }
        for a in distributions
    }
    for _ in range(repeats):
        indices = rng.integers(0, len(ids), size=len(ids))
        for a, values in arrays.items():
            for metric, errors, total in (
                ("ber", "bit_errors", "n_bits"),
                ("ser", "symbol_errors", "n_symbols"),
            ):
                distributions[a][metric].append(
                    float(values[errors][indices].sum() / values[total][indices].sum())
                )
    intervals: dict[str, dict[str, Any]] = {}
    for a in ordered:
        intervals[a] = {
            "bootstrap_seed": seed,
            "bootstrap_repeats": repeats,
            "ci_unstable_few_frames": len(ids) < 20,
            "ci_unit": "independent_frame",
            "zero_ci_does_not_prove_zero": True,
        }
        for metric in ("ber", "ser"):
            bounds = (
                np.quantile(distributions[a][metric], [0.025, 0.975])
                if a in distributions
                else (None, None)
            )
            intervals[a].update({f"{metric}_ci_low": bounds[0], f"{metric}_ci_high": bounds[1]})
    paired = []
    names = list(distributions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for metric in ("ber", "ser"):
                difference = np.asarray(distributions[a][metric]) - distributions[b][metric]
                low, high = np.quantile(difference, [0.025, 0.975])
                key, denominator = (
                    ("bit_errors", "n_bits") if metric == "ber" else ("symbol_errors", "n_symbols")
                )
                observed = (
                    arrays[a][key].sum() / arrays[a][denominator].sum()
                    - arrays[b][key].sum() / arrays[b][denominator].sum()
                )
                paired.append(
                    {
                        "algorithm_a": a,
                        "algorithm_b": b,
                        "metric": metric,
                        "difference_a_minus_b": float(observed),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "n_frames": len(ids),
                        "bootstrap_seed": seed,
                        "bootstrap_repeats": repeats,
                        "ci_unstable_few_frames": len(ids) < 20,
                    }
                )
    return intervals, paired
