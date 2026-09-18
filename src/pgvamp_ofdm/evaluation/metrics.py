"""Integer counts and energy sums; incomplete populations never acquire full rates."""

import math
from typing import Any

import torch

COUNTS = (
    "n_frames",
    "n_blocks",
    "n_bits",
    "n_symbols",
    "bit_errors",
    "symbol_errors",
    "block_errors",
    "frame_errors",
    "successful_blocks",
    "hard_failures",
)
ENERGIES = ("error_energy", "target_energy")


def block_counts(
    bits_hat: torch.Tensor, x_soft: torch.Tensor, bits: torch.Tensor, x: torch.Tensor
) -> dict[str, Any]:
    """Unbatched [N,2] bits / [N] symbols; no post-hoc alignment."""
    if bits.shape != bits_hat.shape or bits.shape != (*x.shape, 2) or x_soft.shape != x.shape:
        raise ValueError("metric shapes disagree")
    if not bool(torch.isfinite(x_soft).all()) or not bool(torch.isfinite(x).all()):
        raise FloatingPointError("nonfinite metric symbols")
    if not bool(((bits_hat == 0) | (bits_hat == 1)).all()):
        raise ValueError("predicted bits must be binary")
    errors = bits != bits_hat
    error_energy = float((x_soft - x).abs().square().sum())
    target_energy = float(x.abs().square().sum())
    if not math.isfinite(error_energy) or not math.isfinite(target_energy):
        raise FloatingPointError("nonfinite metric energy")
    if target_energy <= 0:
        raise ValueError("metric target energy must be positive")
    return {
        "bit_errors": int(errors.sum()),
        "symbol_errors": int(errors.any(-1).sum()),
        "block_errors": int(errors.any()),
        "error_energy": error_energy,
        "target_energy": target_energy,
    }


def frame_counts(blocks: list[dict[str, Any] | None], symbols: int) -> dict[str, Any]:
    if len(blocks) != 8 or symbols < 1:
        raise ValueError("physical frame requires exactly eight planned blocks")
    good = [b for b in blocks if b is not None]
    row: dict[str, Any] = {
        "n_frames": 1,
        "n_blocks": 8,
        "n_bits": 16 * symbols,
        "n_symbols": 8 * symbols,
        "successful_blocks": len(good),
        "hard_failures": 8 - len(good),
        "frame_errors": int(any(b["block_errors"] for b in good)),
    }
    for key in ("bit_errors", "symbol_errors", "block_errors", *ENERGIES):
        row[key] = sum(b[key] for b in good)
    return {**row, **rates(row)}


def rates(row: dict[str, Any]) -> dict[str, Any]:
    complete = row["hard_failures"] == 0
    result: dict[str, Any] = {"status": "complete" if complete else "incomplete_or_failed"}
    for rate, count, total in (
        ("ber", "bit_errors", "n_bits"),
        ("ser", "symbol_errors", "n_symbols"),
        ("bler", "block_errors", "n_blocks"),
        ("fer", "frame_errors", "n_frames"),
    ):
        result[rate] = row[count] / row[total] if complete else None
    nmse = row["error_energy"] / row["target_energy"] if complete else None
    result.update(
        nmse_linear=nmse,
        nmse_db=(10 * math.log10(nmse) if nmse else None),
        nmse_zero=nmse == 0,
        evm_pct=100 * math.sqrt(nmse) if nmse is not None else None,
        goodput_bps=6400 / 0.956 * (1 - result["fer"]) if complete else None,
        fer_zero_upper95=1 - 0.05 ** (1 / row["n_frames"])
        if complete and row["frame_errors"] == 0
        else None,
        zero_bit_errors=complete and row["bit_errors"] == 0,
    )
    successful_bits = row["successful_blocks"] * row["n_bits"] // row["n_blocks"]
    result["conditional_success_bits"] = successful_bits
    result["conditional_success_ber"] = (
        row["bit_errors"] / successful_bits if successful_bits else None
    )
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty frames")
    result = {k: sum(row[k] for row in rows) for k in (*COUNTS, *ENERGIES)}
    return {**result, **rates(result)}
