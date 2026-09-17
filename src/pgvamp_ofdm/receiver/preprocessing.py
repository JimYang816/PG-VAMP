"""Select data receiver rows and eliminate the actual known pilot leakage."""

import math
from dataclasses import dataclass

import torch

from ..modulation.allocation import Allocation, _validate_symbols


@dataclass(frozen=True)
class DetectionInput:
    """Square complex system [...,400,400], [...,400], known complex variance."""

    H: torch.Tensor
    y: torch.Tensor
    sigma2: float
    window_offset_s: float


def preprocess(
    h_grid: torch.Tensor,
    y_grid: torch.Tensor,
    pilots: torch.Tensor,
    allocation: Allocation,
    sigma2: float,
    *,
    csi_mode: str = "perfect",
    window_offset_s: float = 0.0,
) -> DetectionInput:
    """Require identical batch axes, dtype/device; no implicit matrix broadcasting."""
    allocation.validate()
    if csi_mode != "perfect":
        raise ValueError("only perfect CSI is supported; estimated CSI is not implemented")
    _validate_symbols(pilots, "pilots")
    if (
        not isinstance(h_grid, torch.Tensor)
        or not isinstance(y_grid, torch.Tensor)
        or h_grid.ndim < 2
        or h_grid.shape[-2:] != (512, 512)
        or y_grid.shape != (*h_grid.shape[:-2], 512)
        or pilots.shape != (*h_grid.shape[:-2], 64)
    ):
        raise ValueError("H/Y/pilots require matching [...,512,512]/[...,512]/[...,64] shapes")
    for value in (h_grid, y_grid):
        if (
            value.dtype != pilots.dtype
            or value.device != pilots.device
            or value.device != allocation.q.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("H/Y/pilots must be finite and share complex dtype/device")
    if not math.isfinite(sigma2) or sigma2 <= 0 or not math.isfinite(window_offset_s):
        raise ValueError("positive finite sigma2 and finite window offset required")
    rows = h_grid[..., allocation.data_grid_index, :]
    h_dd, h_dp = rows[..., allocation.data_grid_index], rows[..., allocation.pilot_grid_index]
    y = y_grid[..., allocation.data_grid_index] - (h_dp @ pilots.unsqueeze(-1)).squeeze(-1)
    return DetectionInput(h_dd, y, sigma2, window_offset_s)
