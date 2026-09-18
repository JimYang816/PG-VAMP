"""Monotone thresholds and directed soft column-energy gates (§14.1–14.2)."""

import torch
from torch.nn import functional as F


def thresholds(
    raw_gaps: torch.Tensor, raw_mu: torch.Tensor, hi: float, lo: float, gap: float
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = F.softplus(raw_gaps)
    t = torch.arange(1, positive.numel() + 1, dtype=positive.dtype, device=positive.device)
    rho = (
        hi
        - t * gap
        - (hi - lo - positive.numel() * gap) * positive.cumsum(0) / (1 + positive.sum())
    )
    return rho, raw_mu.sigmoid()


def soft_mask(H: torch.Tensor, rho: torch.Tensor, temperature: float) -> torch.Tensor:
    energy = H.abs().square()
    column_energy = energy.sum(-2, keepdim=True)
    denominator = torch.where(column_energy == 0, torch.ones_like(column_energy), column_energy)
    scores = 10 * (energy / denominator).clamp(min=1e-12).log10()
    mask = torch.sigmoid((scores - rho) / temperature) * (energy > 0)
    return mask.masked_fill(torch.eye(H.shape[-1], dtype=torch.bool, device=H.device), 1)
