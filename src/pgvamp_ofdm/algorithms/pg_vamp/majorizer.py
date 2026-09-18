"""Energy compensation and nonnegative O(N²) safety terms (§14.3)."""

import torch


def _exclusive_sum(values: torch.Tensor) -> torch.Tensor:
    # Prefix/suffix sums retain small neighbors next to a large diagonal.
    zero = torch.zeros_like(values[..., :1])
    prefix = torch.cat((zero, values[..., :-1].cumsum(-1)), -1)
    suffix = torch.cat((values[..., 1:].flip(-1).cumsum(-1).flip(-1), zero), -1)
    return prefix + suffix


def majorizer(H: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    magnitude = H.abs()
    retained = mask * magnitude
    removed = (1 - mask) * magnitude
    d = ((1 - mask.square()) * magnitude.square()).sum(-2)
    ell = (removed * _exclusive_sum(magnitude) + retained * _exclusive_sum(removed)).sum(-2)
    gated = mask * H
    G = gated.mH @ gated + torch.diag_embed(d)
    return {"d": d, "ell": ell, "G": G, "Gbar": G + torch.diag_embed(ell)}
