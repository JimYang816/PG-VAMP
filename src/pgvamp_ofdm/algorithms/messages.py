"""Explicit nonlinear extrinsic acceptance and mean-preserving precision cap."""

from dataclasses import dataclass

import torch


@dataclass
class MessageUpdate:
    r: torch.Tensor
    gamma: torch.Tensor
    rejected: torch.Tensor
    capped: torch.Tensor
    underflow: torch.Tensor


def nonlinear_message(
    mean: torch.Tensor,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    vbar: torch.Tensor,
    alpha: torch.Tensor,
    previous_r: torch.Tensor,
    previous_gamma: torch.Tensor,
) -> MessageUpdate:
    """Use §14.6 on complex [B,N] means and paired real [B] scalars.

    Invalid candidates retain the previous valid message. No reciprocal is
    evaluated on rejected or capped entries, including in their backward graph.
    """
    vectors = (mean, r1, previous_r)
    scalars = (gamma1, vbar, alpha, previous_gamma)
    if previous_r.ndim != 2 or min(previous_r.shape) < 1:
        raise ValueError("messages must have nonempty [B,N] shape")
    if previous_r.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("messages must be complex64 or complex128")
    if any(t.shape != previous_r.shape or t.dtype != previous_r.dtype for t in vectors):
        raise ValueError("message vector shapes/dtypes must agree")
    if any(t.shape != previous_r.shape[:1] or t.dtype != previous_r.real.dtype for t in scalars):
        raise ValueError("message scalars must be paired real [B]")
    if any(t.device != previous_r.device for t in (*vectors, *scalars)):
        raise ValueError("messages must share a device")
    if not bool(
        torch.isfinite(previous_r).all()
        & torch.isfinite(previous_gamma).all()
        & (previous_gamma >= 1e-10).all()
        & (previous_gamma <= 1e8).all()
    ):
        raise ValueError("previous message must be valid")
    denominator = 1 - alpha
    underflow = vbar == 0
    eligible = (
        torch.isfinite(vbar)
        & (vbar > 1 / torch.finfo(vbar.dtype).max)
        & torch.isfinite(denominator)
        & (denominator >= 1e-6)
        & torch.isfinite(gamma1)
        & (gamma1 >= 0)
        & torch.isfinite(mean).all(-1)
        & torch.isfinite(r1).all(-1)
    )
    accepted, capped = torch.zeros_like(eligible), torch.zeros_like(eligible)
    r, gamma = previous_r.clone(), previous_gamma.clone()
    indices = eligible.nonzero(as_tuple=True)[0]
    if indices.numel():
        numerator = mean[indices] - alpha[indices, None] * r1[indices]
        # Reject unrepresentable quotients before division. Masking an infinite
        # quotient afterwards leaves 0 * Inf in the denominator's backward.
        limit = torch.finfo(vbar.dtype).max * denominator[indices].clamp(max=1)
        representable = (
            torch.isfinite(numerator).all(-1)
            & (numerator.real.abs() <= limit[:, None]).all(-1)
            & (numerator.imag.abs() <= limit[:, None]).all(-1)
        )
        indices = indices[representable]
        candidate = numerator[representable] / denominator[indices, None]
        # Compare before reciprocal: finite capped forwards must also have finite gradients.
        cap = vbar[indices] < 1 / (1e8 + gamma1[indices])
        precision = torch.full_like(vbar[indices], 1e8)
        regular = (~cap).nonzero(as_tuple=True)[0]
        precision[regular] = 1 / vbar[indices[regular]] - gamma1[indices[regular]]
        valid = torch.isfinite(candidate).all(-1) & torch.isfinite(precision) & (precision >= 1e-10)
        take = indices[valid]
        r[take], gamma[take] = candidate[valid], precision[valid]
        accepted[take], capped[take] = True, cap[valid]
    return MessageUpdate(r, gamma, ~accepted, capped, underflow)
