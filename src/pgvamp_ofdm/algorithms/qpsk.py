"""Analytical unit-energy QPSK posterior, independent of reference operators."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class QPSKPosterior:
    mean: torch.Tensor
    probabilities: torch.Tensor
    variance: torch.Tensor
    vbar: torch.Tensor
    alpha: torch.Tensor


def qpsk_posterior(r: torch.Tensor, gamma: torch.Tensor) -> QPSKPosterior:
    """Finite complex r[B,N], paired real gamma[B]>=0 on one device (§14.6)."""
    if r.ndim != 2 or min(r.shape) < 1 or r.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("r must be nonempty complex [B,N]")
    if gamma.shape != r.shape[:1] or gamma.dtype != r.real.dtype or gamma.device != r.device:
        raise ValueError("gamma must be paired real [B] on r's device")
    if not bool(torch.isfinite(r).all() & torch.isfinite(gamma).all() & (gamma >= 0).all()):
        raise ValueError("posterior inputs must be finite, gamma nonnegative")
    u_real = math.sqrt(2) * gamma[:, None] * r.real
    u_imag = math.sqrt(2) * gamma[:, None] * r.imag
    if not bool(torch.isfinite(u_real).all() & torch.isfinite(u_imag).all()):
        raise FloatingPointError("QPSK scaled input overflow")
    mean = torch.complex(u_real.tanh(), u_imag.tanh()) / math.sqrt(2)
    exp_real, exp_imag = (-2 * u_real.abs()).exp(), (-2 * u_imag.abs()).exp()
    variance = 2 * (exp_real / (1 + exp_real).square() + exp_imag / (1 + exp_imag).square())
    vbar = variance.mean(-1)
    # Independent real/imaginary bits, in class=2*b_real+b_imag order.
    real_log = torch.stack((F.logsigmoid(2 * u_real), F.logsigmoid(-2 * u_real)), -1)
    imag_log = torch.stack((F.logsigmoid(2 * u_imag), F.logsigmoid(-2 * u_imag)), -1)
    probabilities = (real_log[..., :, None] + imag_log[..., None, :]).flatten(-2).exp()
    return QPSKPosterior(mean, probabilities, variance, vbar, gamma * vbar)
