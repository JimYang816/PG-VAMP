"""Centered full-channel innovation and exact local covariance calibration."""

import torch

from ..base import finite_samples, require_valid
from .majorizer import majorizer


def linear_layer(
    H: torch.Tensor,
    y: torch.Tensor,
    sigma2: torch.Tensor,
    r2: torch.Tensor,
    gamma2: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    *,
    jitter: float = 0,
    layer: int = 0,
) -> dict[str, torch.Tensor]:
    batch, n, _ = H.shape
    identity = torch.eye(n, dtype=H.dtype, device=H.device).expand(batch, n, n)
    terms = majorizer(H, mask)
    gamma_w = sigma2.reciprocal()
    P = gamma_w[:, None, None] * terms["Gbar"] + (gamma2 + jitter)[:, None, None] * identity
    P = (P + P.mH) / 2
    require_valid(finite_samples(P), "PG-VAMP", "nonfinite majorizer", H, layer)
    factor, info = torch.linalg.cholesky_ex(P)
    require_valid((info == 0) & finite_samples(factor), "PG-VAMP", "Cholesky failed", H, layer)

    def apply_A(V: torch.Tensor) -> torch.Tensor:
        return gamma_w[:, None, None] * (H.mH @ (H @ V)) + gamma2[:, None, None] * V

    def apply_B(V: torch.Tensor) -> torch.Tensor:
        q = torch.cholesky_solve(V, factor)
        return q + mu * torch.cholesky_solve(V - apply_A(q), factor)

    residual = y - (H @ r2[..., None]).squeeze(-1)
    innovation = apply_B(gamma_w[:, None, None] * (H.mH @ residual[..., None])).squeeze(-1)
    W = apply_B(gamma_w[:, None, None] * H.mH)
    c = (W @ H).diagonal(dim1=-2, dim2=-1).real.mean(-1)
    finfo = torch.finfo(sigma2.dtype)
    if 4 * n * finfo.eps >= 1:
        raise FloatingPointError(
            f"PG-VAMP trace bound inapplicable; layer={layer}, "
            f"batch index={list(range(batch))}, dtype={H.dtype}, device={H.device}"
        )
    # Conditional rounding bound for the computed W/H contraction, not a
    # factorization error bound. No unit-scale floor discards resolved weak H.
    scale = (W.abs() * H.mT.abs()).sum((-2, -1)) / n
    tolerance = (4 * n * finfo.eps / (1 - 4 * n * finfo.eps) * scale).clamp(min=finfo.tiny)
    require_valid(
        finite_samples(W)
        & finite_samples(innovation)
        & torch.isfinite(c)
        & torch.isfinite(tolerance)
        & (c >= -tolerance),
        "PG-VAMP",
        "invalid linear operator/trace",
        H,
        layer,
    )
    informed = c > tolerance
    indices = informed.nonzero(as_tuple=True)[0]
    # A zero-valued graph dependency makes wholly uninformative backward valid.
    zero = 0 * mu + 0 * mask.sum()
    r1, gamma1 = torch.zeros_like(r2) + zero, torch.zeros_like(gamma2) + zero
    K = torch.zeros_like(W)
    if indices.numel():
        # Two square-root divisions avoid the unrepresentable x/c**2 local
        # derivative of a single quotient on resolved, extremely weak inputs.
        root_c = c[indices].sqrt()
        K[indices] = torch.view_as_complex(
            torch.view_as_real(W[indices])
            / root_c[:, None, None, None]
            / root_c[:, None, None, None]
        )
        r1[indices] = r2[indices] + torch.view_as_complex(
            torch.view_as_real(innovation[indices]) / root_c[:, None, None] / root_c[:, None, None]
        )
        error = identity[indices] - K[indices] @ H[indices]
        variance = error.abs().square().sum((-2, -1)) / (n * gamma2[indices])
        variance = variance + sigma2[indices] * K[indices].abs().square().sum((-2, -1)) / n
        valid = torch.ones(batch, dtype=torch.bool, device=H.device)
        valid[indices] = torch.isfinite(variance) & (variance > 0)
        require_valid(valid, "PG-VAMP", "invalid calibrated variance", H, layer)
        gamma1[indices] = variance.reciprocal()
    require_valid(
        finite_samples(K)
        & finite_samples(r1)
        & torch.isfinite(gamma1)
        & finite_samples(2**0.5 * gamma1[:, None] * r1),
        "PG-VAMP",
        "nonfinite posterior input",
        H,
        layer,
    )
    return {
        **terms,
        "P": P,
        "W": W,
        "K": K,
        "innovation": innovation,
        "xhat2": r2 + innovation,
        "c": c,
        "c_tolerance": tolerance,
        "alpha2": 1 - c,
        "r1": r1,
        "gamma1": gamma1,
        "no_information": ~informed,
    }
