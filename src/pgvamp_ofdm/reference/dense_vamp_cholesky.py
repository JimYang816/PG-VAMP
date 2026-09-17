"""Independent exact Cholesky VAMP oracle, specification §13 and §14.6.

No PG operators, graph parameters or production algorithms are imported.
The deliberate duplicate posterior/message arithmetic keeps the oracle readable
and independently checkable against enumeration and the source equations.
"""

import math

import torch
from torch.nn import functional as F

from ..utils.validation import validate_system
from .result import ReferenceResult


def dense_vamp_cholesky(
    H: torch.Tensor,
    y: torch.Tensor,
    sigma2: torch.Tensor,
    *,
    iterations: int = 8,
    return_diagnostics: bool = False,
) -> ReferenceResult:
    """Exact VAMP on H[B,N,N], y[B,N], sigma2[B]>0, paired complex/real dtypes.

    Inputs share one CPU/CUDA device; complex128/float64 is the default contract,
    complex64/float32 is explicit. No learned parameters or implicit conversions.
    """
    validate_system(H, y, sigma2)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    batch, n, _ = H.shape
    identity = torch.eye(n, dtype=H.dtype, device=H.device).expand(batch, n, n)
    gamma_w = 1 / sigma2
    r2, gamma2 = torch.zeros_like(y), torch.ones_like(sigma2)
    gram = gamma_w[:, None, None] * (H.mH @ H)
    layers: list[dict[str, torch.Tensor]] = []
    counts = {
        k: torch.zeros(batch, dtype=torch.int64, device=H.device)
        for k in (
            "no_information",
            "message_rejected",
            "precision_capped",
            "posterior_variance_underflow",
        )
    }
    for t in range(iterations):
        A = gram + gamma2[:, None, None] * identity
        A = (A + A.mH) / 2
        if not bool(torch.isfinite(A).all()):
            raise FloatingPointError(f"VAMP nonfinite A at layer {t}, dtype={H.dtype}")
        try:
            factor = torch.linalg.cholesky(A)
        except torch.linalg.LinAlgError as exc:
            raise FloatingPointError(f"VAMP Cholesky failed at layer {t}, dtype={H.dtype}") from exc
        covariance = torch.cholesky_solve(identity, factor)
        alpha2 = gamma2 * covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1) / n
        W = torch.cholesky_solve(gamma_w[:, None, None] * H.mH, factor)
        # Algebraically 1-alpha2, evaluated as trace(W H) to avoid cancellation.
        c = (W @ H).diagonal(dim1=-2, dim2=-1).real.sum(-1) / n
        finfo = torch.finfo(sigma2.dtype)
        trace_scale = (W.abs() * H.mT.abs()).sum((-2, -1)) / n
        tolerance = (4 * n * finfo.eps / (1 - 4 * n * finfo.eps) * trace_scale).clamp(
            min=finfo.tiny
        )
        if not bool(torch.isfinite(c).all() & torch.isfinite(W).all()):
            raise FloatingPointError(f"VAMP nonfinite linear operator at layer {t}")
        informed = c > tolerance
        counts["no_information"] += ~informed
        delta = (W @ (y - (H @ r2[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        r1, gamma1 = torch.zeros_like(r2), torch.zeros_like(gamma2)
        idx = informed.nonzero(as_tuple=True)[0]
        if idx.numel():
            if not bool((torch.isfinite(alpha2[idx]) & (alpha2[idx] > 0)).all()):
                raise FloatingPointError(f"VAMP invalid alpha2 at layer {t}")
            r1[idx] = r2[idx] + delta[idx] / c[idx, None]
            gamma1[idx] = gamma2[idx] * c[idx] / alpha2[idx]
        u_real = math.sqrt(2) * gamma1[:, None] * r1.real
        u_imag = math.sqrt(2) * gamma1[:, None] * r1.imag
        if not bool(torch.isfinite(u_real).all() & torch.isfinite(u_imag).all()):
            raise FloatingPointError(f"VAMP QPSK input overflow at layer {t}")
        xhat = torch.complex(u_real.tanh(), u_imag.tanh()) / math.sqrt(2)
        er, ei = (-2 * u_real.abs()).exp(), (-2 * u_imag.abs()).exp()
        vbar = ((4 * er / (1 + er).square() + 4 * ei / (1 + ei).square()) / 2).mean(-1)
        alpha1 = gamma1 * vbar
        log_r = torch.stack((F.logsigmoid(2 * u_real), F.logsigmoid(-2 * u_real)), -1)
        log_i = torch.stack((F.logsigmoid(2 * u_imag), F.logsigmoid(-2 * u_imag)), -1)
        probabilities = (log_r[..., :, None] + log_i[..., None, :]).flatten(-2).exp()
        state = {
            "r2": r2,
            "gamma2": gamma2,
            "r1": r1,
            "gamma1": gamma1,
            "xhat1": xhat,
            "probabilities": probabilities,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "c": c,
            "c_tolerance": tolerance,
            "xhat2": r2 + delta,
            "vbar": vbar,
            "A": A,
            "W": W,
        }
        if t + 1 < iterations:
            underflow = vbar == 0
            denom = 1 - alpha1
            eligible = (
                ~underflow
                & torch.isfinite(vbar)
                # Never construct a reciprocal that is unrepresentable.
                & (vbar > 1 / torch.finfo(vbar.dtype).max)
                & torch.isfinite(denom)
                & (denom >= 1e-6)
            )
            accepted = torch.zeros_like(eligible)
            capped = torch.zeros_like(eligible)
            r_next, gamma_next = r2.clone(), gamma2.clone()
            idx = eligible.nonzero(as_tuple=True)[0]
            if idx.numel():
                candidate_r = (xhat[idx] - alpha1[idx, None] * r1[idx]) / denom[idx, None]
                # Branch before reciprocal, because a subsequent clamp can
                # leave an infinite reciprocal derivative in the backward graph.
                cap = vbar[idx] < 1 / (1e8 + gamma1[idx])
                candidate_gamma = torch.full_like(vbar[idx], 1e8)
                uncapped = (~cap).nonzero(as_tuple=True)[0]
                candidate_gamma[uncapped] = 1 / vbar[idx[uncapped]] - gamma1[idx[uncapped]]
                valid = torch.isfinite(candidate_gamma) & (candidate_gamma >= 1e-10)
                valid = valid & torch.isfinite(candidate_r).all(-1)
                take = idx[valid]
                r_next[take] = candidate_r[valid]
                gamma_next[take] = candidate_gamma[valid]
                accepted[take] = True
                capped[take] = cap[valid]
            r2, gamma2 = r_next, gamma_next
            counts["message_rejected"] += ~accepted
            counts["precision_capped"] += capped
            counts["posterior_variance_underflow"] += underflow
            state.update(rejected=~accepted, capped=capped, underflow=underflow)
        if return_diagnostics:
            layers.append(state)
    return ReferenceResult(xhat, probabilities, layers, counts)
