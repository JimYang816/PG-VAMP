"""Independent dense PG-VAMP-VC oracle, engineering specification §§14.1–14.8.

This module intentionally owns its numerical calculations. It imports no
production algorithm or VAMP numerical helper. All linear algebra is batched.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.validation import validate_system
from .result import ReferenceResult


def _posterior(r1: torch.Tensor, gamma1: torch.Tensor) -> tuple[torch.Tensor, ...]:
    u_real = math.sqrt(2) * gamma1[:, None] * r1.real
    u_imag = math.sqrt(2) * gamma1[:, None] * r1.imag
    if not bool(torch.isfinite(u_real).all() & torch.isfinite(u_imag).all()):
        raise FloatingPointError("PG QPSK input overflow")
    xhat = torch.complex(u_real.tanh(), u_imag.tanh()) / math.sqrt(2)
    exp_real, exp_imag = (-2 * u_real.abs()).exp(), (-2 * u_imag.abs()).exp()
    variance = (4 * exp_real / (1 + exp_real).square() + 4 * exp_imag / (1 + exp_imag).square()) / 2
    # Equivalent to -gamma*|r-a|^2, with the class-independent term removed.
    # Separate sign log probabilities avoid squaring an enormous r.
    real_log = torch.stack((F.logsigmoid(2 * u_real), F.logsigmoid(-2 * u_real)), -1)
    imag_log = torch.stack((F.logsigmoid(2 * u_imag), F.logsigmoid(-2 * u_imag)), -1)
    probabilities = (real_log[..., :, None] + imag_log[..., None, :]).flatten(-2).exp()
    vbar = variance.mean(-1)
    return xhat, probabilities, vbar, gamma1 * vbar


def _extrinsic(
    old_r: torch.Tensor,
    old_gamma: torch.Tensor,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    xhat: torch.Tensor,
    vbar: torch.Tensor,
    alpha1: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    r2, gamma2 = old_r.clone(), old_gamma.clone()
    underflow = vbar == 0
    denominator = 1 - alpha1
    possible = (
        ~underflow
        & torch.isfinite(vbar)
        # Reject an unrepresentable reciprocal before building its autograd node.
        & (vbar > 1 / torch.finfo(vbar.dtype).max)
        & torch.isfinite(denominator)
        & (denominator >= 1e-6)
    )
    accepted = torch.zeros_like(possible)
    capped = torch.zeros_like(possible)
    index = possible.nonzero(as_tuple=True)[0]
    if index.numel():
        candidate = (xhat[index] - alpha1[index, None] * r1[index]) / denominator[index, None]
        # Cap before reciprocal evaluation: clamp(1/v) still evaluates the
        # overflowing derivative -1/v**2, yielding 0*Inf=NaN in backward.
        cap = vbar[index] < 1 / (1e8 + gamma1[index])
        precision = torch.full_like(vbar[index], 1e8)
        uncapped = (~cap).nonzero(as_tuple=True)[0]
        precision[uncapped] = 1 / vbar[index[uncapped]] - gamma1[index[uncapped]]
        valid = torch.isfinite(precision) & (precision >= 1e-10)
        valid = valid & torch.isfinite(candidate).all(-1)
        valid_index = index[valid]
        r2[valid_index] = candidate[valid]
        gamma2[valid_index] = precision[valid]
        accepted[valid_index] = True
        capped[valid_index] = cap[valid]
    return r2, gamma2, ~accepted, capped, underflow


class DensePGVAMP(nn.Module):
    """Small dense oracle with exactly raw_gaps[T], raw_mu[T] real parameters.

    Forward requires H[B,N,N], y[B,N], sigma2[B], complex128/float64 by
    default. Explicit complex64 uses float32 parameters. Move the module to
    the input device/real dtype before calling. No automatic conversion occurs.
    """

    def __init__(
        self,
        depth: int = 8,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        rho_hi_db: float = 0,
        rho_lo_db: float = -60,
        min_gap_db: float = 0.5,
        temperature_db: float = 3,
        init_mu: float = 0.8,
        jitter: float = 0,
    ) -> None:
        super().__init__()
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        if dtype not in (torch.float64, torch.float32):
            raise ValueError("PG parameters require float64 or float32")
        values = (rho_hi_db, rho_lo_db, min_gap_db, temperature_db, init_mu, jitter)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("PG settings must be finite")
        if min_gap_db <= 0 or temperature_db <= 0 or jitter < 0 or not 0 < init_mu < 1:
            raise ValueError("invalid PG gap/temperature/mu/jitter")
        if rho_hi_db - rho_lo_db - depth * min_gap_db <= 0:
            raise ValueError("threshold span minus depth*min_gap must be positive")
        self.depth = depth
        self.rho_hi_db, self.rho_lo_db = rho_hi_db, rho_lo_db
        self.min_gap_db, self.temperature_db, self.jitter = min_gap_db, temperature_db, jitter
        self.raw_gaps = nn.Parameter(
            torch.full((depth,), math.log(math.expm1(1 / depth)), dtype=dtype, device=device)
        )
        self.raw_mu = nn.Parameter(
            torch.full((depth,), math.log(init_mu / (1 - init_mu)), dtype=dtype, device=device)
        )

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return monotone rho[T] and mu[T], preserving all raw-parameter gradients."""
        gaps = F.softplus(self.raw_gaps)
        indices = torch.arange(1, self.depth + 1, device=gaps.device, dtype=gaps.dtype)
        span = self.rho_hi_db - self.rho_lo_db - self.depth * self.min_gap_db
        rho = self.rho_hi_db - indices * self.min_gap_db - span * gaps.cumsum(0) / (1 + gaps.sum())
        return rho, self.raw_mu.sigmoid()

    def _mask(self, H: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        energy = H.abs().square()
        columns = energy.sum(-2, keepdim=True)
        # Empty columns have exactly zero score probability; no 0/0 is evaluated.
        denominator = columns.clone()
        denominator[columns == 0] = 1
        score = 10 * (energy / denominator).clamp(min=1e-12).log10()
        mask = ((score - rho) / self.temperature_db).sigmoid()
        mask = mask * (energy > 0)
        diagonal = torch.eye(H.shape[-1], device=H.device, dtype=torch.bool)
        return mask.masked_fill(diagonal, 1)

    def forward(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> ReferenceResult:
        """Run complete §14 loop; return final QPSK posterior, never extrinsic mean."""
        validate_system(H, y, sigma2)
        if self.raw_gaps.dtype != sigma2.dtype or self.raw_gaps.device != H.device:
            raise ValueError("PG real parameters must match input real dtype/device")
        if self.raw_mu.dtype != sigma2.dtype or self.raw_mu.device != H.device:
            raise ValueError("PG raw_mu must match input real dtype/device")
        if not bool(torch.isfinite(self.raw_gaps).all() & torch.isfinite(self.raw_mu).all()):
            raise ValueError("PG raw parameters must be finite")
        batch, n, _ = H.shape
        identity = torch.eye(n, device=H.device, dtype=H.dtype).expand(batch, n, n)
        r2, gamma2 = torch.zeros_like(y), torch.ones_like(sigma2)
        gamma_w = sigma2.reciprocal()
        rho, mu = self.thresholds()
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
        for t in range(self.depth):
            mask = self._mask(H, rho[t])
            magnitude = H.abs()
            U, RF = mask * magnitude, (1 - mask) * magnitude
            d = ((1 - mask.square()) * magnitude.square()).sum(-2)

            # Exclusive row sums use cumsums, avoiding cancellation of large
            # diagonal magnitudes when off-diagonal terms are tiny.
            def exclusive_sum(V: torch.Tensor) -> torch.Tensor:
                zero = torch.zeros_like(V[..., :1])
                left = torch.cat((zero, V.cumsum(-1)[..., :-1]), dim=-1)
                right = torch.cat((V.flip(-1).cumsum(-1)[..., :-1].flip(-1), zero), dim=-1)
                return left + right

            ell = (RF * exclusive_sum(magnitude) + U * exclusive_sum(RF)).sum(-2)
            HD = mask * H
            G = HD.mH @ HD + torch.diag_embed(d)
            Gbar = G + torch.diag_embed(ell)
            P = gamma_w[:, None, None] * Gbar + (gamma2 + self.jitter)[:, None, None] * identity
            P = (P + P.mH) / 2
            if not bool(torch.isfinite(P).all()):
                raise FloatingPointError(f"PG nonfinite majorizer at layer {t}, dtype={H.dtype}")
            try:
                factor = torch.linalg.cholesky(P)
            except torch.linalg.LinAlgError as exc:
                raise FloatingPointError(
                    f"PG Cholesky failed at layer {t}, dtype={H.dtype}"
                ) from exc

            def apply_A(V: torch.Tensor) -> torch.Tensor:
                return gamma_w[:, None, None] * (H.mH @ (H @ V)) + gamma2[:, None, None] * V

            def apply_B(V: torch.Tensor) -> torch.Tensor:
                q = torch.cholesky_solve(V, factor)
                return q + mu[t] * torch.cholesky_solve(V - apply_A(q), factor)

            u = gamma_w[:, None, None] * (H.mH @ (y - (H @ r2[..., None]).squeeze(-1))[..., None])
            innovation = apply_B(u).squeeze(-1)
            W = apply_B(gamma_w[:, None, None] * H.mH)
            F2 = W @ H
            c = F2.diagonal(dim1=-2, dim2=-1).real.sum(-1) / n
            # Trace contraction error scale: sum_ij |W_ij||H_ji| / N.
            # 4N real rounding operations conservatively cover complex products
            # and trace summation: gamma_(4N)=4N*eps/(1-4N*eps).
            # There is NO max(scale,1) floor: weak but resolved signals survive.
            # Below tiny, normalized division no longer has normal precision.
            finfo = torch.finfo(sigma2.dtype)
            error_factor = 4 * n * finfo.eps / (1 - 4 * n * finfo.eps)
            trace_scale = (W.abs() * H.mT.abs()).sum((-2, -1)) / n
            tolerance = (error_factor * trace_scale).clamp(min=finfo.tiny)
            if not bool(torch.isfinite(c).all() & torch.isfinite(W).all()):
                raise FloatingPointError(f"PG nonfinite linear operator at layer {t}")
            informed = c > tolerance
            counts["no_information"] += ~informed
            # A wholly uninformative batch still has a valid zero derivative
            # with respect to the trainable parameters, rather than no graph.
            zero_dependency = 0 * rho[t] + 0 * mu[t]
            r1 = torch.zeros_like(r2) + zero_dependency
            gamma1 = torch.zeros_like(gamma2) + zero_dependency
            K = torch.zeros_like(W)
            idx = informed.nonzero(as_tuple=True)[0]
            if idx.numel():
                # §14.5 quotient, evaluated in two stages so its local
                # derivative does not overflow for resolved weak information.
                root = torch.sqrt(c[idx])
                matrix_root = root[:, None, None]
                K[idx] = torch.complex(
                    (W[idx].real / matrix_root) / matrix_root,
                    (W[idx].imag / matrix_root) / matrix_root,
                )
                vector_root = root[:, None]
                r1[idx] = r2[idx] + torch.complex(
                    (innovation[idx].real / vector_root) / vector_root,
                    (innovation[idx].imag / vector_root) / vector_root,
                )
                variance = (identity[idx] - K[idx] @ H[idx]).abs().square().sum((-2, -1))
                variance = variance / (n * gamma2[idx])
                variance = variance + sigma2[idx] * K[idx].abs().square().sum((-2, -1)) / n
                if not bool((torch.isfinite(variance) & (variance > 0)).all()):
                    raise FloatingPointError(f"PG invalid calibrated variance at layer {t}")
                gamma1[idx] = 1 / variance
            xhat, probabilities, vbar, alpha1 = _posterior(r1, gamma1)
            state = {
                "r2": r2,
                "gamma2": gamma2,
                "r1": r1,
                "gamma1": gamma1,
                "xhat1": xhat,
                "probabilities": probabilities,
                "alpha1": alpha1,
                "alpha2": 1 - c,
                "c": c,
                "c_tolerance": tolerance,
                "rho": rho[t],
                "mu": mu[t],
                "mask": mask,
                "d": d,
                "ell": ell,
                "G": G,
                "Gbar": Gbar,
                "P": P,
                "W": W,
                "K": K,
                "innovation": innovation,
                "xhat2": r2 + innovation,
                "vbar": vbar,
            }
            if t + 1 < self.depth:
                r2, gamma2, rejected, capped, underflow = _extrinsic(
                    r2, gamma2, r1, gamma1, xhat, vbar, alpha1
                )
                counts["message_rejected"] += rejected
                counts["precision_capped"] += capped
                counts["posterior_variance_underflow"] += underflow
                state.update(rejected=rejected, capped=capped, underflow=underflow)
            if return_diagnostics:
                layers.append(state)
        return ReferenceResult(xhat, probabilities, layers, counts)
