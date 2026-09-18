"""Production PG-VAMP-VC, with exactly two real learned scalars per layer."""

import math
from typing import Any

import torch
from torch import nn

from ...utils.validation import validate_system
from ..base import DetectionResult, result
from ..messages import nonlinear_message
from ..qpsk import qpsk_posterior
from .linear import linear_layer
from .topology import soft_mask, thresholds


class PGVAMPDetector(nn.Module):
    """Dense §14 implementation; exact work may be cubic, storage quadratic.

    Parameters must explicitly match the input real dtype/device. Detailed
    diagnostics retain autograd graphs and large matrices for mathematical tests.
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
        mask_mode: str = "soft",
    ) -> None:
        super().__init__()
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        if dtype not in (torch.float64, torch.float32):
            raise ValueError("PG parameters require float64 or float32")
        if not all(
            math.isfinite(v)
            for v in (rho_hi_db, rho_lo_db, min_gap_db, temperature_db, init_mu, jitter)
        ):
            raise ValueError("PG settings must be finite")
        if min_gap_db <= 0 or temperature_db <= 0 or jitter < 0 or not 0 < init_mu < 1:
            raise ValueError("invalid PG gap/temperature/mu/jitter")
        if rho_hi_db - rho_lo_db - depth * min_gap_db <= 0:
            raise ValueError("threshold span minus depth*min_gap must be positive")
        if mask_mode != "soft":
            raise ValueError("PG-VAMP supports only soft masks")
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("PG device must be CPU or explicit CUDA")
        if selected.type == "cuda":
            from ...utils.device import resolve_runtime

            selected = resolve_runtime(
                str(selected), "complex128" if dtype == torch.float64 else "complex64"
            ).device
        self.depth = depth
        self.rho_hi_db, self.rho_lo_db = rho_hi_db, rho_lo_db
        self.min_gap_db, self.temperature_db, self.jitter = min_gap_db, temperature_db, jitter
        self.raw_gaps = nn.Parameter(
            torch.full((depth,), math.log(math.expm1(1 / depth)), dtype=dtype, device=selected)
        )
        self.raw_mu = nn.Parameter(
            torch.full((depth,), math.log(init_mu / (1 - init_mu)), dtype=dtype, device=selected)
        )

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return thresholds(
            self.raw_gaps, self.raw_mu, self.rho_hi_db, self.rho_lo_db, self.min_gap_db
        )

    def _mask(self, H: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return soft_mask(H, rho, self.temperature_db)

    def forward(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> DetectionResult:
        validate_system(H, y, sigma2)
        for parameter in (self.raw_gaps, self.raw_mu):
            if parameter.dtype != sigma2.dtype or parameter.device != H.device:
                raise ValueError("PG real parameters must match input real dtype/device")
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError("PG raw parameters must be finite")
        batch, n, _ = H.shape
        rho, mu = self.thresholds()
        r2, gamma2 = torch.zeros_like(y), torch.ones_like(sigma2)
        counts = {
            key: torch.zeros(batch, dtype=torch.int64, device=H.device)
            for key in (
                "no_information",
                "message_rejected",
                "precision_capped",
                "posterior_variance_underflow",
            )
        }
        layers: list[dict[str, torch.Tensor]] = []
        summaries: list[dict[str, torch.Tensor]] = []
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=H.device)
        candidates = (H.abs() > 0) & off_diagonal
        candidate_count = candidates.sum((-2, -1))
        for t in range(self.depth):
            mask = self._mask(H, rho[t])
            state = linear_layer(H, y, sigma2, r2, gamma2, mask, mu[t], jitter=self.jitter, layer=t)
            posterior = qpsk_posterior(state["r1"], state["gamma1"])
            counts["no_information"] += state["no_information"]
            state.update(
                r2=r2,
                gamma2=gamma2,
                mask=mask,
                rho=rho[t],
                mu=mu[t],
                xhat1=posterior.mean,
                probabilities=posterior.probabilities,
                vbar=posterior.vbar,
                alpha1=posterior.alpha,
            )
            if t + 1 < self.depth:
                message = nonlinear_message(
                    posterior.mean,
                    state["r1"],
                    state["gamma1"],
                    posterior.vbar,
                    posterior.alpha,
                    r2,
                    gamma2,
                )
                r2, gamma2 = message.r, message.gamma
                counts["message_rejected"] += message.rejected
                counts["precision_capped"] += message.capped
                counts["posterior_variance_underflow"] += message.underflow
                state.update(
                    rejected=message.rejected, capped=message.capped, underflow=message.underflow
                )
            active = (candidates & (mask >= 0.5)).sum((-2, -1))
            diagonal = state["G"].diagonal(dim1=-2, dim2=-1).real
            # A common scale cancels from the ratio and avoids squaring huge
            # or tiny finite energies in the logging-only norm calculation.
            scale = torch.maximum(diagonal.abs().amax(-1), state["ell"].abs().amax(-1))
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            baseline = torch.linalg.vector_norm(diagonal / scale[:, None], dim=-1)
            safe_baseline = torch.where(baseline == 0, torch.ones_like(baseline), baseline)
            summary = {key: state[key] for key in ("rho", "mu", "c", "c_tolerance")}
            summary.update(
                jitter=torch.full_like(sigma2, self.jitter),
                candidate_edges=candidate_count,
                all_off_diagonal_edges=torch.full_like(candidate_count, n * (n - 1)),
                effective_candidate_ratio=active.to(sigma2.dtype) / candidate_count.clamp(min=1),
                effective_all_ratio=active.to(sigma2.dtype) / max(n * (n - 1), 1),
                safety_relative=torch.linalg.vector_norm(state["ell"] / scale[:, None], dim=-1)
                / safe_baseline,
                safety_zero_baseline=baseline == 0,
            )
            # Only small logging copies are detached; detailed states stay differentiable.
            summaries.append({key: value.detach() for key, value in summary.items()})
            if return_diagnostics:
                layers.append(state)
        diagnostics: dict[str, Any] = {
            "algorithm": "PG-VAMP-VC",
            **counts,
            "layer_summaries": summaries,
        }
        if return_diagnostics:
            diagnostics["layers"] = layers
        return result(posterior.mean, posterior.probabilities, diagnostics)

    def detect(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> DetectionResult:
        return self(H, y, sigma2, return_diagnostics=return_diagnostics)
