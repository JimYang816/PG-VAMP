"""Exact SVD VAMP with explicit numerical protection (§13)."""

from typing import Any

import torch
from torch import nn

from ..utils.validation import validate_system
from .base import DetectionResult, finite_samples, require_valid, result
from .messages import nonlinear_message
from .qpsk import qpsk_posterior


class VAMPDetector(nn.Module):
    def __init__(self, iterations: int = 8) -> None:
        super().__init__()
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.iterations = iterations
        self.name = f"VAMP-{iterations}, with explicit numerical protection"

    def detect(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> DetectionResult:
        """One reduced SVD per call; finite square complex H and paired real noise."""
        validate_system(H, y, sigma2)
        batch, n, _ = H.shape
        try:
            U, singular, Vh = torch.linalg.svd(H, full_matrices=False)
        except torch.linalg.LinAlgError as exc:
            raise FloatingPointError(
                f"{self.name} SVD failed; layer=None, batch index={list(range(batch))}, "
                f"dtype={H.dtype}, device={H.device}"
            ) from exc
        require_valid(
            finite_samples(U) & finite_samples(singular) & finite_samples(Vh),
            self.name,
            "nonfinite SVD",
            H,
        )
        gamma_w = 1 / sigma2
        eigenvalues = gamma_w[:, None] * singular.square()
        projected_y = (U.mH @ y[..., None]).squeeze(-1)
        require_valid(
            finite_samples(gamma_w) & finite_samples(eigenvalues) & finite_samples(projected_y),
            self.name,
            "nonfinite spectral operator",
            H,
        )
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
        finfo = torch.finfo(sigma2.dtype)
        relative_bound = 4 * n * finfo.eps / (1 - 4 * n * finfo.eps)
        for layer in range(self.iterations):
            denominator = eigenvalues + gamma2[:, None]
            alpha2 = (gamma2[:, None] / denominator).mean(-1)
            c = (eigenvalues / denominator).mean(-1)
            tolerance = (relative_bound * c).clamp(min=finfo.tiny)
            projected_r = (Vh @ r2[..., None]).squeeze(-1)
            gain = gamma_w[:, None] * singular / denominator
            delta = (Vh.mH @ (gain * (projected_y - singular * projected_r))[..., None]).squeeze(-1)
            require_valid(
                finite_samples(denominator)
                & finite_samples(delta)
                & finite_samples(gain)
                & torch.isfinite(alpha2)
                & (alpha2 > 0)
                & torch.isfinite(c),
                self.name,
                "invalid linear operator",
                H,
                layer,
            )
            informed = c > tolerance
            counts["no_information"] += ~informed
            r1, gamma1 = torch.zeros_like(r2), torch.zeros_like(gamma2)
            indices = informed.nonzero(as_tuple=True)[0]
            r1[indices] = r2[indices] + delta[indices] / c[indices, None]
            gamma1[indices] = gamma2[indices] * c[indices] / alpha2[indices]
            require_valid(
                finite_samples(r1)
                & finite_samples(gamma1)
                & finite_samples(2**0.5 * gamma1[:, None] * r1),
                self.name,
                "nonfinite posterior input",
                H,
                layer,
            )
            posterior = qpsk_posterior(r1, gamma1)
            state = {
                "r2": r2,
                "gamma2": gamma2,
                "xhat2": r2 + delta,
                "alpha2": alpha2,
                "c": c,
                "c_tolerance": tolerance,
                "r1": r1,
                "gamma1": gamma1,
                "xhat1": posterior.mean,
                "probabilities": posterior.probabilities,
                "vbar": posterior.vbar,
                "alpha1": posterior.alpha,
            }
            if layer + 1 < self.iterations:
                message = nonlinear_message(
                    posterior.mean, r1, gamma1, posterior.vbar, posterior.alpha, r2, gamma2
                )
                r2, gamma2 = message.r, message.gamma
                counts["message_rejected"] += message.rejected
                counts["precision_capped"] += message.capped
                counts["posterior_variance_underflow"] += message.underflow
                state.update(
                    rejected=message.rejected, capped=message.capped, underflow=message.underflow
                )
            if return_diagnostics:
                layers.append(state)
        diagnostics: dict[str, Any] = {"algorithm": self.name, **counts}
        if return_diagnostics:
            diagnostics["layers"] = layers
        return result(posterior.mean, posterior.probabilities, diagnostics)
