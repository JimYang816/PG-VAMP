"""Full-H Cholesky linear MMSE (§12), with no posterior postprocessing."""

import torch
from torch import nn

from ..utils.validation import validate_system
from .base import DetectionResult, finite_samples, require_valid, result


class MMSEDetector(nn.Module):
    name = "MMSE (linear)"

    def detect(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> DetectionResult:
        """Detect from finite square H[B,N,N], y[B,N], paired sigma2[B]>0."""
        return self._detect(H, y, sigma2)

    def _factor(self, H: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
        gram = H.mH @ H
        system = gram / 2 + gram.mH / 2 + sigma2[:, None, None] * eye
        require_valid(finite_samples(system), self.name, "nonfinite system", H)
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        require_valid(info == 0, self.name, "Cholesky failed", H)
        return factor

    def _detect(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        factor: torch.Tensor | None = None,
    ) -> DetectionResult:
        validate_system(H, y, sigma2)
        rhs = H.mH @ y[..., None]
        require_valid(finite_samples(rhs), self.name, "nonfinite system", H)
        if factor is None:
            factor = self._factor(H, sigma2)
        x = torch.cholesky_solve(rhs, factor).squeeze(-1)
        require_valid(finite_samples(x), self.name, "nonfinite solution", H)
        return result(x, None, {"algorithm": self.name})
