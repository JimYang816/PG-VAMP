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
        validate_system(H, y, sigma2)
        eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
        gram = H.mH @ H
        system = gram / 2 + gram.mH / 2 + sigma2[:, None, None] * eye
        rhs = H.mH @ y[..., None]
        require_valid(
            finite_samples(system) & finite_samples(rhs), self.name, "nonfinite system", H
        )
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        require_valid(info == 0, self.name, "Cholesky failed", H)
        x = torch.cholesky_solve(rhs, factor).squeeze(-1)
        require_valid(finite_samples(x), self.name, "nonfinite solution", H)
        return result(x, None, {"algorithm": self.name})
