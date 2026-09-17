"""Data containers only; no numerical operators are shared between the oracles."""

from dataclasses import dataclass, field

import torch


@dataclass
class ReferenceResult:
    """Final posterior [B,N], probabilities [B,N,4], and optional layer tensors."""

    x_soft: torch.Tensor
    probabilities: torch.Tensor
    layers: list[dict[str, torch.Tensor]] = field(default_factory=list)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def class_hat(self) -> torch.Tensor:
        """Smallest class index wins ties; int64 [B,N] on the input device."""
        return self.probabilities.argmax(dim=-1)

    @property
    def bits_hat(self) -> torch.Tensor:
        """Fixed sign-bit labels, int64 [B,N,2], real bit then imaginary bit."""
        classes = self.class_hat
        return torch.stack((classes // 2, classes % 2), dim=-1)
