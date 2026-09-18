"""Label-free detector interface (§11) and contextual numerical failures."""

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..modulation.qpsk import classes_to_bits, hard_decision


@dataclass
class DetectionResult:
    """Soft complex [B,N], int64 classes, uint8 bits [B,N,2], real probabilities."""

    x_soft: torch.Tensor
    class_hat: torch.Tensor
    bits_hat: torch.Tensor
    probabilities: torch.Tensor | None
    diagnostics: dict[str, Any]


class Detector(Protocol):
    def detect(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> DetectionResult: ...


def require_valid(
    valid: torch.Tensor, algorithm: str, operation: str, H: torch.Tensor, layer: int | None = None
) -> None:
    """Raise with all affected batch indices; callers retain sample identities."""
    if not bool(valid.all()):
        indices = (~valid).nonzero(as_tuple=True)[0].tolist()
        raise FloatingPointError(
            f"{algorithm} {operation}; layer={layer}, batch index={indices}, "
            f"dtype={H.dtype}, device={H.device}"
        )


def finite_samples(tensor: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(-1)


def result(
    x: torch.Tensor, probabilities: torch.Tensor | None, diagnostics: dict[str, Any]
) -> DetectionResult:
    classes = hard_decision(x)
    return DetectionResult(x, classes, classes_to_bits(classes), probabilities, diagnostics)
