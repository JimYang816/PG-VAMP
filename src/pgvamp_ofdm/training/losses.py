"""The exact supervised complex loss in §15.1."""

import torch


def layer_loss(outputs: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Weighted mean squared complex error over all B*N elements, without detaching."""
    if not outputs or target.ndim != 2 or not target.is_complex():
        raise ValueError("loss requires nonempty layers and complex target [B,N]")
    depth = len(outputs)
    loss = target.real.new_zeros(())
    for t, output in enumerate(outputs, 1):
        if output.shape != target.shape or output.dtype != target.dtype:
            raise ValueError("layer output and target shape/dtype mismatch")
        loss = loss + (2 * t / (depth * (depth + 1))) * (output - target).abs().square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite training loss")
    return loss
