"""Validation shared only at reference input boundaries, never numerical operators."""

import torch


def validate_system(H: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor) -> None:
    """Require finite H[B,N,N], y[B,N], sigma2[B]>0, paired dtypes and one device."""
    if not all(isinstance(t, torch.Tensor) for t in (H, y, sigma2)):
        raise ValueError("H, y, sigma2 must be tensors")
    if H.ndim != 3 or H.shape[-1] != H.shape[-2] or min(H.shape) < 1:
        raise ValueError("H must have nonempty square shape [B,N,N]")
    if y.shape != H.shape[:2] or sigma2.shape != H.shape[:1]:
        raise ValueError("y must have shape [B,N] and sigma2 [B]")
    if H.dtype not in (torch.complex128, torch.complex64) or y.dtype != H.dtype:
        raise ValueError("H and y must share complex128 or complex64 dtype")
    real_dtype = torch.float64 if H.dtype == torch.complex128 else torch.float32
    if sigma2.dtype != real_dtype:
        raise ValueError("sigma2 must have the paired real dtype")
    if y.device != H.device or sigma2.device != H.device:
        raise ValueError("H, y, sigma2 must share a device")
    if not all(bool(torch.isfinite(t).all()) for t in (H, y, sigma2)):
        raise ValueError("H, y, sigma2 must be finite")
    if not bool((sigma2 > 0).all()):
        raise ValueError("sigma2 must be strictly positive")
