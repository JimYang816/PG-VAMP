"""Ideal complex-I/Q unitary FFT with explicit signed-grid extraction."""

import torch

from ..modulation.allocation import Allocation


def fft_receive(samples: torch.Tensor, allocation: Allocation) -> torch.Tensor:
    """Map finite complex [...,Nfft] time samples to [...,512], on the same device."""
    allocation.validate()
    if (
        not isinstance(samples, torch.Tensor)
        or samples.ndim < 1
        or samples.shape[-1] != allocation.n_fft_wave
        or samples.numel() == 0
        or samples.dtype not in (torch.complex64, torch.complex128)
        or samples.device != allocation.q.device
        or not bool(torch.isfinite(samples).all())
    ):
        raise ValueError("samples must be finite complex [...,Nfft] matching allocation device")
    return torch.fft.fft(samples, norm="ortho")[..., allocation.baseband_fft_index]
