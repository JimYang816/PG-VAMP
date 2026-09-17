"""8192-point default unitary OFDM; no oversampling amplitude compensation."""

from dataclasses import dataclass

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings, build_allocation


@dataclass(frozen=True)
class OFDMSymbols:
    """Complex [...,Nw] useful and [...,Ncp+Nw] CP-prefixed samples."""

    useful: torch.Tensor
    with_cp: torch.Tensor


def modulate_grid(grid: torch.Tensor, config: Config) -> OFDMSymbols:
    """IFFT finite complex128/64 [...,512] on its device; transform last axis only."""
    w, _ = _physical_settings(config)
    if not isinstance(grid, torch.Tensor) or grid.ndim < 1 or grid.shape[-1] != 512:
        raise ValueError("grid must have shape [...,512]")
    if grid.dtype not in (torch.complex128, torch.complex64) or grid.numel() == 0:
        raise ValueError("grid must be nonempty complex128/complex64")
    if not bool(torch.isfinite(grid).all()):
        raise ValueError("grid must be finite")
    allocation = build_allocation(config, grid.device)
    spectrum = grid.new_zeros((*grid.shape[:-1], w["n_fft_wave"]))
    spectrum[..., allocation.baseband_fft_index] = grid
    useful = torch.fft.ifft(spectrum, dim=-1, norm="ortho")
    if not bool(torch.isfinite(useful).all()):
        raise ValueError("OFDM samples are not representable at this amplitude")
    return OFDMSymbols(useful, torch.cat((useful[..., -w["cp_samples"] :], useful), dim=-1))
