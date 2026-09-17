"""Full grid closed form, using waveform FFT length without ICI truncation."""

import math

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings, build_allocation
from ..waveform.frame import FrameLayout
from .parameters import PathParameters
from .validity import validate_cp_support


def dirichlet_kernel(z: torch.Tensor, length: int) -> torch.Tensor:
    """Normalized finite exponential sum; periodic reduction handles every removable pole."""
    if (
        not isinstance(z, torch.Tensor)
        or z.dtype != torch.float64
        or not bool(torch.isfinite(z).all())
    ):
        raise ValueError("z must be finite float64")
    if type(length) is not int or length < 1:
        raise ValueError("length must be positive integer")
    reduced = torch.remainder(z + length / 2, length) - length / 2
    return (
        torch.exp(1j * math.pi * (length - 1) / length * reduced)
        * torch.sinc(reduced)
        / torch.sinc(reduced / length)
    )


def effective_matrix(
    paths: PathParameters,
    layout: FrameLayout,
    config: Config,
    block: int,
    *,
    window_offset_s: float = 0.0,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Return [512,512] on paths.device; construct in complex128 then explicitly cast.

    The offset is relative to this block's useful start. All other windows remain ideal.
    """
    w, _ = _physical_settings(config)
    if type(block) is not int or not 0 <= block < len(layout.useful):
        raise ValueError("invalid block index")
    if dtype not in (torch.complex128, torch.complex64):
        raise ValueError("output dtype must be complex128 or complex64")
    offsets = tuple(window_offset_s if m == block else 0.0 for m in range(len(layout.useful)))
    validate_cp_support(paths, layout, config, window_offsets_s=offsets)
    q = build_allocation(config, paths.gain.device).q.to(torch.float64)
    fs, n = w["sample_rate_hz"], w["n_fft_wave"]
    df = fs / n
    frequency = w["carrier_hz"] + q * df
    time = layout.useful[block][0] / fs
    h = torch.zeros((512, 512), dtype=torch.complex128, device=q.device)
    for gain, delay, epsilon in zip(paths.gain, paths.delay_s, paths.epsilon):
        nu = epsilon * frequency
        phase = -frequency * (delay - epsilon * time) + (q * df + nu) * window_offset_s
        h = h + gain * torch.exp(2j * math.pi * phase)[None, :] * dirichlet_kernel(
            q[None, :] - q[:, None] + nu[None, :] / df, n
        )
    if not bool(torch.isfinite(h).all()):
        raise ValueError("nonfinite effective matrix")
    return h.to(dtype)
