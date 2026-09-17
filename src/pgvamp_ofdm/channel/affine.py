"""Apply physical time scaling to an independent finite-frame waveform."""

import math

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings
from ..waveform.continuous import evaluate_continuous
from ..waveform.frame import FrameWaveforms
from .parameters import PathParameters
from .validity import validate_cp_support


def affine_waveform(
    frame: FrameWaveforms,
    paths: PathParameters,
    times_s: torch.Tensor,
    config: Config,
    *,
    arrival_offset_samples: int = 0,
    downconvert: bool = True,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Receive [B,K] complex128 at recording times [K], with optional integer padding.

    Extra recording offset is removed before propagation and downconversion; it
    does not reset physical path evolution. No amplitude scaling is applied.
    """
    w, _ = _physical_settings(config)
    paths.validate()
    if type(arrival_offset_samples) is not int or arrival_offset_samples < 0:
        raise ValueError("arrival_offset_samples must be nonnegative integer")
    if (
        not isinstance(times_s, torch.Tensor)
        or times_s.dtype != torch.float64
        or times_s.ndim != 1
        or times_s.device != paths.gain.device
        or frame.grid.device != paths.gain.device
        or not bool(torch.isfinite(times_s).all())
    ):
        raise ValueError("times must be finite float64 [K], matching frame/path device")
    t = times_s - arrival_offset_samples / w["sample_rate_hz"]
    result = torch.zeros((frame.grid.shape[0], t.numel()), dtype=torch.complex128, device=t.device)
    for gain, delay, epsilon in zip(paths.gain, paths.delay_s, paths.epsilon):
        result = result + gain * evaluate_continuous(
            frame, (1 + epsilon) * t - delay, config, chunk_size=chunk_size
        )
    if downconvert:
        result = result * torch.exp(-2j * math.pi * w["carrier_hz"] * t)
    return result


def receive_window(
    frame: FrameWaveforms,
    paths: PathParameters,
    config: Config,
    block: int,
    *,
    window_offset_s: float = 0.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Checked noiseless ideal-I/Q [B,Nfft] for the actual block window."""
    w, _ = _physical_settings(config)
    if type(block) is not int or not 0 <= block < len(frame.layout.useful):
        raise ValueError("invalid block index")
    offsets = tuple(window_offset_s if m == block else 0.0 for m in range(len(frame.layout.useful)))
    validate_cp_support(paths, frame.layout, config, window_offsets_s=offsets)
    times = (
        torch.arange(w["n_fft_wave"], dtype=torch.float64, device=frame.grid.device)
        / w["sample_rate_hz"]
        + frame.layout.useful[block][0] / w["sample_rate_hz"]
        + window_offset_s
    )
    return affine_waveform(frame, paths, times, config, chunk_size=chunk_size)
