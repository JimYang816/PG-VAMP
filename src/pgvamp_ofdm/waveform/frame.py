"""Full transmit frame on one carrier time axis; recording offset is separate."""

import math
from dataclasses import dataclass

import torch

from ..config import Config
from ..modulation.allocation import Allocation, _physical_settings, build_allocation, map_grid
from ..modulation.qpsk import bits_to_symbols
from ..utils.device import Runtime
from .lfm import LFMTemplate, _validate_runtime, make_lfm
from .ofdm import OFDMSymbols, modulate_grid


@dataclass(frozen=True)
class FrameLayout:
    """Half-open sample intervals, all relative to transmit-frame origin."""

    leading_silence: tuple[int, int]
    lfm: tuple[int, int]
    sync_guard: tuple[int, int]
    cp: tuple[tuple[int, int], ...]
    useful: tuple[tuple[int, int], ...]
    trailing_silence: tuple[int, int]
    frame_samples: int
    sample_rate_hz: float
    data_bits_per_frame: int


@dataclass(frozen=True)
class FrameWaveforms:
    """Complex/real [B,Nframe], grid [B,M,512], OFDM and LFM components."""

    analytic: torch.Tensor
    real: torch.Tensor
    grid: torch.Tensor
    ofdm: OFDMSymbols
    lfm: LFMTemplate
    layout: FrameLayout
    allocation: Allocation


def build_frame(
    data_bits: torch.Tensor,
    pilot_symbols: torch.Tensor,
    config: Config,
    runtime: Runtime,
) -> FrameWaveforms:
    """Build from integer bits [B,M,400,2] and complex pilots [B,M,64], same runtime device.

    No RNG is consumed. LFM phase is local to its definition; OFDM carrier phase
    uses global transmit sample positions. Output RF is sqrt(2)*analytic.real.
    """
    w, f = _physical_settings(config)
    _validate_runtime(config, runtime)
    if not isinstance(data_bits, torch.Tensor) or data_bits.ndim != 4:
        raise ValueError("frame bits must have shape [B,M,400,2]")
    if data_bits.shape[1:] != (f["n_ofdm_symbols"], 400, 2) or data_bits.shape[0] == 0:
        raise ValueError("frame bits must have nonempty shape [B,M,400,2] matching config")
    if not isinstance(pilot_symbols, torch.Tensor) or pilot_symbols.dtype != runtime.complex_dtype:
        raise ValueError("pilots must match runtime complex dtype")
    if data_bits.device != runtime.device or pilot_symbols.device != runtime.device:
        raise ValueError("bits and pilots must match runtime device")
    allocation = build_allocation(config, runtime.device)
    data = bits_to_symbols(data_bits, dtype=runtime.complex_dtype)
    grid = map_grid(data, pilot_symbols, allocation)
    ofdm, lfm = modulate_grid(grid, config), make_lfm(config, runtime)
    start = f["leading_silence_samples"]
    lfm_span = (start, start + f["lfm_samples"])
    guard = (lfm_span[1], lfm_span[1] + f["sync_guard_samples"])
    cp_spans, useful_spans = [], []
    start = guard[1]
    for _ in range(f["n_ofdm_symbols"]):
        cp_spans.append((start, start + w["cp_samples"]))
        useful_spans.append((cp_spans[-1][1], cp_spans[-1][1] + w["n_fft_wave"]))
        start = useful_spans[-1][1]
    tail = (start, start + f["trailing_silence_samples"])
    layout = FrameLayout(
        (0, lfm_span[0]),
        lfm_span,
        guard,
        tuple(cp_spans),
        tuple(useful_spans),
        tail,
        tail[1],
        w["sample_rate_hz"],
        f["n_ofdm_symbols"] * 800,
    )
    analytic = grid.new_zeros((grid.shape[0], layout.frame_samples))
    analytic[:, lfm_span[0] : lfm_span[1]] = lfm.analytic
    t = (
        torch.arange(guard[1], tail[0], dtype=torch.float64, device=runtime.device)
        / w["sample_rate_hz"]
    )
    carrier = torch.exp(2j * math.pi * w["carrier_hz"] * t).to(runtime.complex_dtype)
    analytic[:, guard[1] : tail[0]] = ofdm.with_cp.flatten(1) * carrier
    return FrameWaveforms(
        analytic, math.sqrt(2) * analytic.real, grid, ofdm, lfm, layout, allocation
    )


def pad_recording(waveform: torch.Tensor, arrival_offset_samples: int) -> torch.Tensor:
    """Prepend integer >=0 zeros to finite real/complex [...,N]; do not remodulate."""
    if type(arrival_offset_samples) is not int or arrival_offset_samples < 0:
        raise ValueError("arrival_offset_samples must be a nonnegative integer")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim < 1 or waveform.numel() == 0:
        raise ValueError("waveform must have nonempty shape [...,N]")
    if waveform.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise ValueError("waveform must have paired real or complex floating dtype")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must be finite")
    return torch.cat(
        (waveform.new_zeros((*waveform.shape[:-1], arrival_offset_samples)), waveform), -1
    )
