"""Independent direct Fourier/chirp evaluator of the finite transmit frame.

No channel matrix or finite-sum kernel is used here. Time remains absolute for
the carrier, and local to each useful block for its baseband Fourier series.
"""

import math

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings
from .frame import FrameWaveforms


def evaluate_continuous(
    frame: FrameWaveforms, times_s: torch.Tensor, config: Config, *, chunk_size: int = 512
) -> torch.Tensor:
    """Evaluate arbitrary float64 [K] transmit times, returning complex128 [B,K].

    Input grid may be complex64; its represented symbols are promoted explicitly.
    Half-open segments outside the finite frame, including silence, return zero.
    """
    w, f = _physical_settings(config)
    frame.allocation.validate()
    if (
        not isinstance(times_s, torch.Tensor)
        or times_s.dtype != torch.float64
        or times_s.ndim != 1
        or times_s.device != frame.grid.device
        or not bool(torch.isfinite(times_s).all())
    ):
        raise ValueError("times must be finite float64 [K] on frame device")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be positive integer")
    if (
        frame.grid.ndim != 3
        or frame.grid.shape[1:] != (f["n_ofdm_symbols"], 512)
        or frame.grid.dtype not in (torch.complex64, torch.complex128)
        or not bool(torch.isfinite(frame.grid).all())
        or frame.layout.sample_rate_hz != w["sample_rate_hz"]
    ):
        raise ValueError("frame grid/layout does not match config")
    fs, n = w["sample_rate_hz"], w["n_fft_wave"]
    lfm_start = f["leading_silence_samples"]
    lfm_end = lfm_start + f["lfm_samples"]
    ofdm_start = lfm_end + f["sync_guard_samples"]
    expected_cp = tuple(
        (
            ofdm_start + m * (n + w["cp_samples"]),
            ofdm_start + m * (n + w["cp_samples"]) + w["cp_samples"],
        )
        for m in range(f["n_ofdm_symbols"])
    )
    expected_useful = tuple((end, end + n) for _, end in expected_cp)
    tail_start = expected_useful[-1][1]
    if (
        frame.layout.cp != expected_cp
        or frame.layout.useful != expected_useful
        or frame.layout.lfm != (lfm_start, lfm_end)
        or frame.layout.leading_silence != (0, lfm_start)
        or frame.layout.sync_guard != (lfm_end, ofdm_start)
        or frame.layout.trailing_silence != (tail_start, tail_start + f["trailing_silence_samples"])
        or frame.layout.frame_samples != tail_start + f["trailing_silence_samples"]
    ):
        raise ValueError("frame layout spans do not match physical config")
    grid = frame.grid.to(torch.complex128)
    output = grid.new_zeros((grid.shape[0], times_s.numel()))
    q = frame.allocation.q.to(torch.float64)
    for m, (cp, useful) in enumerate(zip(frame.layout.cp, frame.layout.useful)):
        selected = ((times_s >= cp[0] / fs) & (times_s < useful[1] / fs)).nonzero().flatten()
        for indices in selected.split(chunk_size):
            t = times_s[indices]
            local = t - useful[0] / fs
            basis = torch.exp(2j * math.pi * q[:, None] * (fs / n) * local[None, :])
            output[:, indices] = (
                grid[:, m] @ basis / math.sqrt(n) * torch.exp(2j * math.pi * w["carrier_hz"] * t)
            )
    begin, end = frame.layout.lfm
    selected = ((times_s >= begin / fs) & (times_s < end / fs)).nonzero().flatten()
    local = times_s[selected] - begin / fs
    length, alpha = f["lfm_samples"], f["lfm_tukey_alpha"]
    u = local * fs / (length - 1)
    window = torch.ones_like(u)
    if alpha:
        left, right = u < alpha / 2, u > 1 - alpha / 2
        window[left] = (1 + torch.cos(math.pi * (2 * u[left] / alpha - 1))) / 2
        window[right] = (1 + torch.cos(math.pi * (2 * u[right] / alpha - 2 / alpha + 1))) / 2
        # The continuous symmetric window has no rebound beyond its last sample.
        window[(u < 0) | (u > 1)] = 0
    beta = (f["lfm_stop_hz"] - f["lfm_start_hz"]) / (length / fs)
    phase = 2 * math.pi * (f["lfm_start_hz"] * local + beta * local.square() / 2)
    output[:, selected] = frame.lfm.normalization * window * torch.exp(1j * phase)
    return output
