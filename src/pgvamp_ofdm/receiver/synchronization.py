"""Normalized valid-lag matched correlation (§7.1), never oracle-assisted timing."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyncResult:
    """Scores [...,L-N+1]; peak fields [...] and candidate_valid masks candidate_arrival.

    Invalid candidate_arrival is -1. A peak is a template start, not a frame origin
    or earliest physical path. Exact peak ties use the smallest lag.
    """

    scores: torch.Tensor
    peak_start: torch.Tensor
    peak_score: torch.Tensor
    detected: torch.Tensor
    candidate_arrival: torch.Tensor
    candidate_valid: torch.Tensor
    threshold: float
    epsilon_sync: float


def normalized_correlation(
    recording: torch.Tensor,
    template: torch.Tensor,
    threshold: float = 0.1,
) -> SyncResult:
    """Correlate finite real/complex [...,L] with 1D [N], paired precision/device.

    FFT convolution is zero padded for linear correlation. Prefix energies use
    float64 accumulation for both precisions; exact zero windows are identified
    by nonzero sample counts so FFT leakage cannot become a false detection.
    Unrepresentable intermediate amplitudes raise ValueError, never clip scores.
    """
    if not all(isinstance(t, torch.Tensor) for t in (recording, template)):
        raise ValueError("recording and template must be tensors")
    if recording.ndim < 1 or template.ndim != 1 or min(recording.shape) < 1 or template.numel() < 1:
        raise ValueError("recording [...,L] and template [N] must be nonempty")
    if recording.shape[-1] < template.numel():
        raise ValueError("recording must be at least as long as template")
    pairs = {torch.float32: 32, torch.complex64: 32, torch.float64: 64, torch.complex128: 64}
    if recording.dtype not in pairs or template.dtype not in pairs:
        raise ValueError("recording/template require float32/64 or complex64/128")
    if pairs[recording.dtype] != pairs[template.dtype] or recording.device != template.device:
        raise ValueError("recording/template require paired precision and same device")
    if not all(bool(torch.isfinite(t).all()) for t in (recording, template)):
        raise ValueError("recording/template must be finite")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError("threshold must be finite in (0,1]")
    real_dtype = recording.real.dtype
    epsilon = torch.finfo(real_dtype).tiny
    energy_samples = recording.abs().square()
    template_samples = template.abs().square()
    if bool(((recording != 0) & (energy_samples == 0)).any()) or bool(
        ((template != 0) & (template_samples == 0)).any()
    ):
        raise ValueError("nonzero sample energy underflow is not representable")
    template_energy = template_samples.sum()
    if not bool(torch.isfinite(energy_samples).all()) or not bool(torch.isfinite(template_energy)):
        raise ValueError("signal energies are not representable")
    if float(template_energy) <= 0:
        raise ValueError("template must have positive representable energy")
    n, length = template.numel(), recording.shape[-1]
    fft_length = 1 << (length + n - 2).bit_length()
    kernel = template.flip(-1).conj()
    convolution = torch.fft.ifft(
        torch.fft.fft(recording, n=fft_length, dim=-1) * torch.fft.fft(kernel, n=fft_length), dim=-1
    )
    magnitude = convolution[..., n - 1 : length].abs()
    numerator = magnitude.square()
    prefix = torch.cat(
        (
            energy_samples.new_zeros((*recording.shape[:-1], 1), dtype=torch.float64),
            energy_samples.cumsum(-1, dtype=torch.float64),
        ),
        -1,
    )
    window_energy = prefix[..., n:] - prefix[..., :-n]
    tolerance = 8 * torch.finfo(torch.float64).eps * prefix[..., -1:]
    if bool((window_energy < -tolerance).any()):
        raise ValueError("negative sliding energy beyond roundoff tolerance")
    window_energy = window_energy.clamp_min(0).to(real_dtype)
    counts = torch.cat(
        (
            torch.zeros((*recording.shape[:-1], 1), device=recording.device, dtype=torch.int64),
            (recording != 0).long().cumsum(-1),
        ),
        -1,
    )
    nonzero = counts[..., n:] != counts[..., :-n]
    # A positive window lost to prefix cancellation must not be silently treated as zero.
    if bool((nonzero & (window_energy <= 0)).any()):
        raise ValueError(
            "sliding energy lost to cancellation; amplitude range is not representable"
        )
    numerator = torch.where(nonzero, numerator, torch.zeros_like(numerator))
    energy_product = window_energy * template_energy
    if bool((nonzero & (energy_product == 0)).any()) or bool(
        (nonzero & (magnitude != 0) & (numerator == 0)).any()
    ):
        raise ValueError("correlation intermediate underflow is not representable")
    denominator = energy_product + epsilon
    scores = numerator / denominator
    if not all(bool(torch.isfinite(t).all()) for t in (numerator, denominator, scores)):
        raise ValueError("correlation intermediates are not representable")
    peak_score, peak_start = scores.max(dim=-1)
    detected = peak_score > threshold
    candidate = torch.where(detected, peak_start, torch.full_like(peak_start, -1))
    return SyncResult(
        scores, peak_start, peak_score, detected, candidate, detected, threshold, epsilon
    )
