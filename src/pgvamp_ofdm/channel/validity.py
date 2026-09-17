"""Half-open CP support at the actual FFT window, checked at both endpoints."""

import math

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings
from ..waveform.frame import FrameLayout
from .parameters import PathParameters


def validate_cp_support(
    paths: PathParameters,
    layout: FrameLayout,
    config: Config,
    *,
    window_offsets_s: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Return float64 [M,L,2] local endpoints; reject with block/path/endpoint details."""
    w, f = _physical_settings(config)
    paths.validate()
    if (
        layout.sample_rate_hz != w["sample_rate_hz"]
        or len(layout.useful) != f["n_ofdm_symbols"]
        or len(layout.cp) != len(layout.useful)
    ):
        raise ValueError("layout does not match physical config")
    offsets = window_offsets_s if window_offsets_s is not None else (0.0,) * len(layout.useful)
    if len(offsets) != len(layout.useful) or not all(math.isfinite(x) for x in offsets):
        raise ValueError("one finite window offset is required per block")
    fs, n = w["sample_rate_hz"], w["n_fft_wave"]
    results = []
    for m, ((start, end), cp, offset) in enumerate(zip(layout.useful, layout.cp, offsets)):
        expected_start = (
            f["leading_silence_samples"]
            + f["lfm_samples"]
            + f["sync_guard_samples"]
            + w["cp_samples"]
            + m * (n + w["cp_samples"])
        )
        if start != expected_start or end - start != n or cp != (start - w["cp_samples"], start):
            raise ValueError(f"layout block {m} has inconsistent CP/useful spans")
        tau = paths.delay_s - paths.epsilon * (start / fs)
        endpoints = paths.delay_s.new_tensor([offset, offset + (n - 1) / fs])
        xi = (1 + paths.epsilon[:, None]) * endpoints - tau[:, None]
        bad = (xi < -w["cp_samples"] / fs) | (xi >= n / fs)
        if bool(bad.any()):
            path, endpoint = bad.nonzero()[0].tolist()
            raise ValueError(
                f"CP support: block={m} path={path} endpoint={endpoint} "
                f"xi={float(xi[path, endpoint]):.17g}; "
                f"requires {-w['cp_samples'] / fs} <= xi < {n / fs}"
            )
        results.append(xi)
    return torch.stack(results)
