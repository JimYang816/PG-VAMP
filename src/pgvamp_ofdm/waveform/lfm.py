"""Discrete symmetric Tukey LFM, normalized by configuration rather than labels."""

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings, build_allocation
from ..utils.device import Runtime


def tukey_window(
    length: int,
    alpha: float = 0.1,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return real [length], symmetric u=n/(length-1); alpha=0 rectangle, 1 Hann."""
    if type(length) is not int or length < 2:
        raise ValueError("Tukey length must be an integer >= 2")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Tukey alpha must be finite in [0,1]")
    if dtype not in (torch.float64, torch.float32):
        raise ValueError("window dtype must be float64 or float32")
    u = torch.arange(length, dtype=torch.float64, device=device) / (length - 1)
    window = torch.ones_like(u)
    if alpha:
        left, right = u < alpha / 2, u > 1 - alpha / 2
        window[left] = 0.5 * (1 + torch.cos(math.pi * (2 * u[left] / alpha - 1)))
        window[right] = 0.5 * (1 + torch.cos(math.pi * (2 * u[right] / alpha - 2 / alpha + 1)))
    return window.to(dtype)


def _validate_runtime(config: Config, runtime: Runtime) -> None:
    settings = config.values["runtime"]
    requested = torch.device(settings["device"])
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", 0)
    expected = getattr(torch, settings["dtype"])
    real = torch.float64 if expected == torch.complex128 else torch.float32
    if (
        runtime.device != requested
        or runtime.complex_dtype != expected
        or runtime.real_dtype != real
    ):
        raise ValueError("runtime device/dtype must match the resolved configuration")


@dataclass(frozen=True)
class LFMTemplate:
    """Paired complex analytic/real RF [NL], real window and explicit normalization."""

    analytic: torch.Tensor
    real: torch.Tensor
    window: torch.Tensor
    normalization: float
    metadata: dict[str, Any]


def make_lfm(config: Config, runtime: Runtime) -> LFMTemplate:
    """Create [NL] on runtime device/dtype; evaluate time/phase in float64 then cast."""
    w, f = _physical_settings(config)
    _validate_runtime(config, runtime)
    build_allocation(config, runtime.device)
    length, fs = f["lfm_samples"], w["sample_rate_hz"]
    window = tukey_window(length, f["lfm_tukey_alpha"], device=runtime.device)
    window_power = float(window.square().mean())
    if not math.isfinite(window_power) or window_power <= 0:
        raise ValueError("LFM window must have positive finite mean energy")
    power = (w["n_data"] + w["n_pilots"]) / w["n_fft_wave"]
    amplitude = math.sqrt(power / window_power)
    t = torch.arange(length, dtype=torch.float64, device=runtime.device) / fs
    beta = (f["lfm_stop_hz"] - f["lfm_start_hz"]) / (length / fs)
    phase = 2 * math.pi * (f["lfm_start_hz"] * t + 0.5 * beta * t.square())
    analytic = (amplitude * window * torch.exp(1j * phase)).to(runtime.complex_dtype)
    return LFMTemplate(
        analytic,
        math.sqrt(2) * analytic.real,
        window.to(runtime.real_dtype),
        amplitude,
        {
            "window": "tukey",
            "symmetric": True,
            "length": length,
            "alpha": f["lfm_tukey_alpha"],
            "definition": "u=n/(N-1);v1",
            "time_phase_dtype": "float64",
            "output_dtype": str(analytic.dtype),
            "beta_hz_per_s": beta,
            "target_analytic_power": power,
        },
    )
