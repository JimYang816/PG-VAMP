"""Explicit frame-level paths; only sampled path energy is normalized."""

from dataclasses import dataclass

import torch

from ..config import Config
from ..modulation.allocation import _physical_settings

SCENARIOS = (
    "identity_awgn",
    "static_multipath",
    "affine_doppler_mild",
    "affine_doppler_moderate",
    "affine_doppler_strong",
)


@dataclass(frozen=True)
class PathParameters:
    """One frame: complex128 gain[L], float64 delay_s/epsilon[L], same device.

    Explicit paths need not have unit energy: propagation scale is preserved.
    """

    gain: torch.Tensor
    delay_s: torch.Tensor
    epsilon: torch.Tensor
    scenario: str

    def validate(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError("unsupported channel scenario")
        if not isinstance(self.gain, torch.Tensor) or self.gain.ndim != 1:
            raise ValueError("gain must be one-dimensional")
        if self.gain.dtype != torch.complex128 or self.gain.numel() == 0:
            raise ValueError("gain must be nonempty complex128")
        for name in ("delay_s", "epsilon"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != self.gain.shape
                or value.dtype != torch.float64
                or value.device != self.gain.device
            ):
                raise ValueError(f"{name} must be float64 with gain shape/device")
        if not all(bool(torch.isfinite(v).all()) for v in (self.gain, self.delay_s, self.epsilon)):
            raise ValueError("paths must be finite")
        if bool((self.delay_s < 0).any()) or bool((self.delay_s[1:] < self.delay_s[:-1]).any()):
            raise ValueError("delays must be nonnegative and sorted")
        if bool((self.epsilon <= -1).any()):
            raise ValueError("paths require 1+epsilon > 0")
        if self.scenario in ("identity_awgn", "static_multipath") and bool(self.epsilon.any()):
            raise ValueError("identity/static paths require zero epsilon")
        if self.scenario == "identity_awgn" and not (
            self.gain.numel() == 1 and self.gain.item() == 1 and self.delay_s.item() == 0
        ):
            raise ValueError("identity requires one unit gain and zero delay")


def sample_paths(config: Config, scenario: str, *, generator: torch.Generator) -> PathParameters:
    """Sample on generator.device in double precision; caller checks frame CP support."""
    _physical_settings(config)
    if scenario not in SCENARIOS:
        raise ValueError("unsupported channel scenario")
    device = generator.device
    if scenario == "identity_awgn":
        return PathParameters(
            torch.ones(1, dtype=torch.complex128, device=device),
            torch.zeros(1, dtype=torch.float64, device=device),
            torch.zeros(1, dtype=torch.float64, device=device),
            scenario,
        )
    c = config.values["channel"]
    count = int(
        torch.randint(c["min_paths"], c["max_paths"] + 1, (), generator=generator, device=device)
    )
    # 1-U gives (0,1], so the additional delays exclude the first-path endpoint.
    u = 1 - torch.rand(count - 1, generator=generator, dtype=torch.float64, device=device)
    delays = torch.cat(
        (
            u.new_tensor([c["first_delay_s"]]),
            c["first_delay_s"] + u.sort().values * (c["max_delay_s"] - c["first_delay_s"]),
        )
    )
    power = torch.exp(-(delays - c["first_delay_s"]) / c["power_delay_tau_s"])
    raw = torch.randn(count, 2, generator=generator, dtype=torch.float64, device=device)
    gain = torch.complex(raw[:, 0], raw[:, 1]) * torch.sqrt(power / 2)
    energy = gain.abs().square().sum()
    if not bool(torch.isfinite(energy)) or not bool(energy > 0):
        raise ValueError("sampled path energy is not positive finite")
    gain = gain / energy.sqrt()
    epsilon = torch.zeros_like(delays)
    if scenario != "static_multipath":
        epsilon = (
            2 * torch.rand(count, generator=generator, dtype=torch.float64, device=device) - 1
        ) * c["epsilon_max"][scenario]
    result = PathParameters(gain, delays, epsilon, scenario)
    result.validate()
    return result
