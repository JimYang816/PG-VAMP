"""Sorted physical grid and exact default 400/64/47/1 allocation (§3)."""

from dataclasses import dataclass, fields
from typing import Any

import torch

from ..config import Config, ConfigError, _physical


def _physical_settings(config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(config, Config) or config.mode != "physical":
        raise ConfigError("physical waveform APIs do not accept algebra_fixture")
    _physical(config.values)
    w, f = config.values["waveform"], config.values["frame"]
    for key, expected in {
        "coding": "none",
        "fft_norm": "ortho",
        "ofdm_window": "none",
        "modulation": "qpsk_fixed_sign_bits",
        "symbol_energy": 1.0,
        "pilot_energy": 1.0,
    }.items():
        if w[key] != expected:
            raise ConfigError(f"waveform.{key} must be {expected!r}")
    return w, f


@dataclass(frozen=True)
class Allocation:
    """Int64 one-dimensional indices on one device; q and group q are ascending."""

    q: torch.Tensor
    grid_index: torch.Tensor
    baseband_fft_index: torch.Tensor
    passband_fft_index: torch.Tensor
    data_q: torch.Tensor
    pilot_q: torch.Tensor
    guard_q: torch.Tensor
    dc_q: torch.Tensor
    data_grid_index: torch.Tensor
    pilot_grid_index: torch.Tensor
    guard_grid_index: torch.Tensor
    dc_grid_index: torch.Tensor
    n_fft_wave: int
    carrier_bin: int

    def validate(self) -> None:
        """Reject corrupted indices, counts, ordering, overlap or nonpositive RF bins."""
        if type(self.n_fft_wave) is not int or type(self.carrier_bin) is not int:
            raise ValueError("allocation FFT length and carrier bin must be integers")
        if not isinstance(self.q, torch.Tensor):
            raise ValueError("allocation indices must be tensors")
        for field in fields(self):
            if field.name in ("n_fft_wave", "carrier_bin"):
                continue
            value = getattr(self, field.name)
            if not isinstance(value, torch.Tensor):
                raise ValueError("allocation indices must be tensors")
            if value.dtype != torch.int64 or value.ndim != 1 or value.device != self.q.device:
                raise ValueError("allocation indices must be 1D int64 on one device")
        q = torch.arange(-256, 256, device=self.q.device)
        if not torch.equal(self.q, q) or not torch.equal(self.grid_index, q + 256):
            raise ValueError("invalid physical grid ordering")
        if self.n_fft_wave < 512 or not torch.equal(self.baseband_fft_index, q % self.n_fft_wave):
            raise ValueError("invalid baseband FFT indices")
        if not torch.equal(self.passband_fft_index, self.carrier_bin + q):
            raise ValueError("invalid passband FFT indices")
        if not bool(
            ((self.passband_fft_index > 0) & (self.passband_fft_index < self.n_fft_wave / 2)).all()
        ):
            raise ValueError("passband grid must avoid DC, Nyquist and negative-frequency mirrors")
        groups = []
        for name, count in (("data", 400), ("pilot", 64), ("guard", 47), ("dc", 1)):
            group = getattr(self, name + "_q")
            positions = getattr(self, name + "_grid_index")
            if group.numel() != count or not bool((group[1:] > group[:-1]).all()):
                raise ValueError(f"invalid {name} count/order")
            if not torch.equal(positions, group + 256):
                raise ValueError(f"invalid {name} grid positions")
            groups.append(group)
        if not torch.equal(torch.cat(groups).sort().values, q):
            raise ValueError("allocation must cover the grid without overlap or duplicates")
        expected_guard = q[(q < -232) | (q > 232)]
        j = torch.arange(32, device=q.device, dtype=torch.float64)
        pos = 1 + torch.floor((j + 0.5) * 232 / 32).long()
        if not torch.equal(self.pilot_q, torch.cat((-pos.flip(0), pos))):
            raise ValueError("pilot positions must follow the fixed allocation")
        if not torch.equal(self.guard_q, expected_guard) or not torch.equal(self.dc_q, q[q == 0]):
            raise ValueError("invalid guard/DC positions")


def build_allocation(config: Config, device: torch.device | str = "cpu") -> Allocation:
    """Build exact physical indices on the explicitly selected device; no CUDA probing."""
    w, _ = _physical_settings(config)
    q = torch.arange(-256, 256, dtype=torch.int64, device=device)
    j = torch.arange(32, dtype=torch.float64, device=device)
    pos = 1 + torch.floor((j + 0.5) * 232 / 32).long()
    pilot = torch.cat((-pos.flip(0), pos))
    guard = q[(q < -232) | (q > 232)]
    dc = q[q == 0]
    data = q[(q >= -232) & (q <= 232) & (q != 0) & ~torch.isin(q, pilot)]
    spacing = w["sample_rate_hz"] / w["n_fft_wave"]
    carrier = round(w["carrier_hz"] / spacing)
    result = Allocation(
        q,
        q + 256,
        q % w["n_fft_wave"],
        carrier + q,
        data,
        pilot,
        guard,
        dc,
        data + 256,
        pilot + 256,
        guard + 256,
        dc + 256,
        w["n_fft_wave"],
        carrier,
    )
    result.validate()
    return result


def _validate_symbols(symbols: torch.Tensor, name: str) -> None:
    if not isinstance(symbols, torch.Tensor) or symbols.dtype not in (
        torch.complex128,
        torch.complex64,
    ):
        raise ValueError(f"{name} must be complex128/complex64")
    if symbols.numel() == 0 or not bool(torch.isfinite(symbols).all()):
        raise ValueError(f"{name} must be nonempty and finite")
    tol = 2e-6 if symbols.dtype == torch.complex64 else 1e-12
    target = 2**-0.5
    if not bool(((symbols.real.abs() - target).abs() <= tol).all()) or not bool(
        ((symbols.imag.abs() - target).abs() <= tol).all()
    ):
        raise ValueError(f"{name} must contain fixed unit-energy QPSK symbols")


def map_grid(data: torch.Tensor, pilots: torch.Tensor, allocation: Allocation) -> torch.Tensor:
    """Map same dtype/device QPSK [...,400] and [...,64] to complex [...,512]."""
    allocation.validate()
    _validate_symbols(data, "data")
    _validate_symbols(pilots, "pilots")
    if data.ndim < 1 or data.shape[-1] != 400 or pilots.shape != (*data.shape[:-1], 64):
        raise ValueError("data/pilots must have matching [...,400]/[...,64] shapes")
    if (
        data.dtype != pilots.dtype
        or data.device != pilots.device
        or data.device != allocation.q.device
    ):
        raise ValueError("data, pilots and allocation must share dtype/device as applicable")
    grid = data.new_zeros((*data.shape[:-1], 512))
    grid[..., allocation.data_grid_index] = data
    grid[..., allocation.pilot_grid_index] = pilots
    return grid
