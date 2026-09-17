"""The sole production QPSK mapping: class = 2*b_real + b_imag (§4.1)."""

import math

import torch


def bits_to_classes(bits: torch.Tensor) -> torch.Tensor:
    """Map nonempty bool/uint8/int64 bits [...,2] to int64 [...] on the same device."""
    if not isinstance(bits, torch.Tensor) or bits.ndim < 1 or bits.shape[-1] != 2:
        raise ValueError("bits must have shape [...,2]")
    if bits.numel() == 0 or bits.dtype not in (torch.bool, torch.uint8, torch.int64):
        raise ValueError("bits must be nonempty bool, uint8 or int64")
    if not bool(((bits == 0) | (bits == 1)).all()):
        raise ValueError("bits must contain only 0 and 1")
    return 2 * bits[..., 0].long() + bits[..., 1].long()


def classes_to_bits(classes: torch.Tensor) -> torch.Tensor:
    """Map nonempty int64 classes [...] in 0..3 to uint8 [...,2], R bit first."""
    if not isinstance(classes, torch.Tensor) or classes.dtype != torch.int64:
        raise ValueError("classes must be an int64 tensor")
    if classes.numel() == 0 or not bool(((classes >= 0) & (classes < 4)).all()):
        raise ValueError("classes must be nonempty and in 0..3")
    return torch.stack((classes // 2, classes % 2), dim=-1).to(torch.uint8)


def classes_to_symbols(
    classes: torch.Tensor, *, dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
    """Map int64 [...] to unit-energy complex128/64 [...], preserving device."""
    if dtype not in (torch.complex128, torch.complex64):
        raise ValueError("QPSK dtype must be complex128 or complex64")
    bits = classes_to_bits(classes)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    signs = 1 - 2 * bits.to(real_dtype)
    return torch.complex(signs[..., 0], signs[..., 1]) / math.sqrt(2)


def bits_to_symbols(bits: torch.Tensor, *, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Map bool/uint8/int64 [...,2] to complex128/64 [...], same device."""
    return classes_to_symbols(bits_to_classes(bits), dtype=dtype)


def hard_decision(symbols: torch.Tensor) -> torch.Tensor:
    """Return int64 classes for finite complex [...]; axis ties choose smallest class."""
    if not isinstance(symbols, torch.Tensor) or symbols.dtype not in (
        torch.complex128,
        torch.complex64,
    ):
        raise ValueError("symbols must be complex128 or complex64")
    if symbols.numel() == 0 or not bool(torch.isfinite(symbols).all()):
        raise ValueError("symbols must be nonempty and finite")
    return 2 * (symbols.real < 0).long() + (symbols.imag < 0).long()
