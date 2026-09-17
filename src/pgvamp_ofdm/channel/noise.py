"""Unit-symbol Es/N0 and independent real/imaginary time-I/Q AWGN streams."""

import math

import torch


def sigma2_from_esn0(esn0_db: float) -> float:
    """Complex noise variance for Es=1; never use measured received power."""
    if not math.isfinite(esn0_db):
        raise ValueError("Es/N0 must be finite")
    try:
        variance = 10.0 ** (-esn0_db / 10)
    except OverflowError as exc:
        raise ValueError("noise variance must be representable") from exc
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("noise variance must be positive finite")
    return variance


def complex_awgn(
    shape: tuple[int, ...],
    sigma2: float,
    *,
    gen_r: torch.Generator,
    gen_i: torch.Generator,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """CN(0,sigma2) on generator device; requires separate generator objects."""
    if (
        not shape
        or any(type(n) is not int or n <= 0 for n in shape)
        or not math.isfinite(sigma2)
        or sigma2 <= 0
    ):
        raise ValueError("shape and variance must be positive finite")
    if (
        gen_r is gen_i
        or gen_r.device != gen_i.device
        or torch.equal(gen_r.get_state(), gen_i.get_state())
    ):
        raise ValueError("independent real/imaginary generators on one device required")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("noise dtype must be complex64 or complex128")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    scale = math.sqrt(sigma2) / math.sqrt(2)
    if not 0 < float(torch.tensor(scale, dtype=real_dtype)) < math.inf:
        raise ValueError("noise scale is unrepresentable in requested dtype")
    real = torch.randn(shape, generator=gen_r, dtype=real_dtype, device=gen_r.device)
    imag = torch.randn(shape, generator=gen_i, dtype=real_dtype, device=gen_i.device)
    result = torch.complex(real * scale, imag * scale)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("noise realization is unrepresentable in requested dtype")
    return result
