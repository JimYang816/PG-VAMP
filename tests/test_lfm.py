"""Independent Tukey/chirp and direct matched-correlation oracles."""

import math

import pytest
import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.receiver.synchronization import normalized_correlation
from pgvamp_ofdm.utils.device import resolve_runtime
from pgvamp_ofdm.waveform.lfm import make_lfm, tukey_window


@pytest.mark.parametrize("alpha", [0, 0.1, 1])
def test_tukey_window_independent_scalar_definition(alpha):
    length = 81

    def scalar(n):
        if alpha == 0:
            return 1.0
        u = n / (length - 1)
        if u < alpha / 2:
            return (1 - math.cos(2 * math.pi * u / alpha)) / 2
        if u > 1 - alpha / 2:
            return (1 - math.cos(2 * math.pi * (1 - u) / alpha)) / 2
        return 1.0

    expected = torch.tensor([scalar(n) for n in range(length)], dtype=torch.float64)
    actual = tukey_window(length, alpha)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual, actual.flip(0), atol=1e-14, rtol=1e-14)
    if alpha == 1:
        torch.testing.assert_close(
            actual, torch.hann_window(length, periodic=False, dtype=torch.float64)
        )


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_lfm_phase_power_length_and_fixed_normalization(dtype):
    config = load_config(dtype=dtype)
    lfm = make_lfm(config, resolve_runtime(dtype=dtype))
    assert lfm.analytic.shape == (3840,)
    assert (3840 - 1) / 96000 < 0.04
    torch.testing.assert_close(
        lfm.analytic.abs().square().mean(), lfm.analytic.real.new_tensor(464 / 8192)
    )
    phase_step = (lfm.analytic[1:] * lfm.analytic[:-1].conj()).angle().double()
    frequency = phase_step * 96000 / (2 * math.pi)
    n = torch.arange(3839, dtype=torch.float64)
    expected = 21000 + 150000 * (n + 0.5) / 96000
    valid = (lfm.window[1:] > 0) & (lfm.window[:-1] > 0)
    torch.testing.assert_close(
        frequency[valid], expected[valid], atol=0.004 if dtype == "complex64" else 1e-7, rtol=0
    )
    assert bool((frequency[valid][1:] > frequency[valid][:-1]).all())
    assert frequency[valid][-1] < 27000
    assert torch.equal(lfm.real, math.sqrt(2) * lfm.analytic.real)
    assert lfm.normalization == make_lfm(config, resolve_runtime(dtype=dtype)).normalization


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("offset", [0, 51, 5000])
def test_full_lfm_known_lag_including_last_valid(kind, offset):
    lfm = make_lfm(load_config(), resolve_runtime())
    template = lfm.real if kind == "real" else lfm.analytic
    recording = template.new_zeros((2, 5000 + 3840))
    recording[:, offset : offset + 3840] = template
    result = normalized_correlation(recording, template)
    assert result.peak_start.tolist() == [offset, offset]
    torch.testing.assert_close(
        result.peak_score, torch.ones(2, dtype=torch.float64), atol=1e-9, rtol=1e-8
    )
    assert result.detected.all() and result.candidate_valid.all()
    assert result.candidate_arrival.tolist() == [offset, offset]


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128, torch.float32, torch.complex64])
def test_direct_sliding_inner_product_and_scale(dtype):
    recording = torch.randn(
        2, 3, 37, dtype=dtype, generator=torch.Generator().manual_seed(33)
    ).transpose(0, 1)
    template = torch.randn(7, dtype=dtype, generator=torch.Generator().manual_seed(34))
    direct = []
    for d in range(31):
        segment = recording[..., d : d + 7]
        direct.append(
            (segment * template.conj()).sum(-1).abs().square()
            / (
                segment.abs().square().sum(-1) * template.abs().square().sum()
                + torch.finfo(template.real.dtype).tiny
            )
        )
    expected = torch.stack(direct, -1)
    actual = normalized_correlation(recording, template)
    tol = 2e-5 if template.real.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual.scores, expected, atol=tol, rtol=tol)
    scaled = normalized_correlation(recording * 2, template * 0.3)
    torch.testing.assert_close(scaled.scores, expected, atol=tol, rtol=tol)


def test_zero_windows_ties_and_threshold_failure():
    template = torch.tensor([1.0, -1.0, 0.3, 0.7], dtype=torch.float64)
    zero = normalized_correlation(torch.zeros((2, 50), dtype=torch.float64), template)
    assert torch.count_nonzero(zero.scores) == 0
    assert zero.peak_start.tolist() == [0, 0]
    assert not zero.detected.any()
    assert zero.candidate_arrival.tolist() == [-1, -1]
    recording = torch.zeros(1000, dtype=torch.float64)
    recording[40:44] = template
    result = normalized_correlation(recording, template)
    assert result.peak_start == 40
    assert torch.count_nonzero(result.scores[44:]) == 0
    rejected = normalized_correlation(torch.ones(40, dtype=torch.float64), template, threshold=0.99)
    assert not rejected.detected and rejected.candidate_arrival == -1


def test_noise_windows_and_probability_bound():
    lfm = make_lfm(load_config(), resolve_runtime())
    generator = torch.Generator().manual_seed(854)
    real = torch.randn(12, 6000, generator=generator, dtype=torch.float64)
    imag = torch.randn(12, 6000, generator=torch.Generator().manual_seed(855), dtype=torch.float64)
    complex_noise = torch.complex(real, imag) / math.sqrt(2)
    result = normalized_correlation(complex_noise, lfm.analytic)
    # Each single-window C ~ Beta(1,N-1); union bound needs no window independence.
    bound = 12 * (6000 - 3840 + 1) * math.exp((3840 - 1) * math.log1p(-0.1))
    assert bound < 1e-100
    assert not result.detected.any()
    real_result = normalized_correlation(real, lfm.real)
    assert not real_result.detected.any()  # fixed-seed fixture, not a universal false-alarm claim
    assert result.peak_score.max() > 0 and real_result.peak_score.max() > 0


@pytest.mark.parametrize(
    "recording,template",
    [
        (torch.ones(3, dtype=torch.float64), torch.ones(4, dtype=torch.float64)),
        (torch.ones(8), torch.ones(4, dtype=torch.float64)),
        (torch.ones(8), torch.zeros(4)),
        (torch.ones(8), torch.ones(2, 2)),
        (torch.ones(8), torch.full((4,), float("nan"))),
        (torch.full((8,), float("inf")), torch.ones(4)),
        (torch.full((8,), 1e30), torch.ones(4)),
        (torch.ones(8), torch.full((4,), 1e30)),
    ],
)
def test_invalid_correlation(recording, template):
    with pytest.raises(ValueError):
        normalized_correlation(recording, template)


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        normalized_correlation(torch.ones(8), torch.ones(4), threshold)


def test_invalid_window_and_zero_energy_lfm():
    for n, alpha in [(1, 0.1), (0, 0.1), (2.5, 0.1), (10, -0.1), (10, 1.1), (10, float("nan"))]:
        with pytest.raises(ValueError):
            tukey_window(n, alpha)
    config = load_config()
    config.values["frame"].update(lfm_samples=2, lfm_tukey_alpha=1)
    with pytest.raises(ValueError, match="energy"):
        make_lfm(config, resolve_runtime())


def test_cancellation_does_not_silently_hide_nonzero_windows():
    recording = torch.cat(
        (torch.ones(10, dtype=torch.float64) * 1e10, torch.ones(20, dtype=torch.float64))
    )
    with pytest.raises(ValueError, match="cancellation"):
        normalized_correlation(recording, torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("dtype,amplitude", [(torch.float64, 1e-100), (torch.float32, 1e-15)])
def test_positive_energy_product_underflow_is_rejected(dtype, amplitude):
    recording = torch.full((8,), amplitude, dtype=dtype)
    template = recording[:4].clone()
    assert bool((recording.square() > 0).all())
    assert recording[:4].square().sum() * template.square().sum() == 0
    with pytest.raises(ValueError, match="underflow"):
        normalized_correlation(recording, template)


@pytest.mark.parametrize("dtype,amplitude", [(torch.float64, 1e-170), (torch.float32, 1e-30)])
def test_partial_sample_energy_underflow_is_rejected(dtype, amplitude):
    recording = torch.ones(8, dtype=dtype)
    recording[0] = amplitude
    with pytest.raises(ValueError, match="underflow"):
        normalized_correlation(recording, torch.ones(4, dtype=dtype))


def test_small_representable_correlation_preserves_additive_epsilon():
    recording = torch.full((8,), 1e-78, dtype=torch.float64)
    template = recording[:4].clone()
    product = (4 * 1e-156) ** 2
    expected = product / (product + torch.finfo(torch.float64).tiny)
    actual = normalized_correlation(recording, template)
    torch.testing.assert_close(actual.scores, torch.full((5,), expected, dtype=torch.float64))
