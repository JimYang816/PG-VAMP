import pytest
import torch

from pgvamp_ofdm.algorithms.messages import nonlinear_message
from pgvamp_ofdm.algorithms.qpsk import qpsk_posterior
from pgvamp_ofdm.modulation.qpsk import classes_to_symbols


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_posterior_against_four_point_enumeration(dtype):
    gen = torch.Generator().manual_seed(171)
    r = torch.randn(4, 16, dtype=dtype, generator=gen)
    gamma = torch.tensor([0, 0.2, 1.5, 20], dtype=r.real.dtype)
    alphabet = classes_to_symbols(torch.arange(4), dtype=dtype)
    probabilities = (-gamma[:, None, None] * (r[..., None] - alphabet).abs().square()).softmax(-1)
    mean = (probabilities * alphabet).sum(-1)
    variance = (probabilities * (alphabet - mean[..., None]).abs().square()).sum(-1)
    posterior = qpsk_posterior(r, gamma)
    tol = (
        {"atol": 1e-9, "rtol": 1e-8} if dtype == torch.complex128 else {"atol": 2e-6, "rtol": 2e-5}
    )
    torch.testing.assert_close(posterior.mean, mean, **tol)
    torch.testing.assert_close(posterior.probabilities, probabilities, **tol)
    torch.testing.assert_close(posterior.variance, variance, **tol)
    torch.testing.assert_close(posterior.alpha, gamma * variance.mean(-1), **tol)
    saturated = qpsk_posterior(torch.full_like(r, 1000 + 1000j), torch.ones_like(gamma))
    assert saturated.vbar.count_nonzero() == 0
    assert torch.isfinite(saturated.probabilities).all()


def test_true_alpha_is_not_clipped():
    posterior = qpsk_posterior(
        torch.zeros(1, 2, dtype=torch.complex128), torch.tensor([2.0], dtype=torch.float64)
    )
    assert posterior.alpha.item() == 2


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_message_protection_branches_and_mean_preserving_cap(dtype):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    # ordinary, negative precision, low precision, bad denom, zero variance,
    # nonfinite variance, reciprocal overflow, valid cap, nonfinite mean.
    vbar = torch.tensor(
        [0.5, 1, 1, 0.5, 0, float("nan"), torch.finfo(dtype).tiny / 8, 1e-12, 0.5], dtype=dtype
    )
    gamma1 = torch.tensor([1, 2, 1, 1, 1, 1, 1, 1, 1], dtype=dtype)
    alpha = torch.tensor([0.5, 0, 0, 1, 0, 0, 0, 0.2, 0], dtype=dtype)
    mean = torch.full((9, 2), 0.3 + 0.2j, dtype=complex_dtype)
    mean[-1, 0] = complex(float("nan"), 0)
    r1 = torch.full_like(mean, 0.1 - 0.1j)
    old_r = torch.full_like(mean, 0.7 + 0.1j)
    old_gamma = torch.ones(9, dtype=dtype)
    message = nonlinear_message(mean, r1, gamma1, vbar, alpha, old_r, old_gamma)
    assert message.rejected.tolist() == [False, True, True, True, True, True, True, False, True]
    assert message.capped.tolist() == [False] * 7 + [True, False]
    assert message.underflow[4]
    torch.testing.assert_close(message.r[7], (mean[7] - alpha[7] * r1[7]) / (1 - alpha[7]))
    assert message.gamma[7] == 1e8
    assert torch.equal(message.r[message.rejected], old_r[message.rejected])
    assert torch.equal(message.gamma[message.rejected], old_gamma[message.rejected])


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("case", ["cap", "zero", "reciprocal"])
def test_protected_message_backward(dtype, case):
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    v = {
        "cap": 1e-200 if dtype == torch.float64 else 1e-25,
        "zero": 0,
        "reciprocal": torch.finfo(dtype).tiny / 8,
    }[case]
    mean = torch.full((1, 2), 0.2 + 0.1j, dtype=cdtype, requires_grad=True)
    r1 = torch.full_like(mean, 0.1, requires_grad=True)
    gamma1 = torch.ones(1, dtype=dtype, requires_grad=True)
    vbar = torch.tensor([v], dtype=dtype, requires_grad=True)
    alpha = gamma1 * vbar
    old_r = torch.ones_like(mean, requires_grad=True)
    old_gamma = torch.ones_like(gamma1, requires_grad=True)
    message = nonlinear_message(mean, r1, gamma1, vbar, alpha, old_r, old_gamma)
    # Explicit zero dependencies allow checking unused/rejected inputs as zero gradients.
    loss = message.r.abs().square().sum() + message.gamma.sum()
    loss = loss + sum(t.abs().sum() * 0 for t in (mean, r1, gamma1, vbar))
    loss.backward()
    for tensor in (mean, r1, gamma1, vbar, old_r, old_gamma):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_posterior_gradcheck():
    r = torch.tensor([[0.3 + 0.1j, -0.2 + 0.4j]], dtype=torch.complex128, requires_grad=True)
    gamma = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, g: qpsk_posterior(x, g).mean, (r, gamma), atol=2e-5, rtol=2e-4
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("numerator_overflow", [False, True])
def test_finite_candidate_overflow_rejects_before_division_backward(dtype, numerator_overflow):
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    magnitude = torch.finfo(dtype).max * (0.9 if numerator_overflow else 0.55)
    mean = torch.full((1, 2), -magnitude, dtype=cdtype, requires_grad=True)
    r1 = torch.full_like(mean, magnitude, requires_grad=True)
    gamma = torch.ones(1, dtype=dtype, requires_grad=True)
    variance = torch.full_like(gamma, 0.5, requires_grad=True)
    previous = torch.ones_like(mean, requires_grad=True)
    previous_gamma = torch.ones_like(gamma, requires_grad=True)
    update = nonlinear_message(
        mean, r1, gamma, variance, gamma * variance, previous, previous_gamma
    )
    assert update.rejected.item()
    assert torch.equal(update.r, previous)
    (update.r.real.sum() + update.gamma.sum()).backward()
    for tensor in (mean, r1, gamma, variance, previous, previous_gamma):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
