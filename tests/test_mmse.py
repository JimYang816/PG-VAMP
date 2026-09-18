import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_mmse_solve_residual_and_raw_output(dtype):
    gen = torch.Generator().manual_seed(491)
    H = torch.randn(3, 8, 8, dtype=dtype, generator=gen) / 8**0.5
    H[1, :, 4:] = 0  # rank-deficient system remains positive definite after noise.
    H[2] = 0
    y = torch.randn(3, 8, dtype=dtype, generator=gen)
    sigma = torch.tensor([0.3, 0.7, 2], dtype=H.real.dtype)
    before = H.clone(), y.clone(), sigma.clone()
    result = MMSEDetector().detect(H, y, sigma)
    A = H.mH @ H + sigma[:, None, None] * torch.eye(8, dtype=dtype)
    rhs = (H.mH @ y[..., None]).squeeze(-1)
    expected = torch.linalg.solve(A, rhs)
    tol = (
        {"atol": 1e-9, "rtol": 1e-8} if dtype == torch.complex128 else {"atol": 2e-6, "rtol": 2e-5}
    )
    torch.testing.assert_close(result.x_soft, expected, **tol)
    torch.testing.assert_close((A @ result.x_soft[..., None]).squeeze(-1), rhs, **tol)
    assert result.probabilities is None
    assert result.class_hat.dtype == torch.int64 and result.bits_hat.dtype == torch.uint8
    assert result.bits_hat.shape == (3, 8, 2)
    assert result.class_hat[2].count_nonzero() == 0
    assert sum(p.numel() for p in MMSEDetector().parameters()) == 0
    for original, actual in zip(before, (H, y, sigma), strict=True):
        assert torch.equal(original, actual)


@pytest.mark.parametrize("diagonal", [[1, 1, 1], [2, 0, 0.5]])
def test_mmse_diagonal_closed_form(diagonal):
    d = torch.tensor(diagonal, dtype=torch.complex128)
    y = torch.tensor([[2 + 1j, -1j, -3]], dtype=d.dtype)
    result = MMSEDetector().detect(d.diag()[None], y, torch.tensor([0.4], dtype=torch.float64))
    torch.testing.assert_close(result.x_soft, d.conj() * y / (d.abs().square() + 0.4))
