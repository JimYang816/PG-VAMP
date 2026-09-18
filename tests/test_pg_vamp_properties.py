import pytest
import torch
from test_pg_vamp import system

from pgvamp_ofdm.algorithms import PGVAMPDetector
from pgvamp_ofdm.algorithms.pg_vamp.linear import linear_layer
from pgvamp_ofdm.algorithms.pg_vamp.majorizer import majorizer


def test_independent_safety_definition_energy_and_loewner():
    H, _, _ = system(n=8)
    model = PGVAMPDetector(4)
    rho, _ = model.thresholds()
    previous = None
    for r in rho:
        M = model._mask(H, r)
        state = majorizer(H, M)
        expected = torch.zeros_like(state["ell"])
        for i in range(8):
            for k in range(8):
                for j in range(8):
                    if j != i:
                        expected[:, i] += (
                            (1 - M[:, k, i] * M[:, k, j]) * H[:, k, i].abs() * H[:, k, j].abs()
                        )
        torch.testing.assert_close(state["ell"], expected, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(
            state["G"].diagonal(dim1=-2, dim2=-1),
            (H.mH @ H).diagonal(dim1=-2, dim2=-1),
            atol=1e-9,
            rtol=1e-8,
        )
        assert torch.linalg.eigvalsh(state["Gbar"] - H.mH @ H).min() >= -1e-9
        if previous is not None:
            assert torch.linalg.eigvalsh(previous - state["Gbar"]).min() >= -1e-9
        previous = state["Gbar"]


@pytest.mark.parametrize("jitter", [0, 0.125])
def test_operator_objective_covariance_and_jacobian(jitter):
    H, y, s = system(n=4, batch=1)
    model = PGVAMPDetector(2)
    rho, mu = model.thresholds()
    mask = model._mask(H, rho[0]).detach()
    mu = mu[0].detach()
    r2 = y * (0.2 + 0.3j)
    gamma = torch.tensor([1.3], dtype=s.dtype)
    state = linear_layer(H, y, s, r2, gamma, mask, mu, jitter=jitter)
    eye = torch.eye(4, dtype=H.dtype)[None]
    A = H.mH @ H / s[:, None, None] + gamma[:, None, None] * eye
    Pinv = torch.linalg.solve(state["P"], eye)
    B = (1 + mu) * Pinv - mu * Pinv @ A @ Pinv
    torch.testing.assert_close(B, B.mH, atol=1e-9, rtol=1e-8)
    assert torch.linalg.eigvalsh(B).min() > 0
    assert torch.linalg.eigvalsh(torch.linalg.solve(A, eye) - B).min() >= -1e-9
    u = H.mH @ (y - (H @ r2[..., None]).squeeze(-1))[..., None] / s[:, None, None]
    torch.testing.assert_close(state["innovation"], (B @ u).squeeze(-1), atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(state["W"], B @ H.mH / s[:, None, None], atol=1e-9, rtol=1e-8)

    def objective(x):
        return (y - (H @ x[..., None]).squeeze(-1)).abs().square().sum(-1) / s + gamma * (
            x - r2
        ).abs().square().sum(-1)

    assert (objective(state["xhat2"]) <= objective(r2) + 1e-9).all()
    E = eye - state["K"] @ H
    covariance = E @ E.mH / gamma[:, None, None] + s[:, None, None] * state["K"] @ state["K"].mH
    variance = covariance.diagonal(dim1=-2, dim2=-1).real.mean(-1)
    torch.testing.assert_close(1 / state["gamma1"], variance, atol=1e-9, rtol=1e-8)

    def real_map(real_r):
        state = linear_layer(H, y, s, torch.view_as_complex(real_r), gamma, mask, mu, jitter=jitter)
        return torch.view_as_real(state["xhat2"])

    J = torch.autograd.functional.jacobian(real_map, torch.view_as_real(r2)).reshape(8, 8)
    torch.testing.assert_close(J.trace() / 8, state["alpha2"][0], atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        state["P"], state["Gbar"] / s[:, None, None] + (gamma + jitter)[:, None, None] * eye
    )


def test_exclusive_sums_preserve_tiny_neighbors():
    H = torch.tensor([[[1e10, 1e-12], [1e-12, 1e10]]], dtype=torch.complex128)
    mask = torch.eye(2, dtype=torch.float64)[None]
    state = majorizer(H, mask)
    torch.testing.assert_close(state["ell"], torch.full((1, 2), 0.02, dtype=torch.float64))


def test_trace_bound_non_diagonal_cancellation_and_rank():
    H, y, s = system(n=8, batch=2)
    H[0, :, 1] = H[0, :, 0]
    # Phase-rich, rank-deficient systems audit contraction cancellation.
    out = PGVAMPDetector(3)(H, y, s, return_diagnostics=True)
    eps = torch.finfo(s.dtype).eps
    for st in out.diagnostics["layers"]:
        products = st["W"] * H.mT
        contracted = products.sum((-2, -1)).real / 8
        torch.testing.assert_close(contracted, st["c"], atol=1e-9, rtol=1e-8)
        bound = 32 * eps / (1 - 32 * eps) * products.abs().sum((-2, -1)) / 8
        torch.testing.assert_close(st["c_tolerance"], bound, atol=1e-30, rtol=1e-12)
        assert (st["c"] > st["c_tolerance"]).all()


def test_hard_failure_before_no_information(monkeypatch):
    H, y, s = system(n=2)
    model = PGVAMPDetector(1)
    monkeypatch.setattr(
        torch.linalg, "cholesky_ex", lambda P: (P, torch.ones(P.shape[0], dtype=torch.int32))
    )
    with pytest.raises(
        FloatingPointError, match="Cholesky failed.*layer=0.*batch index=.*dtype=.*device="
    ):
        model(H, y, s)
