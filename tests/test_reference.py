import math

import pytest
import torch

from pgvamp_ofdm.reference.dense_pg_vamp import DensePGVAMP, _extrinsic, _posterior
from pgvamp_ofdm.reference.dense_vamp_cholesky import dense_vamp_cholesky


def fixture(n=4, batch=2, dtype=torch.complex128):
    generator = torch.Generator().manual_seed(917)
    H = torch.randn(batch, n, n, generator=generator, dtype=dtype) / math.sqrt(n)
    y = torch.randn(batch, n, generator=generator, dtype=dtype)
    s = torch.full((batch,), 0.7, dtype=H.real.dtype)
    return H, y, s


class FullGraph(DensePGVAMP):
    """Only this test fixture forces a full graph; no runtime mask option exists."""

    def _mask(self, H, rho):
        return torch.ones_like(H.real)


def test_full_graph_multilayer_limit():
    H, y, s = fixture()
    pg = FullGraph(3)(H, y, s, return_diagnostics=True)
    vamp = dense_vamp_cholesky(H, y, s, iterations=3, return_diagnostics=True)
    for p, v in zip(pg.layers, vamp.layers, strict=True):
        for key in ("xhat1", "xhat2", "r1", "gamma1", "r2", "gamma2", "alpha1", "alpha2"):
            torch.testing.assert_close(p[key], v[key], atol=1e-9, rtol=1e-8)
    model = FullGraph(3, init_mu=0.2)
    torch.testing.assert_close(model(H, y, s).x_soft, pg.x_soft, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("kind", ["identity", "diagonal", "zero", "rank_deficient", "mixed"])
def test_boundary_systems(kind):
    H, y, s = fixture()
    if kind == "identity":
        H = torch.eye(4, dtype=H.dtype).expand(2, 4, 4)
    elif kind == "diagonal":
        diagonal = torch.tensor([1, 1j, 0.5, -2j], dtype=H.dtype)
        H = torch.diag(diagonal).expand(2, 4, 4)
    elif kind == "zero":
        H = torch.zeros_like(H)
    elif kind == "rank_deficient":
        H[:, :, 1] = H[:, :, 0]
    else:
        H[0] = 0
        H[1, :, 1] = H[1, :, 0]
    for result in (
        DensePGVAMP(3)(H, y, s, return_diagnostics=True),
        dense_vamp_cholesky(H, y, s, iterations=3, return_diagnostics=True),
    ):
        assert result.x_soft.shape == (2, 4)
        assert result.probabilities.shape == (2, 4, 4)
        assert bool(torch.isfinite(result.x_soft).all())
        torch.testing.assert_close(result.probabilities.sum(-1), torch.ones_like(y.real))
        if kind in ("zero", "mixed"):
            assert torch.count_nonzero(result.x_soft[0]) == 0
            assert bool((result.probabilities[0] == 0.25).all())
            assert result.diagnostics["no_information"][0] == 3
        if kind == "identity":
            expected = torch.complex(
                (math.sqrt(2) * y.real / s[:, None]).tanh(),
                (math.sqrt(2) * y.imag / s[:, None]).tanh(),
            ) / math.sqrt(2)
            torch.testing.assert_close(result.x_soft, expected, atol=1e-9, rtol=1e-8)
        if kind == "diagonal":
            expected_class = 2 * ((y / H.diagonal(dim1=-2, dim2=-1)).real < 0) + (
                (y / H.diagonal(dim1=-2, dim2=-1)).imag < 0
            )
            assert torch.equal(result.class_hat, expected_class)


def test_qpsk_against_four_point_enumeration():
    _, r, gamma = fixture(n=8)
    gamma = gamma * 2
    x, p, v, alpha = _posterior(r, gamma)
    symbols = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=r.dtype) / math.sqrt(2)
    expected = (-gamma[:, None, None] * (r[..., None] - symbols).abs().square()).softmax(-1)
    mean = (expected * symbols).sum(-1)
    variance = (expected * (symbols - mean[..., None]).abs().square()).sum(-1).mean(-1)
    for actual, wanted in ((p, expected), (x, mean), (v, variance), (alpha, gamma * variance)):
        torch.testing.assert_close(actual, wanted, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_parameters_gradients_and_dtype(dtype):
    H, y, s = fixture(dtype=dtype)
    model = DensePGVAMP(3, dtype=s.dtype)
    assert sum(p.numel() for p in model.parameters()) == 6
    assert set(dict(model.named_parameters())) == {"raw_gaps", "raw_mu"}
    rho, mu = model.thresholds()
    assert bool(((rho[:-1] - rho[1:]) >= 0.5).all())
    assert bool((rho > -60).all() & (rho < 0).all())
    torch.testing.assert_close(mu, torch.full_like(mu, 0.8))
    result = model(H, y, s, return_diagnostics=True)
    result.x_soft.abs().square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad.abs() > 0).all())
    for state in result.layers:
        for tensor in state.values():
            assert tensor.device == H.device
            assert tensor.dtype in (H.dtype, s.dtype, torch.bool)
            assert bool(torch.isfinite(tensor).all())
    vamp = dense_vamp_cholesky(H, y, s, iterations=3)
    assert bool(torch.isfinite(vamp.x_soft).all())
    if dtype == torch.complex64:
        double = DensePGVAMP(3)(H.to(torch.complex128), y.to(torch.complex128), s.double())
        # Single precision: O(N*eps) accumulation across small, well-conditioned solves.
        torch.testing.assert_close(
            result.x_soft.to(torch.complex128), double.x_soft, atol=2e-5, rtol=2e-4
        )


def test_raw_parameter_gradcheck():
    H, y, s = fixture(n=3, batch=1)
    model = DensePGVAMP(2)

    def evaluate(gaps, mu):
        result = torch.func.functional_call(model, {"raw_gaps": gaps, "raw_mu": mu}, (H, y, s))
        return torch.view_as_real(result.x_soft)

    assert torch.autograd.gradcheck(
        evaluate, (model.raw_gaps, model.raw_mu), eps=1e-6, atol=2e-5, rtol=2e-4
    )


def test_energy_majorizer_and_scale():
    H, y, s = fixture()
    model = DensePGVAMP(3)
    original = model(H, y, s, return_diagnostics=True)
    for state in original.layers:
        torch.testing.assert_close(
            state["G"].diagonal(dim1=-2, dim2=-1),
            (H.mH @ H).diagonal(dim1=-2, dim2=-1),
            atol=1e-9,
            rtol=1e-8,
        )
        assert torch.linalg.eigvalsh(state["Gbar"] - H.mH @ H).min() >= -1e-9
    a = 2 + 0.7j
    scaled = model(a * H, a * y, abs(a) ** 2 * s)
    torch.testing.assert_close(scaled.x_soft, original.x_soft, atol=1e-9, rtol=1e-8)


def test_message_protection_underflow_reject_and_cap():
    old_r = torch.full((4, 2), 0.2 + 0.1j, dtype=torch.complex128)
    old_g = torch.ones(4, dtype=torch.float64)
    r1 = torch.ones_like(old_r)
    gamma = torch.tensor([2, 1, 1, 1], dtype=torch.float64)
    vbar = torch.tensor([1.0, 0.0, 1e-10, 0.5], dtype=torch.float64)
    alpha = torch.tensor([2.0, 0.0, 1e-10, 0.5], dtype=torch.float64)
    xhat = torch.full_like(old_r, 0.5)
    r, g, rejected, capped, underflow = _extrinsic(old_r, old_g, r1, gamma, xhat, vbar, alpha)
    assert rejected.tolist() == [True, True, False, False]
    assert underflow.tolist() == [False, True, False, False]
    assert capped.tolist() == [False, False, True, False]
    torch.testing.assert_close(r[:2], old_r[:2])
    assert g[2] == 1e8
    torch.testing.assert_close(r[2], (xhat[2] - alpha[2] * r1[2]) / (1 - alpha[2]))


def test_underflow_in_full_loops():
    H = torch.eye(2, dtype=torch.complex128)[None]
    y = torch.full((1, 2), 1 + 1j, dtype=torch.complex128)
    s = torch.tensor([1e-6], dtype=torch.float64)
    for result in (DensePGVAMP(2)(H, y, s), dense_vamp_cholesky(H, y, s, iterations=2)):
        assert result.diagnostics["posterior_variance_underflow"].item() == 1
        assert result.diagnostics["message_rejected"].item() == 1
        assert bool(torch.isfinite(result.x_soft).all())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("at_boundary", [False, True])
def test_extreme_variance_protection_has_finite_gradients(dtype, at_boundary):
    # First row cannot represent 1/v; second can, but its reciprocal derivative
    # overflows. Both must avoid evaluating that derivative on the skipped path.
    tiny = 1e-310 if dtype == torch.float64 else 1e-40
    if at_boundary:
        tiny = 1 / torch.finfo(dtype).max
    capped_v = 1e-220 if dtype == torch.float64 else 1e-25
    vbar = torch.tensor([tiny, capped_v, 0.5], dtype=dtype, requires_grad=True)
    gamma = torch.ones(3, dtype=dtype)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    old_r = torch.full((3, 2), 0.2 + 0.1j, dtype=cdtype)
    r1 = torch.ones_like(old_r)
    xhat = torch.full_like(old_r, 0.5)
    alpha = gamma * vbar
    r, g, rejected, capped, _ = _extrinsic(old_r, gamma, r1, gamma, xhat, vbar, alpha)
    assert rejected.tolist() == [True, False, False]
    assert capped.tolist() == [False, True, False]
    torch.testing.assert_close(r[0], old_r[0])
    torch.testing.assert_close(r[1], (xhat[1] - alpha[1] * r1[1]) / (1 - alpha[1]))
    assert g[1] == 1e8
    (r.abs().square().sum() + g.sum()).backward()
    assert bool(torch.isfinite(vbar.grad).all())
    assert vbar.grad[0] == 0


@pytest.mark.parametrize("algorithm", ["pg", "vamp"])
@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_full_loop_precision_cap_backward(algorithm, dtype):
    H = torch.tensor([[[1, 0.1j], [0.1, 1]]], dtype=dtype, requires_grad=True)
    amplitude = 200 if dtype == torch.complex128 else 25
    y = torch.full((1, 2), amplitude * (1 + 1j), dtype=dtype, requires_grad=True)
    s = torch.ones(1, dtype=H.real.dtype, requires_grad=True)
    model = DensePGVAMP(3, dtype=s.dtype)
    result = model(H, y, s) if algorithm == "pg" else dense_vamp_cholesky(H, y, s, iterations=3)
    assert result.diagnostics["precision_capped"].item() == 2
    result.x_soft.abs().square().sum().backward()
    for tensor in [H, y, s] + (list(model.parameters()) if algorithm == "pg" else []):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())


@pytest.mark.parametrize("amplitude,informed", [(1e-20, True), (1e-160, False), (0, False)])
def test_no_information_boundary(amplitude, informed):
    H = amplitude * torch.eye(2, dtype=torch.complex128)[None]
    y = torch.full((1, 2), 0.2 + 0.1j, dtype=torch.complex128)
    s = torch.ones(1, dtype=torch.float64)
    for result in (
        DensePGVAMP(1)(H, y, s, return_diagnostics=True),
        dense_vamp_cholesky(H, y, s, iterations=1, return_diagnostics=True),
    ):
        assert result.diagnostics["no_information"].item() == int(not informed)
        assert bool(torch.isfinite(result.x_soft).all())
        if informed:
            assert bool((result.x_soft.abs() > 0).all())


@pytest.mark.parametrize("amplitude", [0, 1e-160])
def test_all_uninformative_batch_has_zero_parameter_gradients(amplitude):
    H = amplitude * torch.eye(2, dtype=torch.complex128)[None]
    y = torch.ones(1, 2, dtype=torch.complex128)
    model = DensePGVAMP(3)
    result = model(H, y, torch.ones(1, dtype=torch.float64))
    assert torch.count_nonzero(result.x_soft) == 0
    assert bool((result.probabilities == 0.25).all())
    (result.x_soft - y).abs().square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("problem", ["shape", "dtype", "noise", "finite", "empty"])
def test_input_validation(problem):
    H, y, s = fixture()
    if problem == "shape":
        H = H[:, :, :3]
    elif problem == "dtype":
        s = s.float()
    elif problem == "noise":
        s[0] = 0
    elif problem == "finite":
        y[0, 0] = float("nan")
    else:
        H, y, s = H[:0], y[:0], s[:0]
    with pytest.raises(ValueError):
        DensePGVAMP(2)(H, y, s)
    with pytest.raises(ValueError):
        dense_vamp_cholesky(H, y, s)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_algebra_fixture_dimensions_and_exact_linear_module(n):
    H, y, s = fixture(n=n, batch=1)
    vamp = dense_vamp_cholesky(H, y, s, iterations=2, return_diagnostics=True)
    pg = DensePGVAMP(2)(H, y, s)
    assert bool(torch.isfinite(pg.x_soft).all())
    assert pg.x_soft.shape == (1, n)
    for layer in vamp.layers:
        rhs = (H.mH @ y[..., None]).squeeze(-1) / s[:, None]
        rhs = rhs + layer["gamma2"][:, None] * layer["r2"]
        expected = torch.linalg.solve(layer["A"], rhs[..., None]).squeeze(-1)
        torch.testing.assert_close(layer["xhat2"], expected, atol=1e-9, rtol=1e-8)


def test_no_forbidden_reference_calls():
    import ast
    from pathlib import Path

    forbidden = {"inverse", "inv", "pinv", "detach", "nan_to_num", "cg"}
    for path in Path("src/pgvamp_ofdm/reference").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert name not in forbidden, (path, name)
            if isinstance(node, ast.ImportFrom):
                assert "algorithms" not in (node.module or "")
