import math

import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector, PGVAMPDetector, VAMPDetector
from pgvamp_ofdm.reference.dense_vamp_cholesky import dense_vamp_cholesky


def system(n=8, batch=2, dtype=torch.complex128):
    gen = torch.Generator().manual_seed(852)
    H = torch.randn(batch, n, n, generator=gen, dtype=dtype) / math.sqrt(n)
    y = torch.randn(batch, n, generator=gen, dtype=dtype)
    return H, y, torch.full((batch,), 0.7, dtype=H.real.dtype)


class FullGraph(PGVAMPDetector):
    def _mask(self, H, rho):
        return torch.ones_like(H.real)


@pytest.mark.parametrize("depth", [1, 3, 8, 32])
def test_parameters_thresholds_graph(depth):
    model = PGVAMPDetector(depth)
    assert set(dict(model.named_parameters())) == {"raw_gaps", "raw_mu"}
    assert sum(p.numel() for p in model.parameters()) == 2 * depth
    assert sum(p.numel() for p in MMSEDetector().parameters()) == 0
    assert sum(p.numel() for p in VAMPDetector().parameters()) == 0
    rho, mu = model.thresholds()
    expected = -torch.arange(1, depth + 1, dtype=rho.dtype) * (
        0.5 + (60 - depth * 0.5) / (2 * depth)
    )
    torch.testing.assert_close(rho, expected)
    torch.testing.assert_close(mu, torch.full_like(mu, 0.8))
    assert ((rho[:-1] - rho[1:]) >= 0.5).all()
    assert ((rho > -60) & (rho < 0)).all()
    H, _, _ = system()
    H[:, :, 0] = 0
    masks = [model._mask(H, r) for r in rho]
    for a, b in zip(masks, masks[1:]):
        assert (a <= b).all()
    assert (masks[0].diagonal(dim1=-2, dim2=-1) == 1).all()
    assert (masks[0][:, 1:, 0] == 0).all()
    assert not torch.equal(masks[0], masks[0].mT)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_full_graph_exact_vamp(n):
    H, y, s = system(n)
    pg = FullGraph(3)(H, y, s, return_diagnostics=True)
    svd = VAMPDetector(3).detect(H, y, s, return_diagnostics=True)
    chol = dense_vamp_cholesky(H, y, s, iterations=3, return_diagnostics=True)
    for p, v, c in zip(
        pg.diagnostics["layers"], svd.diagnostics["layers"], chol.layers, strict=True
    ):
        for key in (
            "r2",
            "gamma2",
            "r1",
            "gamma1",
            "xhat2",
            "xhat1",
            "alpha1",
            "alpha2",
            "probabilities",
        ):
            torch.testing.assert_close(p[key], v[key], atol=1e-9, rtol=1e-8)
            torch.testing.assert_close(p[key], c[key], atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        FullGraph(3, init_mu=0.17)(H, y, s).x_soft, pg.x_soft, atol=1e-9, rtol=1e-8
    )


@pytest.mark.parametrize("kind", ["identity", "diagonal", "zero", "rank", "mixed"])
def test_limits_and_scale(kind):
    H, y, s = system()
    if kind in ("identity", "diagonal"):
        diag = (
            torch.ones(8, dtype=H.dtype)
            if kind == "identity"
            else torch.arange(1, 9, dtype=s.dtype) * (0.2 + 0.3j)
        )
        H = torch.diag(diag).expand_as(H)
    elif kind == "zero":
        H.zero_()
    else:
        H[:, :, 1] = H[:, :, 0]
        if kind == "mixed":
            H[0] = 0
    model = PGVAMPDetector(3)
    out = model(H, y, s)
    assert torch.isfinite(out.x_soft).all()
    if kind in ("zero", "mixed"):
        assert (out.x_soft[0] == 0).all()
        assert (out.probabilities[0] == 0.25).all()
    if kind in ("identity", "diagonal"):
        expected = y / diag
        assert torch.equal(out.class_hat, 2 * (expected.real < 0) + (expected.imag < 0))
    a = 2 + 0.7j
    torch.testing.assert_close(
        model(a * H, a * y, abs(a) ** 2 * s).x_soft, out.x_soft, atol=1e-9, rtol=1e-8
    )


@pytest.mark.parametrize("amplitude,informed", [(0, False), (1e-160, False), (1e-20, True)])
def test_weak_channel_and_zero_backward(amplitude, informed):
    H = amplitude * torch.eye(2, dtype=torch.complex128)[None]
    y = torch.ones(1, 2, dtype=H.dtype) * (0.2 + 0.1j)
    model = PGVAMPDetector(3)
    out = model(H, y, torch.ones(1, dtype=torch.float64))
    assert out.diagnostics["no_information"].item() == (0 if informed else 3)
    out.x_soft.abs().square().sum().backward()
    for p in model.parameters():
        assert torch.isfinite(p.grad).all()
        if not informed:
            assert (p.grad == 0).all()
    if not informed:
        assert (out.probabilities == 0.25).all()
    else:
        assert (out.x_soft.abs() > 0).all()


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
@pytest.mark.parametrize("amplitude", [25, 200, 1e6])
def test_saturation_backward(dtype, amplitude):
    H = torch.tensor([[[1, 0.1j], [0.1, 1]]], dtype=dtype, requires_grad=True)
    y = torch.full((1, 2), amplitude * (1 + 1j), dtype=dtype, requires_grad=True)
    s = torch.ones(1, dtype=H.real.dtype, requires_grad=True)
    model = PGVAMPDetector(3, dtype=s.dtype)
    out = model(H, y, s)
    assert (
        out.diagnostics["precision_capped"] + out.diagnostics["posterior_variance_underflow"]
    ).item() > 0
    out.x_soft.abs().square().sum().backward()
    for p in [H, y, s, *model.parameters()]:
        assert torch.isfinite(p.grad).all()


def test_factor_reuse_and_new_graph(monkeypatch):
    original = torch.linalg.cholesky_ex
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", counted)
    H, y, s = system()
    model = PGVAMPDetector(3)
    for _ in range(2):
        model(H, y, s).x_soft.abs().square().sum().backward()
        model.zero_grad()
    assert len(calls) == 6


def test_diagnostics_denominators_and_lightweight():
    H, y, s = system(n=1)
    H[0] = 0
    out = PGVAMPDetector(2)(H, y, s)
    assert "layers" not in out.diagnostics
    for layer in out.diagnostics["layer_summaries"]:
        assert (layer["candidate_edges"] == 0).all()
        assert (layer["all_off_diagonal_edges"] == 0).all()
        assert (layer["effective_candidate_ratio"] == 0).all()
        assert (layer["effective_all_ratio"] == 0).all()
        assert layer["safety_zero_baseline"][0]
        assert (layer["safety_relative"] == 0).all()


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_safety_diagnostic_is_scale_invariant(scale):
    H, y, s = system(n=3, batch=1)
    model = PGVAMPDetector(2)
    reference = model(H, y, s)
    actual = model(scale * H, scale * y, scale**2 * s)
    for a, b in zip(
        actual.diagnostics["layer_summaries"], reference.diagnostics["layer_summaries"], strict=True
    ):
        assert torch.isfinite(a["safety_relative"]).all()
        assert not a["safety_zero_baseline"].any()
        torch.testing.assert_close(a["safety_relative"], b["safety_relative"], atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": True},
        {"depth": 120},
        {"dtype": torch.float16},
        {"jitter": -1},
        {"temperature_db": 0},
        {"init_mu": 1},
        {"rho_hi_db": float("nan")},
        {"mask_mode": "hard"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PGVAMPDetector(**kwargs)


def test_parameter_validation_and_cuda_unavailable(monkeypatch):
    H, y, s = system()
    with pytest.raises(ValueError, match="dtype/device"):
        PGVAMPDetector(dtype=torch.float32)(H, y, s)
    model = PGVAMPDetector()
    with torch.no_grad():
        model.raw_mu[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(H, y, s)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        PGVAMPDetector(device="cuda")


def test_no_forbidden_apis():
    import ast
    from pathlib import Path

    for path in Path("src/pgvamp_ofdm/algorithms/pg_vamp").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert name not in {"inverse", "inv", "pinv", "cg", "nan_to_num", "randn", "rand"}
                if name == "detach":
                    assert path.name == "model.py"  # Only the small logging-copy comprehension.
            if isinstance(node, ast.ImportFrom):
                assert "reference" not in (node.module or "")
