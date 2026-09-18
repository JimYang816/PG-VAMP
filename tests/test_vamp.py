import ast
from pathlib import Path

import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.reference.dense_vamp_cholesky import dense_vamp_cholesky


def system(n=8, dtype=torch.complex128):
    gen = torch.Generator().manual_seed(197)
    H = torch.randn(3, n, n, dtype=dtype, generator=gen) / n**0.5
    H[1, :, n // 2 :] = 0
    H[2] = 0
    y = torch.randn(3, n, dtype=dtype, generator=gen)
    sigma = torch.tensor([0.7, 1.2, 0.5], dtype=H.real.dtype)
    return H, y, sigma


@pytest.mark.parametrize("n", [8, 16, 32])
@pytest.mark.parametrize("iterations", [8, 32])
def test_layerwise_cholesky_equivalence(n, iterations):
    H, y, sigma = system(n)
    actual = VAMPDetector(iterations).detect(H, y, sigma, return_diagnostics=True)
    expected = dense_vamp_cholesky(H, y, sigma, iterations=iterations, return_diagnostics=True)
    for left, right in zip(actual.diagnostics["layers"], expected.layers, strict=True):
        for key in (
            "r2",
            "gamma2",
            "xhat2",
            "alpha2",
            "c",
            "r1",
            "gamma1",
            "xhat1",
            "probabilities",
            "vbar",
            "alpha1",
        ):
            torch.testing.assert_close(left[key], right[key], atol=1e-9, rtol=1e-8)
        for key in ("rejected", "capped", "underflow"):
            if key in right:
                assert torch.equal(left[key], right[key])
    for key, value in expected.diagnostics.items():
        assert torch.equal(actual.diagnostics[key], value)
    torch.testing.assert_close(actual.x_soft, expected.x_soft, atol=1e-9, rtol=1e-8)
    assert actual.diagnostics["no_information"].tolist() == [0, 0, iterations]
    assert actual.x_soft[2].count_nonzero() == 0
    assert (actual.probabilities[2] == 0.25).all()


def test_weak_channel_mixed_batch():
    H = torch.tensor([0, 1e-160, 1e-20, 1.0], dtype=torch.float64)[:, None, None] * torch.eye(
        2, dtype=torch.complex128
    )
    y = torch.full((4, 2), 0.2 + 0.1j, dtype=H.dtype)
    sigma = torch.ones(4, dtype=torch.float64)
    actual = VAMPDetector().detect(H, y, sigma, return_diagnostics=True)
    oracle = dense_vamp_cholesky(H, y, sigma)
    assert actual.diagnostics["no_information"].tolist() == [8, 8, 0, 0]
    assert torch.equal(actual.diagnostics["no_information"], oracle.diagnostics["no_information"])
    assert actual.x_soft[:2].count_nonzero() == 0
    assert (actual.probabilities[:2] == 0.25).all()
    assert (actual.x_soft[2].abs() > 0).all()
    torch.testing.assert_close(actual.x_soft, oracle.x_soft, atol=0, rtol=1e-8)


def test_identity_matches_scalar_qpsk_enumeration():
    from pgvamp_ofdm.modulation.qpsk import classes_to_symbols

    H = torch.eye(4, dtype=torch.complex128)[None]
    y = torch.tensor([[0, 0.3 + 0.5j, -0.2j, -0.4 - 0.1j]], dtype=H.dtype)
    sigma = torch.tensor([0.6], dtype=torch.float64)
    alphabet = classes_to_symbols(torch.arange(4))
    probabilities = (-(y[..., None] - alphabet).abs().square() / sigma[:, None, None]).softmax(-1)
    output = VAMPDetector().detect(H, y, sigma)
    torch.testing.assert_close(output.probabilities, probabilities, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        output.x_soft, (probabilities * alphabet).sum(-1), atol=1e-9, rtol=1e-8
    )


@pytest.mark.parametrize("detector", [MMSEDetector, VAMPDetector])
def test_scale_invariance(detector):
    H, y, sigma = system()
    base = detector().detect(H, y, sigma)
    for a in (0.01, 10, 0.3 + 0.7j):
        scaled = detector().detect(H * a, y * a, sigma * abs(a) ** 2)
        torch.testing.assert_close(base.x_soft, scaled.x_soft, atol=1e-9, rtol=1e-8)
        assert torch.equal(base.class_hat, scaled.class_hat)


def test_single_svd_per_call_no_cache_parameters_or_last_extrinsic(monkeypatch):
    original = torch.linalg.svd
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", spy)
    model = VAMPDetector()
    assert sum(p.numel() for p in model.parameters()) == 0
    first = model.detect(*system(), return_diagnostics=True)
    second = model.detect(*system())
    assert calls == [{"full_matrices": False}] * 2
    assert "layers" not in second.diagnostics
    assert "rejected" not in first.diagnostics["layers"][-1]
    assert load_config("configs/vamp_reference_32.yaml").values["vamp"]["iterations"] == 32
    assert load_config("configs/cpu_dev.yaml").values["vamp"]["iterations"] == 8


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_iterations(value):
    with pytest.raises(ValueError):
        VAMPDetector(value)


def test_single_precision_oracle_and_backward():
    H, y, sigma = system(dtype=torch.complex64)
    actual = VAMPDetector().detect(H, y, sigma)
    oracle = dense_vamp_cholesky(H, y, sigma)
    # sigma>=0.5 and ||H|| around 2 bound linear-module conditioning in this fixture.
    torch.testing.assert_close(actual.x_soft, oracle.x_soft, atol=2e-6, rtol=2e-5)
    for dtype in (torch.complex64, torch.complex128):
        H = torch.tensor([[[1, 0.1j], [0.1, 1]]], dtype=dtype, requires_grad=True)
        amplitude = 200 if dtype == torch.complex128 else 25
        y = torch.full((1, 2), amplitude * (1 + 1j), dtype=dtype, requires_grad=True)
        sigma = torch.ones(1, dtype=H.real.dtype, requires_grad=True)
        output = VAMPDetector(3).detect(H, y, sigma)
        assert output.diagnostics["precision_capped"].item() == 2
        output.x_soft.abs().square().sum().backward()
        for tensor in (H, y, sigma):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_forbidden_calls_and_oracle_independence():
    for path in Path("src/pgvamp_ofdm/algorithms").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert name not in {"inv", "inverse", "pinv", "cg", "detach", "nan_to_num"}
            if isinstance(node, ast.ImportFrom):
                assert "reference" not in (node.module or "")
