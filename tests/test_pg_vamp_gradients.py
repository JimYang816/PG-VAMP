import pytest
import torch
from test_pg_vamp import system

from pgvamp_ofdm.algorithms import PGVAMPDetector
from pgvamp_ofdm.reference.dense_pg_vamp import DensePGVAMP


@pytest.mark.parametrize("n", [8, 16, 32])
def test_layerwise_oracle_and_both_parameter_gradients(n):
    H, y, s = system(n)
    model, oracle = PGVAMPDetector(3), DensePGVAMP(3)
    with torch.no_grad():
        model.raw_gaps.copy_(torch.tensor([-0.7, 0.3, -1.1]))
        model.raw_mu.copy_(torch.tensor([0.2, -0.4, 1.3]))
        oracle.load_state_dict(model.state_dict())
    actual = model(H, y, s, return_diagnostics=True)
    expected = oracle(H, y, s, return_diagnostics=True)
    for a, e in zip(actual.diagnostics["layers"], expected.layers, strict=True):
        for key in e:
            torch.testing.assert_close(a[key], e[key], atol=1e-9, rtol=1e-8)

    def loss(layers):
        return sum(
            (i + 1) * (st["xhat1"] - y).abs().square().mean()
            + 0.03 * st["probabilities"].square().mean()
            for i, st in enumerate(layers)
        )

    loss(actual.diagnostics["layers"]).backward()
    loss(expected.layers).backward()
    for a, e in zip(model.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(a.grad, e.grad, atol=1e-9, rtol=1e-8)
        assert torch.isfinite(a.grad).all()
        assert (a.grad.abs() > 0).all()


def test_raw_parameters_gradcheck():
    H, y, s = system(n=3, batch=1)
    model = PGVAMPDetector(2)

    def evaluate(gaps, mu):
        out = torch.func.functional_call(model, {"raw_gaps": gaps, "raw_mu": mu}, (H, y, s))
        return torch.view_as_real(out.x_soft)

    assert torch.autograd.gradcheck(
        evaluate, (model.raw_gaps, model.raw_mu), eps=1e-6, atol=2e-5, rtol=2e-4
    )


def test_single_precision_output_states_and_backward():
    H, y, s = system(dtype=torch.complex64)
    model = PGVAMPDetector(3, dtype=torch.float32)
    out = model(H, y, s, return_diagnostics=True)
    out.x_soft.abs().square().mean().backward()
    for p in model.parameters():
        assert torch.isfinite(p.grad).all()
        assert (p.grad.abs() > 0).all()
    for st in out.diagnostics["layers"]:
        for tensor in st.values():
            assert tensor.device == H.device
            assert tensor.dtype in (torch.complex64, torch.float32, torch.bool)
            assert torch.isfinite(tensor).all()
    double = PGVAMPDetector(3)(H.to(torch.complex128), y.to(torch.complex128), s.double())
    # O(N eps) dense accumulation in small well-conditioned solves.
    torch.testing.assert_close(out.x_soft.to(torch.complex128), double.x_soft, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype,amplitude", [(torch.complex128, 1e-140), (torch.complex64, 1e-17)])
def test_resolved_extremely_weak_nondiagonal_backward(dtype, amplitude):
    base = torch.tensor([[1, 0.2j], [0.3, 1]], dtype=dtype)
    H = torch.stack((amplitude * base, torch.zeros_like(base), base))
    y = torch.ones(3, 2, dtype=dtype)
    s = torch.ones(3, dtype=H.real.dtype)
    model = PGVAMPDetector(2, dtype=s.dtype)
    output = model(H, y, s)
    assert output.diagnostics["no_information"].tolist() == [0, 2, 0]
    assert (output.x_soft[0].abs() > 0).all()
    # Independent weak-channel first-order posterior: H^H y / sigma2.
    expected = base.mH @ y[0]
    torch.testing.assert_close(output.x_soft[0] / amplitude, expected, atol=2e-6, rtol=2e-5)
    output.x_soft.real.sum().backward()
    isolated = PGVAMPDetector(2, dtype=s.dtype)
    isolated(H[2:], y[2:], s[2:]).x_soft.real.sum().backward()
    for actual, reference in zip(model.parameters(), isolated.parameters(), strict=True):
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, reference.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_output_and_gradient_parity():
    H, y, s = system()
    cpu, gpu = PGVAMPDetector(3), PGVAMPDetector(3, device="cuda")
    a, b = cpu(H, y, s), gpu(H.cuda(), y.cuda(), s.cuda())
    torch.testing.assert_close(a.x_soft, b.x_soft.cpu(), atol=1e-9, rtol=1e-8)
    a.x_soft.abs().square().sum().backward()
    b.x_soft.abs().square().sum().backward()
    for p, q in zip(cpu.parameters(), gpu.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad.cpu(), atol=1e-9, rtol=1e-8)
