import os

import pytest
import torch

from pgvamp_ofdm.cli import _provenance
from pgvamp_ofdm.reference.dense_pg_vamp import DensePGVAMP
from pgvamp_ofdm.reference.dense_vamp_cholesky import dense_vamp_cholesky
from pgvamp_ofdm.utils.device import resolve_runtime


def test_cpu_never_accesses_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU called CUDA")

    for name in (
        "is_available",
        "device_count",
        "_lazy_init",
        "manual_seed",
        "manual_seed_all",
        "synchronize",
        "current_device",
        "get_device_name",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    runtime = resolve_runtime()
    assert runtime.device.type == "cpu"
    assert runtime.cpu_threads == min(4, os.cpu_count() or 1)
    assert (runtime.complex_dtype, runtime.real_dtype) == (torch.complex128, torch.float64)
    _provenance(runtime)
    H = torch.eye(3, dtype=runtime.complex_dtype)[None]
    y = torch.ones(1, 3, dtype=runtime.complex_dtype)
    s = torch.ones(1, dtype=runtime.real_dtype)
    DensePGVAMP(2)(H, y, s)
    dense_vamp_cholesky(H, y, s, iterations=2)


def test_cpu_stays_default_with_gpu_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_runtime().device.type == "cpu"


def test_requested_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CPU fallback"):
        resolve_runtime("cuda:0")


def test_pairing_and_errors(monkeypatch):
    assert resolve_runtime(dtype="complex64").real_dtype == torch.float32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="index 2"):
        resolve_runtime("cuda:2")
    for kwargs in ({"device": "mps"}, {"dtype": "float16"}, {"cpu_threads": 0}):
        with pytest.raises(ValueError):
            resolve_runtime(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_numeric_equivalence():
    H = torch.tensor([[[1, 0.2j], [0.1, 1]]], dtype=torch.complex128)
    y = torch.tensor([[0.3 + 0.2j, -0.4 + 0.8j]], dtype=torch.complex128)
    s = torch.tensor([0.5], dtype=torch.float64)
    cpu = DensePGVAMP(2)(H, y, s)
    gpu = DensePGVAMP(2, device="cuda")(H.cuda(), y.cuda(), s.cuda())
    torch.testing.assert_close(cpu.x_soft, gpu.x_soft.cpu(), atol=1e-9, rtol=1e-8)
    cv = dense_vamp_cholesky(H, y, s, iterations=2)
    gv = dense_vamp_cholesky(H.cuda(), y.cuda(), s.cuda(), iterations=2)
    torch.testing.assert_close(cv.x_soft, gv.x_soft.cpu(), atol=1e-9, rtol=1e-8)
