import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector, PGVAMPDetector, VAMPDetector
from pgvamp_ofdm.algorithms.prepared import PreparedDetector, timed_detect
from pgvamp_ofdm.evaluation.timing import measure_detector, synchronize


def system(dtype=torch.complex128):
    gen = torch.Generator().manual_seed(713)
    h = torch.randn(1, 8, 8, dtype=dtype, generator=gen) / 8**0.5
    y = torch.randn(6, 8, dtype=dtype, generator=gen)
    return h, y, torch.tensor([0.7], dtype=h.real.dtype)


@pytest.mark.parametrize("factory", [MMSEDetector, VAMPDetector, PGVAMPDetector])
@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_prepared_multiple_observations_equal_and_invalidation(factory, dtype):
    h, ys, s = system(dtype)
    model = factory().to(dtype=h.real.dtype).eval()
    with pytest.raises(ValueError, match="inference_mode"):
        PreparedDetector(model, h, s)
    with torch.inference_mode():
        prepared = PreparedDetector(model, h, s)
        for y in ys:
            cold = model.detect(h, y[None], s)
            cached = prepared.detect(h, y[None], s)
            torch.testing.assert_close(cold.x_soft, cached.x_soft, atol=0, rtol=0)
            torch.testing.assert_close(cold.bits_hat, cached.bits_hat)
        for altered_h, altered_s in (
            (h * 2, s),
            (h, s * 2),
            (h.to(torch.complex64 if dtype == torch.complex128 else torch.complex128), s),
        ):
            with pytest.raises(ValueError, match="invalidated"):
                prepared.detect(altered_h, ys[:1], altered_s)
        if isinstance(model, PGVAMPDetector):
            model.raw_mu.add_(0.1)
            with pytest.raises(ValueError, match="model"):
                prepared.detect(h, ys[:1], s)
        elif isinstance(model, VAMPDetector):
            model.iterations += 1
            with pytest.raises(ValueError, match="model"):
                prepared.detect(h, ys[:1], s)
        model.train()
        with pytest.raises(ValueError, match="eval"):
            prepared.detect(h, ys[:1], s)


@pytest.mark.parametrize(
    "factory,expected_cold,expected_cached",
    [(MMSEDetector, 1, 0), (PGVAMPDetector, 8, 8), (VAMPDetector, 0, 0)],
)
def test_factorization_boundaries(monkeypatch, factory, expected_cold, expected_cached):
    h, y, s = system()
    model = factory().eval()
    calls = {"chol": 0, "svd": 0}
    chol, svd = torch.linalg.cholesky_ex, torch.linalg.svd

    def wrap_chol(*a, **k):
        calls["chol"] += 1
        return chol(*a, **k)

    def wrap_svd(*a, **k):
        calls["svd"] += 1
        return svd(*a, **k)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", wrap_chol)
    monkeypatch.setattr(torch.linalg, "svd", wrap_svd)
    with torch.inference_mode():
        timed_detect(model, h, y[:1], s)
        assert calls == {"chol": expected_cold, "svd": int(factory is VAMPDetector)}
        prepared = PreparedDetector(model, h, s)
        calls.update(chol=0, svd=0)
        prepared.detect(h, y[1:2], s)
        assert calls == {"chol": expected_cached, "svd": 0}


def test_timing_metadata_b1_batch_and_cpu_no_cuda(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("CPU benchmark invoked CUDA")

    for name in ("synchronize", "is_available", "max_memory_allocated", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    h, y, s = system()
    for mode in ("per_observation_cold_H", "same_H_amortized"):
        rows = measure_detector(
            MMSEDetector(), h, y, s, mode=mode, batch_size=2, warmup=1, repeats=3
        )
        assert [r["batch_size"] for r in rows] == [1, 2]
        assert rows[0]["latency_kind"] == "single_block_online"
        assert rows[1]["latency_kind"] == "batch_latency"
        assert rows[1]["amortized_per_block_ms"] == rows[1]["mean_latency_ms"] / 2
        assert rows[1]["data_bits_per_second"] == 16 * rows[1]["blocks_per_second"]
        assert rows[0]["cpu_process_peak_bytes"] > 0
        assert (
            rows[0]["prepare_ms"] > 0 if mode == "same_H_amortized" else rows[0]["prepare_ms"] == 0
        )


def test_explicit_cuda_sync_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: calls.append(d))
    synchronize(torch.device("cuda:0"))
    assert calls == [torch.device("cuda:0")]
