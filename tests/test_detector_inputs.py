import pytest
import torch

from pgvamp_ofdm.algorithms import MMSEDetector, PGVAMPDetector, VAMPDetector
from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.records import tensor_hash


@pytest.mark.parametrize("detector", [MMSEDetector, VAMPDetector, PGVAMPDetector])
@pytest.mark.parametrize(
    "problem",
    [
        "square",
        "batch",
        "empty",
        "real_H",
        "y_dtype",
        "noise_dtype",
        "zero_noise",
        "negative_noise",
        "nan_H",
        "inf_y",
        "nan_noise",
        "device",
    ],
)
def test_invalid_system(detector, problem):
    H = torch.eye(2, dtype=torch.complex128)[None]
    y = torch.ones(1, 2, dtype=H.dtype)
    sigma = torch.ones(1, dtype=torch.float64)
    if problem == "square":
        H = H[:, :, :1]
    elif problem == "batch":
        y = y.repeat(2, 1)
    elif problem == "empty":
        H, y, sigma = H[:0], y[:0], sigma[:0]
    elif problem == "real_H":
        H = H.real
    elif problem == "y_dtype":
        y = y.to(torch.complex64)
    elif problem == "noise_dtype":
        sigma = sigma.float()
    elif problem == "zero_noise":
        sigma[0] = 0
    elif problem == "negative_noise":
        sigma[0] = -1
    elif problem == "nan_H":
        H[0, 0, 0] = float("nan")
    elif problem == "inf_y":
        y[0, 0] = float("inf")
    elif problem == "nan_noise":
        sigma[0] = float("nan")
    elif problem == "device":
        y = y.to("meta")
    with pytest.raises(ValueError):
        detector().detect(H, y, sigma)


@pytest.mark.parametrize("detector", [MMSEDetector, VAMPDetector, PGVAMPDetector])
def test_nonfinite_operator_is_hard_failure(detector):
    H = torch.eye(2, dtype=torch.complex128)[None] * 1e200
    y = torch.ones(1, 2, dtype=H.dtype)
    with pytest.raises(FloatingPointError, match=r"batch index=.*dtype=.*device="):
        detector().detect(H, y, torch.ones(1, dtype=torch.float64))


def test_factorization_failure_context(monkeypatch):
    H = torch.eye(2, dtype=torch.complex128)[None]
    y = torch.ones(1, 2, dtype=H.dtype)
    sigma = torch.ones(1, dtype=torch.float64)

    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("injected failure")

    monkeypatch.setattr(torch.linalg, "svd", fail)
    with pytest.raises(FloatingPointError, match="SVD failed.*batch index=.*device=cpu"):
        VAMPDetector().detect(H, y, sigma)
    monkeypatch.setattr(
        torch.linalg, "cholesky_ex", lambda *a, **kw: (H, torch.ones(1, dtype=torch.int32))
    )
    with pytest.raises(FloatingPointError, match="Cholesky failed.*batch index=.*device=cpu"):
        MMSEDetector().detect(H, y, sigma)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("detector", [MMSEDetector, VAMPDetector])
def test_cuda_parity(detector):
    gen = torch.Generator().manual_seed(6)
    H = torch.randn(2, 8, 8, dtype=torch.complex128, generator=gen)
    y = torch.randn(2, 8, dtype=H.dtype, generator=gen)
    sigma = torch.ones(2, dtype=torch.float64)
    cpu = detector().detect(H, y, sigma)
    gpu = detector().detect(H.cuda(), y.cuda(), sigma.cuda())
    torch.testing.assert_close(cpu.x_soft, gpu.x_soft.cpu(), atol=1e-9, rtol=1e-8)


def test_real_400_dimensional_shared_inputs_and_label_isolation(data_manifest):
    dataset = EffectiveDataset(data_manifest, "test")
    sample = dataset[16]  # WP3 fixed-seed nonzero, unequal time-scaling scenario.
    assert torch.count_nonzero(dataset.records[2]["path_epsilon"]) > 0
    assert sample["H"].shape == (400, 400)
    original = detection_inputs(sample)
    fingerprint = tensor_hash(original)
    identities = []
    for detector in (MMSEDetector(), VAMPDetector(), PGVAMPDetector()):
        sample = dataset[16]  # Each detector starts with the original physical labels.
        payload = detection_inputs(sample)
        identities.append((sample["sample_id"], tensor_hash(payload)))
        baseline = detector.detect(**{k: v.unsqueeze(0) for k, v in payload.items()})
        assert torch.isfinite(baseline.x_soft).all()
        assert tensor_hash(payload) == fingerprint
        sample["x"], sample["bits"] = (
            torch.zeros_like(sample["x"]),
            torch.zeros_like(sample["bits"]),
        )
        modified = detector.detect(
            **{k: v.unsqueeze(0) for k, v in detection_inputs(sample).items()}
        )
        unlabelled = {k: v for k, v in sample.items() if k not in ("x", "bits")}
        absent = detector.detect(
            **{k: v.unsqueeze(0) for k, v in detection_inputs(unlabelled).items()}
        )
        assert torch.equal(baseline.x_soft, modified.x_soft)
        assert torch.equal(baseline.x_soft, absent.x_soft)

        def predictions(output):
            tensors = {key: getattr(output, key) for key in ("x_soft", "class_hat", "bits_hat")}
            if output.probabilities is not None:
                tensors["probabilities"] = output.probabilities
            return tensor_hash(tensors)

        assert predictions(baseline) == predictions(modified) == predictions(absent)
        assert tensor_hash(detection_inputs(sample)) == fingerprint
        assert tensor_hash(detection_inputs(unlabelled)) == fingerprint
    assert len(set(identities)) == 1
    assert tensor_hash(detection_inputs(sample)) == fingerprint
