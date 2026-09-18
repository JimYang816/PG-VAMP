"""Acceptance counts are checked using constructed errors, not detector agreement."""

import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.modulation.qpsk import bits_to_symbols
from pgvamp_ofdm.smoke import error_counts, smoke


def test_hand_counts_and_incomplete_frame():
    target = torch.zeros(3, 2, 2, dtype=torch.uint8)
    prediction = target.clone()
    prediction[0, 0, 0] = 1
    prediction[1, 1, :] = 1
    truth = bits_to_symbols(target)
    estimate = bits_to_symbols(prediction)
    counts = error_counts(prediction, target, estimate, truth, ["a", "a", "b"], [0, 1, 0], 2)
    assert counts["bit_errors"] == 3 and counts["bits"] == 12
    assert counts["symbol_errors"] == 2 and counts["symbols"] == 6
    assert counts["block_errors"] == 2 and counts["blocks"] == 3
    assert (
        counts["frame_errors"] == 1 and counts["frames"] == 1 and counts["incomplete_frames"] == 1
    )
    assert abs(counts["squared_error_energy"] - 6) < 1e-12
    assert abs(counts["target_energy"] - 6) < 1e-12


def test_math_smoke_cpu_forbidden_cuda(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU called CUDA")

    for key in (
        "is_available",
        "get_rng_state_all",
        "set_rng_state_all",
        "manual_seed_all",
        "device_count",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, key, forbidden)
    report = smoke(load_config("configs/smoke_math.yaml"), tmp_path / "smoke")
    assert report["updates"] == 2 and report["dimension"] == 32
