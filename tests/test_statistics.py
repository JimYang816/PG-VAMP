import numpy as np
import pytest
import torch

from pgvamp_ofdm.evaluation.metrics import aggregate, block_counts, frame_counts
from pgvamp_ofdm.evaluation.statistics import bootstrap
from pgvamp_ofdm.modulation.qpsk import bits_to_symbols


def test_hand_counts_energy_before_db_and_failures():
    bits = torch.zeros(3, 2, dtype=torch.uint8)
    target = bits_to_symbols(bits)
    predicted = bits.clone()
    predicted[0, 0] = 1
    predicted[1] = 1
    bad = block_counts(predicted, target * 2, bits, target)
    good = block_counts(bits, target, bits, target)
    assert bad["bit_errors"] == 3 and bad["symbol_errors"] == 2
    first = frame_counts([bad] + [good] * 7, 3)
    second = frame_counts([good] * 8, 3)
    combined = aggregate([first, second])
    assert combined["ber"] == 3 / 96
    assert combined["ser"] == 2 / 48
    assert combined["bler"] == 1 / 16
    assert combined["fer"] == 1 / 2
    assert combined["nmse_linear"] == pytest.approx(1 / 16)
    assert combined["nmse_db"] == pytest.approx(10 * np.log10(1 / 16))
    assert combined["evm_pct"] == pytest.approx(25)
    assert combined["goodput_bps"] == pytest.approx(6400 / 0.956 / 2)
    failed = frame_counts([None, bad] + [good] * 6, 3)
    assert failed["n_bits"] == 48 and failed["successful_blocks"] == 7
    assert failed["ber"] is None and failed["fer"] is None
    assert failed["conditional_success_ber"] == 3 / 42
    assert aggregate([failed, second])["status"] == "incomplete_or_failed"
    assert second["ber"] == 0 and second["fer_zero_upper95"] == pytest.approx(0.95)
    assert second["nmse_db"] is None and second["nmse_zero"]
    with pytest.raises(ValueError, match="eight"):
        frame_counts([good], 3)


def frames():
    return [
        {
            "frame_id": str(i),
            "bit_errors": b,
            "symbol_errors": b,
            "n_bits": n,
            "n_symbols": n // 2,
            "status": "complete",
        }
        for i, (b, n) in enumerate(((0, 10), (2, 20), (8, 40)))
    ]


def test_bootstrap_exact_shared_frame_indices_and_fixed_seed():
    a = frames()
    b = [{**r, "bit_errors": r["bit_errors"] // 2, "symbol_errors": 0} for r in a]
    intervals, paired = bootstrap({"a": a, "b": b}, seed=57, repeats=2000)
    assert (intervals, paired) == bootstrap({"a": a[::-1], "b": b}, seed=57)
    rng = np.random.default_rng(57)
    sampled, delta = [], []
    for _ in range(2000):
        idx = rng.integers(0, 3, size=3)
        denominator = sum(a[i]["n_bits"] for i in idx)
        sampled.append(sum(a[i]["bit_errors"] for i in idx) / denominator)
        delta.append(sum(a[i]["bit_errors"] - b[i]["bit_errors"] for i in idx) / denominator)
    assert intervals["a"]["ber_ci_low"] == np.quantile(sampled, 0.025)
    assert paired[0]["ci_high"] == np.quantile(delta, 0.975)
    assert intervals["a"]["ci_unstable_few_frames"]


def test_zero_failed_and_unpaired_bootstrap():
    a = [{**r, "bit_errors": 0, "symbol_errors": 0} for r in frames()]
    intervals, paired = bootstrap({"a": a}, seed=2)
    assert intervals["a"]["ber_ci_low"] == intervals["a"]["ber_ci_high"] == 0
    assert intervals["a"]["zero_ci_does_not_prove_zero"] and not paired
    failed = [{**r, "status": "incomplete_or_failed"} for r in a]
    intervals, paired = bootstrap({"a": a, "failed": failed}, seed=2)
    assert intervals["failed"]["ber_ci_low"] is None and not paired
    with pytest.raises(ValueError, match="identities"):
        bootstrap({"a": a, "b": a[:-1]}, seed=2)
