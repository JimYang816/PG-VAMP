"""Fixed source index list is an independent oracle, including invalid mappings."""

from dataclasses import replace

import pytest
import torch

from pgvamp_ofdm.config import ConfigError, load_config
from pgvamp_ofdm.modulation.allocation import build_allocation, map_grid
from pgvamp_ofdm.modulation.qpsk import classes_to_symbols


def test_exact_indices_and_resource_partition():
    a = build_allocation(load_config())
    positive = [
        4,
        11,
        19,
        26,
        33,
        40,
        48,
        55,
        62,
        69,
        77,
        84,
        91,
        98,
        106,
        113,
        120,
        127,
        135,
        142,
        149,
        156,
        164,
        171,
        178,
        185,
        193,
        200,
        207,
        214,
        222,
        229,
    ]
    assert a.pilot_q.tolist() == [-q for q in positive[::-1]] + positive
    groups = [set(t.tolist()) for t in (a.data_q, a.pilot_q, a.guard_q, a.dc_q)]
    assert list(map(len, groups)) == [400, 64, 47, 1]
    assert set.union(*groups) == set(range(-256, 256))
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert a.guard_q.tolist() == list(range(-256, -232)) + list(range(233, 256))
    assert a.dc_q.tolist() == [0]
    assert a.q.tolist() == list(range(-256, 256))
    assert a.grid_index.tolist() == list(range(512))
    assert a.baseband_fft_index.tolist() == list(range(7936, 8192)) + list(range(256))
    assert a.passband_fft_index.tolist() == list(range(1792, 2304))
    assert torch.equal(a.q, a.grid_index - 256)
    assert torch.equal(a.q, a.passband_fft_index - 2048)
    freqs = 24000 + a.q.double() * (96000 / 8192)
    assert freqs[0] == 21000 and freqs[-1] == 26988.28125
    assert 96000 / 8192 == 6000 / 512


def test_map_and_independent_pilot_seed():
    a = build_allocation(load_config())

    def generate(seed):
        return classes_to_symbols(
            torch.randint(4, (2, 8, 400), generator=torch.Generator().manual_seed(seed))
        )

    pilots = classes_to_symbols(
        torch.randint(4, (2, 8, 64), generator=torch.Generator().manual_seed(991))
    )
    first, second = map_grid(generate(1), pilots, a), map_grid(generate(2), pilots, a)
    assert torch.equal(first[..., a.pilot_grid_index], second[..., a.pilot_grid_index])
    assert not torch.equal(first[..., a.data_grid_index], second[..., a.data_grid_index])
    assert torch.count_nonzero(first[..., a.guard_grid_index]) == 0
    assert torch.count_nonzero(first[..., a.dc_grid_index]) == 0
    torch.testing.assert_close(
        first.abs().square().sum(-1), torch.full((2, 8), 464.0, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "field",
    [
        "q",
        "grid_index",
        "baseband_fft_index",
        "passband_fft_index",
        "data_q",
        "pilot_q",
        "guard_q",
        "dc_q",
        "data_grid_index",
    ],
)
def test_corrupted_allocation_rejected(field):
    a = build_allocation(load_config())
    value = getattr(a, field).clone()
    value[0] += 1
    with pytest.raises(ValueError):
        replace(a, **{field: value}).validate()


def test_wrong_precision_and_nonqpsk_map_rejected():
    a = build_allocation(load_config())
    data = classes_to_symbols(torch.zeros(400, dtype=torch.int64))
    pilots = classes_to_symbols(torch.zeros(64, dtype=torch.int64))
    for bad_data, bad_pilot in [
        (data[:-1], pilots),
        (data, pilots.to(torch.complex64)),
        (data * 2, pilots),
        (data, pilots * 1j * 0),
        (data * complex("nan"), pilots),
        (data.real, pilots),
    ]:
        with pytest.raises(ValueError):
            map_grid(bad_data, bad_pilot, a)


def test_algebra_coding_and_overlap_rejected():
    with pytest.raises(ConfigError, match="algebra"):
        build_allocation(load_config("configs/smoke_math.yaml"))
    config = load_config()
    config.values["waveform"]["coding"] = "ldpc"
    with pytest.raises(ConfigError, match="coding"):
        build_allocation(config)
    a = build_allocation(load_config())
    with pytest.raises(ValueError, match="DC"):
        replace(a, carrier_bin=256, passband_fft_index=a.q + 256).validate()


@pytest.mark.parametrize(
    "changes", [{"q": None}, {"pilot_q": [4]}, {"n_fft_wave": 8192.0}, {"carrier_bin": True}]
)
def test_malformed_allocation_fields_rejected(changes):
    with pytest.raises(ValueError, match="allocation"):
        replace(build_allocation(load_config()), **changes).validate()
