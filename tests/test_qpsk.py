"""Independent enumeration of §4.1 labels, energy and tie behavior."""

import math

import pytest
import torch

from pgvamp_ofdm.modulation.qpsk import (
    bits_to_classes,
    bits_to_symbols,
    classes_to_bits,
    classes_to_symbols,
    hard_decision,
)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
@pytest.mark.parametrize("bit_dtype", [torch.bool, torch.uint8, torch.int64])
def test_all_four_labels_energy_and_roundtrip(dtype, bit_dtype):
    bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bit_dtype)
    classes = torch.arange(4)
    expected = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=dtype) / math.sqrt(2)
    assert torch.equal(bits_to_classes(bits), classes)
    assert torch.equal(classes_to_bits(classes), bits.to(torch.uint8))
    assert torch.equal(classes_to_symbols(classes, dtype=dtype), expected)
    symbols = bits_to_symbols(bits, dtype=dtype)
    assert torch.equal(symbols, expected)
    assert torch.equal(hard_decision(symbols), classes)
    torch.testing.assert_close(symbols.abs().square(), torch.ones_like(symbols.real))


def test_ties_against_independent_nearest_constellation():
    constellation = torch.tensor(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=torch.complex128
    ) / math.sqrt(2)
    points = torch.tensor([0, 1, -1, 1j, -1j, -0.0, 0.12 - 0.7j, -8 + 2j], dtype=torch.complex128)
    expected = (points[:, None] - constellation).abs().square().argmin(-1)
    assert torch.equal(hard_decision(points), expected)
    assert hard_decision(points)[:5].tolist() == [0, 0, 2, 0, 1]


def test_batch_order_and_noncontiguous():
    bits = torch.tensor([[[[0, 1], [1, 0]], [[1, 1], [0, 0]]]], dtype=torch.uint8)
    bits = bits.transpose(1, 2)
    assert not bits.is_contiguous()
    assert torch.equal(classes_to_bits(bits_to_classes(bits)), bits)
    assert torch.equal(classes_to_bits(hard_decision(bits_to_symbols(bits))), bits)


@pytest.mark.parametrize(
    "bits",
    [
        torch.tensor([0]),
        torch.empty((0, 2), dtype=torch.int64),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0, 2]]),
        torch.tensor([[-1, 0]]),
    ],
)
def test_invalid_bits(bits):
    with pytest.raises(ValueError):
        bits_to_symbols(bits)


@pytest.mark.parametrize(
    "classes",
    [torch.tensor([4]), torch.tensor([-1]), torch.tensor([0.0]), torch.empty(0, dtype=torch.int64)],
)
def test_invalid_classes(classes):
    with pytest.raises(ValueError):
        classes_to_symbols(classes)


@pytest.mark.parametrize(
    "symbols",
    [
        torch.tensor([float("nan") + 0j]),
        torch.tensor([complex("inf")]),
        torch.ones(2),
        torch.empty(0, dtype=torch.complex128),
    ],
)
def test_invalid_decisions(symbols):
    with pytest.raises(ValueError):
        hard_decision(symbols)


def test_invalid_symbol_dtype():
    with pytest.raises(ValueError, match="dtype"):
        classes_to_symbols(torch.arange(4), dtype=torch.float64)
