"""Independent physical FFT/Parseval/frame tests at the full 512/8192/400 size."""

import math

import pytest
import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.modulation.qpsk import classes_to_bits, classes_to_symbols, hard_decision
from pgvamp_ofdm.utils.device import resolve_runtime
from pgvamp_ofdm.waveform.frame import build_frame, pad_recording
from pgvamp_ofdm.waveform.ofdm import modulate_grid


def make_frame(dtype="complex128", device="cpu", *, leading=None):
    config = load_config(dtype=dtype, device=device)
    if leading is not None:
        config.values["frame"]["leading_silence_samples"] = leading
    runtime = resolve_runtime(device=device, dtype=dtype)
    # CPU streams then explicit move also make CPU/GPU comparisons use identical labels.
    bits = torch.randint(
        2, (2, 8, 400, 2), generator=torch.Generator().manual_seed(14), dtype=torch.uint8
    ).to(device)
    pilots = classes_to_symbols(
        torch.randint(4, (2, 8, 64), generator=torch.Generator().manual_seed(29)),
        dtype=runtime.complex_dtype,
    ).to(device)
    return build_frame(bits, pilots, config, runtime), bits, config, runtime


@pytest.mark.parametrize(
    "dtype,atol,rtol", [(torch.complex128, 1e-9, 1e-8), (torch.complex64, 2e-5, 2e-4)]
)
def test_multibatch_fft_parseval_and_cp(dtype, atol, rtol):
    grid = torch.randn(
        3, 2, 512, dtype=dtype, generator=torch.Generator().manual_seed(43)
    ).transpose(0, 1)
    assert not grid.is_contiguous()
    ofdm = modulate_grid(grid, load_config())
    expected_spectrum = grid.new_zeros((2, 3, 8192))
    expected_spectrum[..., list(range(7936, 8192)) + list(range(256))] = grid
    actual = torch.fft.fft(ofdm.useful, dim=-1, norm="ortho")
    torch.testing.assert_close(actual, expected_spectrum, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        ofdm.useful.abs().square().sum(-1), grid.abs().square().sum(-1), atol=atol, rtol=rtol
    )
    assert ofdm.with_cp.shape == (2, 3, 10240)
    assert torch.equal(ofdm.with_cp[..., :2048], ofdm.useful[..., -2048:])
    assert torch.equal(ofdm.with_cp[..., 2048:], ofdm.useful)


@pytest.mark.parametrize("dtype,atol,rtol", [("complex128", 1e-9, 1e-8), ("complex64", 2e-5, 2e-4)])
@pytest.mark.parametrize("leading", [1920, 1921])
def test_complete_frame_rf_recovery_absolute_phase(dtype, atol, rtol, leading):
    frame, bits, config, runtime = make_frame(dtype, leading=leading)
    layout = frame.layout
    assert frame.real.shape == (2, 91776 + leading - 1920)
    assert layout.frame_samples == frame.real.shape[-1]
    assert layout.data_bits_per_frame == 6400
    if leading == 1920:
        assert layout.frame_samples / layout.sample_rate_hz == 0.956
        assert layout.lfm == (1920, 5760)
        assert layout.cp[0] == (7808, 9856)
        assert layout.useful[0] == (9856, 18048)
        assert layout.trailing_silence == (89728, 91776)
    for start, stop in (layout.leading_silence, layout.sync_guard, layout.trailing_silence):
        assert torch.count_nonzero(frame.analytic[:, start:stop]) == 0
    torch.testing.assert_close(
        frame.ofdm.useful.abs().square().mean(-1),
        torch.full((2, 8), 464 / 8192, dtype=runtime.real_dtype),
        atol=atol,
        rtol=rtol,
    )
    q = torch.arange(-256, 256)
    for m, ((cp_start, cp_stop), (start, stop)) in enumerate(
        zip(layout.cp, layout.useful, strict=True)
    ):
        assert start == 9856 + leading - 1920 + m * 10240
        assert cp_stop == start and stop - start == 8192
        spectrum = torch.fft.fft(frame.real[:, start:stop], norm="ortho", dim=-1)
        phase = torch.exp(
            torch.tensor(2j * math.pi * 24000 * start / 96000, dtype=torch.complex128)
        ).to(runtime.complex_dtype)
        recovered = math.sqrt(2) * spectrum[:, 2048 + q] / phase
        torch.testing.assert_close(recovered, frame.grid[:, m], atol=atol, rtol=rtol)
        recovered_bits = classes_to_bits(
            hard_decision(recovered[:, frame.allocation.data_grid_index])
        )
        assert torch.equal(recovered_bits, bits[:, m])
        # Explicit independent sample formula checks whole CP+useful carrier phase.
        time = (
            torch.arange(cp_start, stop, dtype=torch.float64)
            / config.values["waveform"]["sample_rate_hz"]
        )
        expected = frame.ofdm.with_cp[:, m] * torch.exp(2j * math.pi * 24000 * time).to(
            runtime.complex_dtype
        )
        torch.testing.assert_close(frame.analytic[:, cp_start:stop], expected, atol=atol, rtol=rtol)
    assert torch.equal(frame.real, math.sqrt(2) * frame.analytic.real)
    assert frame.analytic.dtype == runtime.complex_dtype and frame.real.dtype == runtime.real_dtype


def test_arrival_padding_and_no_rng_consumption():
    frame, bits, config, runtime = make_frame()
    rng_before = torch.random.get_rng_state()
    rebuilt = build_frame(bits, frame.grid[..., frame.allocation.pilot_grid_index], config, runtime)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(rebuilt.analytic, frame.analytic)
    changed = build_frame(
        1 - bits, frame.grid[..., frame.allocation.pilot_grid_index], config, runtime
    )
    assert torch.equal(changed.lfm.analytic, frame.lfm.analytic)
    assert changed.lfm.normalization == frame.lfm.normalization
    for offset in (0, 19, 1920):
        recording = pad_recording(frame.analytic, offset)
        assert recording.shape[-1] == 91776 + offset
        assert torch.count_nonzero(recording[..., :offset]) == 0
        assert torch.equal(recording[..., offset:], frame.analytic)
        assert frame.layout.frame_samples == 91776


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_ofdm_has_finite_energy_gradient(dtype):
    grid = torch.randn(
        2, 512, dtype=dtype, generator=torch.Generator().manual_seed(712), requires_grad=True
    )
    modulate_grid(grid, load_config()).useful.abs().square().sum().backward()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()
    torch.testing.assert_close(
        grid.grad,
        2 * grid,
        atol=2e-5 if dtype == torch.complex64 else 1e-9,
        rtol=2e-4 if dtype == torch.complex64 else 1e-8,
    )


def test_supported_alternate_fft_cp_and_block_count():
    config = load_config()
    config.values["waveform"].update(
        sample_rate_hz=48000,
        band_hz=[9000, 15000],
        carrier_hz=12000,
        n_fft_wave=4096,
        cp_samples=1024,
    )
    config.values["frame"].update(n_ofdm_symbols=3, lfm_start_hz=9000, lfm_stop_hz=15000)
    runtime = resolve_runtime()
    bits = torch.zeros((1, 3, 400, 2), dtype=torch.uint8)
    pilots = classes_to_symbols(torch.zeros((1, 3, 64), dtype=torch.int64))
    frame = build_frame(bits, pilots, config, runtime)
    assert frame.ofdm.useful.shape == (1, 3, 4096)
    assert frame.layout.frame_samples == 1920 + 3840 + 2048 + 3 * 5120 + 2048
    assert frame.layout.data_bits_per_frame == 2400


def test_cpu_path_does_not_call_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU must not probe CUDA")

    for name in ("is_available", "device_count", "manual_seed", "manual_seed_all", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    make_frame()


def test_cuda_explicit_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; no GPU computation performed")
    cpu, *_ = make_frame()
    gpu, *_ = make_frame(device="cuda:0")
    torch.testing.assert_close(gpu.analytic.cpu(), cpu.analytic, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize(
    "grid",
    [
        torch.ones(512),
        torch.ones(511, dtype=torch.complex128),
        torch.empty((0, 512), dtype=torch.complex128),
        torch.full((512,), complex("nan"), dtype=torch.complex128),
    ],
)
def test_invalid_grid(grid):
    with pytest.raises(ValueError):
        modulate_grid(grid, load_config())


def test_invalid_frame_and_offsets():
    frame, bits, config, runtime = make_frame()
    pilots = frame.grid[..., frame.allocation.pilot_grid_index]
    for bad_bits, bad_pilots in [
        (bits[:, :2], pilots),
        (bits, pilots[..., :-1]),
        (bits, pilots.to(torch.complex64)),
        (bits.float(), pilots),
        (bits[:0], pilots[:0]),
    ]:
        with pytest.raises(ValueError):
            build_frame(bad_bits, bad_pilots, config, runtime)
    for offset in (-1, 1.5, True):
        with pytest.raises(ValueError):
            pad_recording(frame.real, offset)
    with pytest.raises(ValueError, match="runtime"):
        build_frame(bits, pilots, config, resolve_runtime(dtype="complex64"))
