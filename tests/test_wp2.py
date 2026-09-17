"""Full-size independent physical and statistical WP2 acceptance tests."""

import math
from dataclasses import replace

import pytest
import torch

from pgvamp_ofdm.channel.affine import affine_waveform, receive_window
from pgvamp_ofdm.channel.effective_matrix import dirichlet_kernel, effective_matrix
from pgvamp_ofdm.channel.noise import complex_awgn, sigma2_from_esn0
from pgvamp_ofdm.channel.parameters import SCENARIOS, PathParameters, sample_paths
from pgvamp_ofdm.channel.validity import validate_cp_support
from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.modulation.qpsk import classes_to_symbols
from pgvamp_ofdm.receiver.fft_receiver import fft_receive
from pgvamp_ofdm.receiver.preprocessing import preprocess
from pgvamp_ofdm.utils.device import resolve_runtime
from pgvamp_ofdm.waveform.continuous import evaluate_continuous
from pgvamp_ofdm.waveform.frame import build_frame


def generator(seed):
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def physical():
    config = load_config()
    bits = torch.randint(2, (1, 8, 400, 2), generator=generator(121))
    pilots = classes_to_symbols(torch.randint(4, (1, 8, 64), generator=generator(122)))
    return config, build_frame(bits, pilots, config, resolve_runtime())


def explicit_paths(epsilon=(0.00019, -0.00017)):
    return PathParameters(
        torch.tensor([0.8 + 0.1j, -0.3 + 0.5j], dtype=torch.complex128),
        torch.tensor([0.002, 0.013], dtype=torch.float64),
        torch.tensor(epsilon, dtype=torch.float64),
        "affine_doppler_strong",
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_path_sampling(physical, scenario):
    config, frame = physical
    for seed in range(16):
        paths = sample_paths(config, scenario, generator=generator(seed))
        again = sample_paths(config, scenario, generator=generator(seed))
        paths.validate()
        assert torch.equal(paths.gain, again.gain)
        assert torch.equal(paths.delay_s, again.delay_s)
        assert torch.equal(paths.epsilon, again.epsilon)
        assert float(paths.gain.abs().square().sum()) == pytest.approx(1)
        validate_cp_support(paths, frame.layout, config)
        if scenario != "identity_awgn":
            assert 3 <= paths.gain.numel() <= 6
            assert paths.delay_s[0] == 0.002
            assert bool(((paths.delay_s[1:] > 0.002) & (paths.delay_s[1:] <= 0.014)).all())
        if scenario.startswith("affine"):
            assert (
                float(paths.epsilon.abs().max())
                <= config.values["channel"]["epsilon_max"][scenario]
            )


def test_cp_boundaries_and_last_block(physical):
    config, frame = physical
    # Use the same represented CP duration to test lower-endpoint inclusion exactly.
    cp = 2048 / 96000
    p = PathParameters(
        torch.ones(1, dtype=torch.complex128),
        torch.tensor([cp], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        "static_multipath",
    )
    endpoints = validate_cp_support(p, frame.layout, config)
    assert endpoints[0, 0, 0] == -cp
    with pytest.raises(ValueError, match="block=0 path=0 endpoint=0"):
        validate_cp_support(replace(p, delay_s=p.delay_s + 1e-10), frame.layout, config)
    identity = sample_paths(config, "identity_awgn", generator=generator(1))
    upper_offset = 8192 / 96000 - 8191 / 96000
    assert upper_offset + 8191 / 96000 == 8192 / 96000
    with pytest.raises(ValueError, match="endpoint=1"):
        validate_cp_support(identity, frame.layout, config, window_offsets_s=(upper_offset,) * 8)
    with pytest.raises(ValueError, match="endpoint=1"):
        validate_cp_support(identity, frame.layout, config, window_offsets_s=(2 / 96000,) * 8)
    # Positive scaling runs beyond useful support only late in the frame.
    late = replace(
        p,
        delay_s=torch.tensor([0.002], dtype=torch.float64),
        epsilon=torch.tensor([0.0025], dtype=torch.float64),
        scenario="affine_doppler_strong",
    )
    with pytest.raises(ValueError, match="block=6|block=7"):
        validate_cp_support(late, frame.layout, config)
    validate_cp_support(
        explicit_paths(), frame.layout, config, window_offsets_s=(-0.0002, 0, 0, 0, 0, 0, 0, 0.0003)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("epsilon", torch.tensor([-1.0, 0.0], dtype=torch.float64)),
        ("delay_s", torch.tensor([0.02, 0.001], dtype=torch.float64)),
        ("gain", torch.tensor([complex(float("nan"), 0), 1], dtype=torch.complex128)),
        ("delay_s", torch.zeros(2, dtype=torch.float32)),
    ],
)
def test_invalid_paths(field, value):
    with pytest.raises(ValueError):
        replace(explicit_paths(), **{field: value}).validate()


def test_dirichlet_periodic_poles_and_direct_sum():
    z = torch.tensor([0.0, 1.0, -7.0, 0.123, 8192.0, -8192.0, 8192.123], dtype=torch.float64)
    n = torch.arange(8192, dtype=torch.float64)
    expected = torch.exp(2j * math.pi * z[:, None] * n / 8192).mean(-1)
    torch.testing.assert_close(dirichlet_kernel(z, 8192), expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("scenario", ["identity_awgn", "static_multipath"])
def test_static_limits(physical, scenario):
    config, frame = physical
    p = sample_paths(config, scenario, generator=generator(14))
    h = effective_matrix(p, frame.layout, config, 7)
    frequency = 24000 + frame.allocation.q.double() * 96000 / 8192
    diagonal = (
        p.gain[:, None] * torch.exp(-2j * math.pi * p.delay_s[:, None] * frequency[None, :])
    ).sum(0)
    torch.testing.assert_close(h, torch.diag(diagonal), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("block,offset", [(0, 0.0), (7, 0.0), (0, -0.0002), (7, 0.0003)])
def test_nonzero_physical_equivalence(physical, block, offset):
    config, frame = physical
    p = explicit_paths()
    h = effective_matrix(p, frame.layout, config, block, window_offset_s=offset)
    waveform = receive_window(frame, p, config, block, window_offset_s=offset)
    actual = fft_receive(waveform, frame.allocation)[0]
    expected = h @ frame.grid[0, block]
    assert (
        float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(actual)) < 1e-9
    )
    offdiag = h - torch.diag(h.diag())
    assert float(offdiag.abs().square().sum() / h.abs().square().sum()) > 0.03
    # Wideband Doppler is q-dependent, and the full matrix retains remote ICI.
    assert abs(float(p.epsilon[0]) * (26988.28125 - 21000)) > 1
    assert abs(h[0, -1]) > 1e-6


def test_continuous_integer_frame_and_noninteger_boundaries(physical):
    config, frame = physical
    t = torch.arange(frame.layout.frame_samples, dtype=torch.float64) / 96000
    actual = evaluate_continuous(frame, t, config)
    torch.testing.assert_close(actual, frame.analytic, atol=3e-11, rtol=1e-9)
    silence = torch.tensor([-1.0, 0.0, 0.01, 0.065, 1.0, 2.0], dtype=torch.float64)
    assert not bool(evaluate_continuous(frame, silence, config).any())
    # Last taper sample through the following guard stays zero; no cosine rebound.
    edge = torch.tensor([5759.25, 5759.5, 5760.0], dtype=torch.float64) / 96000
    assert not bool(evaluate_continuous(frame, edge, config).any())
    # Interior chirp evaluated independently at a fractional sample.
    local = 1000.375 / 96000
    chirp = evaluate_continuous(frame, torch.tensor([0.02 + local], dtype=torch.float64), config)
    expected = frame.lfm.normalization * complex(
        math.cos(2 * math.pi * (21000 * local + 75000 * local**2)),
        math.sin(2 * math.pi * (21000 * local + 75000 * local**2)),
    )
    torch.testing.assert_close(chirp, torch.full_like(chirp, expected), atol=1e-12, rtol=1e-12)
    # Fractional CP sample uses the current block, including its absolute carrier.
    time = (frame.layout.cp[1][0] + 0.375) / 96000
    local = time - frame.layout.useful[1][0] / 96000
    expected = sum(
        complex(frame.grid[0, 1, j])
        * complex(
            math.cos(2 * math.pi * int(q) * 96000 / 8192 * local),
            math.sin(2 * math.pi * int(q) * 96000 / 8192 * local),
        )
        for j, q in enumerate(frame.allocation.q)
    ) / math.sqrt(8192)
    expected *= complex(math.cos(2 * math.pi * 24000 * time), math.sin(2 * math.pi * 24000 * time))
    actual = evaluate_continuous(frame, torch.tensor([time], dtype=torch.float64), config)
    assert complex(actual[0, 0]) == pytest.approx(expected, abs=1e-12)


def test_affine_recording_offset(physical):
    config, frame = physical
    t = torch.tensor([0.0, 0.025, 0.15, 0.9, 1.1], dtype=torch.float64)
    p = explicit_paths()
    original = affine_waveform(frame, p, t, config)
    padded = affine_waveform(frame, p, t + 13 / 96000, config, arrival_offset_samples=13)
    torch.testing.assert_close(original, padded, atol=1e-12, rtol=1e-10)


def test_pilot_cancellation_and_noise(physical):
    config, frame = physical
    h = effective_matrix(explicit_paths(), frame.layout, config, 7)
    wave = receive_window(frame, explicit_paths(), config, 7)[0]
    idx, pi = frame.allocation.data_grid_index, frame.allocation.pilot_grid_index
    pilots, data = frame.grid[0, 7, pi], frame.grid[0, 7, idx]
    leakage = h[idx][:, pi] @ pilots
    assert float(leakage.abs().square().sum()) > 1
    for noise in [
        torch.zeros_like(wave),
        complex_awgn((8192,), 0.1, gen_r=generator(6), gen_i=generator(7)),
    ]:
        observed = fft_receive(wave + noise, frame.allocation)
        sample = preprocess(h, observed, pilots, frame.allocation, 0.1)
        expected = sample.H @ data + fft_receive(noise, frame.allocation)[idx]
        torch.testing.assert_close(sample.y, expected, atol=1e-9, rtol=1e-9)
    with pytest.raises(ValueError, match="estimated"):
        preprocess(h, observed, pilots, frame.allocation, 0.1, csi_mode="estimated")
    with pytest.raises(ValueError, match="shapes"):
        preprocess(h[None], observed, pilots, frame.allocation, 0.1)


def test_noise_statistics_and_identity_ber(physical):
    _, frame = physical
    variance = sigma2_from_esn0(4.0)
    # 2,097,152 time samples; 6 standard errors < 0.6% for component variance.
    noise = complex_awgn((256, 8192), variance, gen_r=generator(130), gen_i=generator(131))
    assert float(noise.real.var()) == pytest.approx(variance / 2, rel=0.006)
    assert float(noise.imag.var()) == pytest.approx(variance / 2, rel=0.006)
    assert abs(float(noise.real.mean())) < 6 * math.sqrt(variance / (2 * noise.numel()))
    assert abs(float(noise.imag.mean())) < 6 * math.sqrt(variance / (2 * noise.numel()))
    assert abs(float((noise.real * noise.imag).mean())) < 3 * variance / math.sqrt(noise.numel())
    freq = fft_receive(noise, frame.allocation)
    assert float(freq.abs().square().mean()) == pytest.approx(variance, rel=0.018)
    # Disjoint frequency pairs pooled over 256 windows: 65,536 independent pairs.
    covariance = (freq[:, :256] * freq[:, 256:].conj()).mean()
    assert abs(covariance) < 6 * variance / math.sqrt(65536)
    selected = freq[:, [0, 17, 91, 200, 301, 400, 450, 511]]
    matrix = selected.T @ selected.conj() / selected.shape[0]
    # Seven-sigma simultaneous guard for all 64 covariance entries (not their average).
    assert float((matrix - variance * torch.eye(8)).abs().max()) < 7 * variance / 16
    # Identity transmit -> IFFT -> time AWGN -> FFT -> hard decisions.
    bits = torch.randint(2, (256, 400, 2), generator=generator(132))
    symbols = torch.complex(
        1 - 2 * bits[..., 0].double(), 1 - 2 * bits[..., 1].double()
    ) / math.sqrt(2)
    spectrum = noise.new_zeros((256, 8192))
    spectrum[:, frame.allocation.baseband_fft_index[frame.allocation.data_grid_index]] = symbols
    received = fft_receive(torch.fft.ifft(spectrum, norm="ortho") + noise, frame.allocation)
    y = received[:, frame.allocation.data_grid_index]
    decoded = torch.stack((y.real < 0, y.imag < 0), -1)
    errors = int((decoded != bits).sum())
    theory = 0.5 * math.erfc(math.sqrt(1 / variance) / math.sqrt(2))
    assert abs(errors - bits.numel() * theory) < 6 * math.sqrt(bits.numel() * theory * (1 - theory))


def test_noise_rejects_shared_or_identical_streams():
    for gr, gi in [(generator(1), generator(1))]:
        with pytest.raises(ValueError, match="independent"):
            complex_awgn((10,), 1.0, gen_r=gr, gen_i=gi)
    for snr in [float("nan"), 10000.0, -10000.0]:
        with pytest.raises(ValueError):
            sigma2_from_esn0(snr)


def test_public_inputs_and_algebra_fixture(physical):
    config, frame = physical
    p = explicit_paths()
    with pytest.raises(ValueError, match="float64"):
        evaluate_continuous(frame, [0.1], config)
    with pytest.raises(ValueError, match="float64"):
        affine_waveform(frame, p, [0.1], config)
    with pytest.raises(ValueError, match="samples"):
        fft_receive([1j], frame.allocation)
    with pytest.raises(ValueError, match="float64"):
        dirichlet_kernel([0.0], 8192)
    fixture = load_config("configs/smoke_math.yaml")
    with pytest.raises(ValueError, match="algebra_fixture"):
        effective_matrix(p, frame.layout, fixture, 0)
    with pytest.raises(ValueError, match="algebra_fixture"):
        sample_paths(fixture, "identity_awgn", generator=generator(1))


@pytest.mark.parametrize("kind", ["missing", "overlap", "lfm"])
def test_continuous_rejects_inconsistent_layout(physical, kind):
    config, frame = physical
    if kind == "missing":
        layout = replace(frame.layout, cp=frame.layout.cp[:-1])
    elif kind == "overlap":
        layout = replace(
            frame.layout,
            cp=(frame.layout.cp[1], *frame.layout.cp[1:]),
            useful=(frame.layout.useful[1], *frame.layout.useful[1:]),
        )
        with pytest.raises(ValueError, match="layout"):
            effective_matrix(explicit_paths(), layout, config, 0)
    else:
        layout = replace(frame.layout, lfm=(0, 3840))
    with pytest.raises(ValueError, match="layout"):
        evaluate_continuous(
            replace(frame, layout=layout), torch.tensor([0.1], dtype=torch.float64), config
        )


def test_cpu_paths_do_not_probe_cuda(physical, monkeypatch):
    config, frame = physical

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU channel called CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    p = sample_paths(config, "static_multipath", generator=generator(67))
    h = effective_matrix(p, frame.layout, config, 0)
    assert h.device.type == "cpu"


def test_complex64_explicit_conversion(physical):
    config, frame = physical
    p = explicit_paths()
    h = effective_matrix(p, frame.layout, config, 7, dtype=torch.complex64)
    actual = fft_receive(receive_window(frame, p, config, 7).to(torch.complex64), frame.allocation)[
        0
    ]
    expected = h @ frame.grid[0, 7].to(torch.complex64)
    assert (
        float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(actual)) < 2e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; CPU is default")
def test_cuda_physical_equivalence(physical):
    config, frame = physical
    config = load_config(device="cuda")
    gpu = build_frame(
        torch.zeros((1, 8, 400, 2), dtype=torch.long, device="cuda"),
        frame.grid[..., frame.allocation.pilot_grid_index].cuda(),
        config,
        resolve_runtime("cuda"),
    )
    p = explicit_paths()
    p = replace(p, gain=p.gain.cuda(), delay_s=p.delay_s.cuda(), epsilon=p.epsilon.cuda())
    h = effective_matrix(p, gpu.layout, config, 7)
    actual = fft_receive(receive_window(gpu, p, config, 7), gpu.allocation)[0]
    assert (
        float(
            torch.linalg.vector_norm(actual - h @ gpu.grid[0, 7]) / torch.linalg.vector_norm(actual)
        )
        < 1e-9
    )
