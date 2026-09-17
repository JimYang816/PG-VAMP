# WP2 physical channel interface

The authority is CODEX_ENGINEERING_SPEC.md §§5–9. These interfaces implement
ideal complex I/Q, oracle timing, perfect CSI and uncoded unit-energy QPSK.

`sample_paths(config, scenario, generator=...)` returns one frame's complex128
gains and float64 sorted initial delays and path-dependent time scales. Its
explicit generator determines the device. Sampled gains have unit total energy;
explicit `PathParameters` preserve the supplied channel scale. No H normalization
or ICI truncation is performed.

`validate_cp_support(paths, layout, config, window_offsets_s=...)` checks both
endpoints for every path and block. The lower CP boundary is included; the useful
duration upper boundary is excluded. Invalid windows raise a block/path/endpoint
error. This implementation rejects invalid frames and performs no resampling.

`receive_window(frame, paths, config, block, window_offset_s=...)` evaluates the
finite transmit frame independently using direct Fourier exponentials and a
continuous local chirp. It applies each affine path at the actual transmit time,
then downconverts on the absolute physical receive time axis. No matrix/kernel is
imported by the continuous evaluator. `affine_waveform` can also evaluate arbitrary
recording times; `arrival_offset_samples` removes external recording padding
before propagation and downconversion. Continuous waveforms are complex128;
single precision is an explicit caller conversion after reference evaluation.

`effective_matrix(..., block, window_offset_s=...)` builds all 512×512 entries
in complex128 with the 8192-point finite-sum kernel. An optional complex64 output
conversion is explicit. At offset Delta each path/column acquires the additional
phase `2*pi*(q*df + epsilon*(fc+q*df))*Delta`. The same actual window must be used
for the waveform and H. Both window APIs validate CP support across the frame.

`fft_receive(samples, allocation)` applies an ortho FFT and explicit baseband
indices. `preprocess(H_grid, Y_grid, pilots, allocation, sigma2, ...)` selects
the 400 data rows, removes true `H_DP @ pilots`, and returns H_DD, y, sigma2
and the caller's window offset. Batch dimensions must match exactly. The caller
is responsible for supplying matching window metadata and observations; bare
tensors cannot prove their provenance. Estimated CSI is explicitly rejected.

`complex_awgn(shape, sigma2, gen_r=..., gen_i=...)` uses separate real/imaginary
generator streams, rejecting identical states, to produce CN(0,sigma2).
`sigma2_from_esn0` assumes Es=1. Noise never depends on measured received power.
The audit retains means, variances, selected-bin covariance and counted hard-QPSK
errors after the actual identity IFFT/time-AWGN/FFT chain.

Run the dedicated audit with a fresh output directory:

```powershell
$env:MKL_THREADING_LAYER='TBB'
.venv/Scripts/python.exe scripts/run_wp2_audit.py --config configs/cpu_dev.yaml --output runs/wp2-audit-new
```

It checks two physical frames, all eight blocks plus shifted edge-block windows,
saves seeds/paths/support/environment/source fingerprints, and reports separate
double/single precision errors. Per-frame tensor artifacts retain grid/bits/paths,
full H, independent IQ samples, time noise and processed observations for every
window. The report hashes these artifacts and the resolved configuration and
records `SNR_rx = 10*log10(||H_DD||_F^2/(400*sigma2))`. Load artifacts with
`torch.load(path, map_location="cpu", weights_only=True)` for safe inspection.
Malformed or overlapping frame intervals are rejected before evaluation.
It is not a dataset backend, detector benchmark,
training run or complete system smoke. Those remain later work packages.
