# WP2 physical channel and receiver contract

Authority: docs/CODEX_ENGINEERING_SPEC.md §§5–9, 19–23. This interface guide
does not change the physical equations or replace the complete source tests.

## 1. Scope / trigger
Read when modifying or consuming physical paths, continuous waveform evaluation,
effective matrices, receive FFT or pilot elimination. Dataset persistence and
detectors remain later work packages; physical acceptance does not validate them.

## 2. Signatures
```text
PathParameters(gain, delay_s, epsilon, scenario).validate()
sample_paths(config, scenario, *, generator) -> PathParameters
validate_cp_support(paths, layout, config, *, window_offsets_s=None) -> [M,L,2]
effective_matrix(paths, layout, config, block, *, window_offset_s=0,
                 dtype=complex128) -> [512,512]
evaluate_continuous(frame, times_s, config, *, chunk_size=512) -> [B,K]
affine_waveform(frame, paths, times_s, config, *, arrival_offset_samples=0,
                downconvert=True, chunk_size=512) -> [B,K]
receive_window(frame, paths, config, block, *, window_offset_s=0,
               chunk_size=512) -> [B,8192]
fft_receive(samples[...,8192], allocation) -> [...,512]
preprocess(h_grid, y_grid, pilots, allocation, sigma2,
           *, csi_mode="perfect", window_offset_s=0) -> DetectionInput
sigma2_from_esn0(esn0_db) -> float
complex_awgn(shape, sigma2, *, gen_r, gen_i, dtype=complex128) -> Tensor
```

## 3. Contracts
- Paths are one physical frame: complex128 gain[L], float64 delay_s/epsilon[L],
  shared device. Sampling normalizes total path energy; explicit physical gains
  preserve their input propagation scale. No H column/Frobenius normalization.
- Frame metadata and FFT indices come from WP1. Physical q, centered grid index,
  and baseband FFT bin are different quantities. Complete H uses all 512×512
  entries and the 8192-length finite sum; no fixed ICI band or random deletion.
- Continuous reference directly evaluates the selected block's Fourier series
  or local analytic chirp, with silence/frame exterior zero. It may share layout
  and allocation metadata, never the effective-matrix kernel being verified.
- Actual receive-window offsets apply to both CP support and H phase. Validate
  every path and block: lower bound -Tcp is inclusive, upper bound Tu exclusive.
  Integer recording offsets are removed before physical propagation/time evolution.
- Ideal IQ uses the absolute receive clock for downconversion. Shifted H per path
  has the additional phase exp(j*2*pi*(q*df+epsilon*(fc+q*df))*Delta).
- Construct physical reference and matrix in double precision; explicitly cast
  only afterward. CPU is default; no implicit device migration or CUDA probing.
- preprocess requires matching batch axes for H/Y/pilots, selects data rows and
  data/pilot columns from allocation, subtracts H_DP p and returns H/y/sigma2/
  window_offset_s. It does not estimate CSI or infer that its inputs share timing.
- Noise sigma2 is E|w|², each quadrature has sigma2/2. Use explicit independent
  random streams and unit-symbol Es/N0; never scale from measured frame power.

## 4. Validation and error matrix
| Condition | Required behavior |
| --- | --- |
| Invalid path scenario/shape/precision/device or nonfinite value | Readable ValueError |
| Unsorted/negative delay or nonpositive affine slope | Reject before propagation |
| Static/identity scenario with nonzero epsilon | Reject |
| CP crossing for any path/block/actual window | Reject with block/path/endpoint details |
| Missing, inconsistent or configuration-mismatched frame layout | Reject before silently dropping/reassigning waveform segments |
| Mismatched H/Y/pilot batch axes or allocation device | Reject, no implicit broadcasting |
| estimated CSI | Explicit unsupported error |
| Nonpositive/nonfinite/unrepresentable noise variance | Reject |
| Shared or identical-state real/imag RNG streams | Reject correlated quadratures |
| Algebra fixture at physical API | Reject via physical configuration contract |

## 5. Good / base / bad cases
- Base: CPU complex128, 8192 FFT, 512 grid, 400 unknowns, CP 2048, 8 blocks.
- Good: two distinct nonzero path epsilons, valid first/last block windows,
  independent received Fourier/chirp samples followed by actual FFT.
- Bad: validate H against H@X as the supposed waveform, or shift a receive window
  while leaving H and CP checks at the ideal timing.

## 6. Required tests
Static diagonal and identity limits; frequency-dependent path shift; full-size
nonzero unequal epsilon waveform/H relative error <1e-9; CP boundary/last-block/
shifted-window failures; integer and fractional piecewise times; significant pilot
leakage before cancellation; real/imag and FFT noise variance/cross covariance;
identity QPSK BER with binomial sample-count tolerance. Preserve WP0/WP1 regression.
Record fixed seeds, counts, device/dtype, actual commands/results and source hashes.
Complex64 is separately reported; missing CUDA hardware is a skip, not acceptance.

The dedicated `scripts/run_wp2_audit.py --config configs/cpu_dev.yaml --output
runs/<fresh-directory>` records two frames and 20 actual windows, plus noise
statistics and counted identity-AWGN BER. Persist grid/bits/paths and each
window's H, independently generated IQ samples, noise and processed observations
as tensor/basic-type artifacts. Hash them and use `torch.load(weights_only=True)`
for independent FFT/cancellation/SNR reconstruction. Record SNR_rx from H_DD and
known sigma2; summary JSON alone is insufficient for this replay check.

## 7. Wrong vs correct
Wrong: two Generator objects necessarily produce independent quadratures.
Correct: identical generator states produce the same random sequence; enforce
independent streams and test empirical quadrature/cross-frequency covariance.

Wrong: max initial delay < CP establishes model validity for the whole frame.
Correct: epsilon changes effective delay with absolute block time; test both
endpoints for every block and recheck after every receive-window shift.
