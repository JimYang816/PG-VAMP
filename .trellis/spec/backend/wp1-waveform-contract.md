# WP1 modulation, transmit frame and synchronization contract

Authority: docs/CODEX_ENGINEERING_SPEC.md §§3–5, 7, 19–23.
This contract describes WP1 interfaces, not WP2 channel/receiver acceptance.

## 1. Scope / trigger

Use when changing modulation, discrete OFDM/LFM frame generation, matched
correlation, or when a later physical model consumes their outputs.
WP0 independent references remain separate numerical implementations.

## 2. Signatures

```text
build_allocation(config, device="cpu") -> Allocation
map_grid(data[...,400], pilots[...,64], allocation) -> grid[...,512]
modulate_grid(grid, config) -> OFDMSymbols
make_lfm(config, runtime) -> LFMTemplate
build_frame(data_bits[B,M,400,2], pilot_symbols[B,M,64], config, runtime)
    -> FrameWaveforms
pad_recording(waveform[...,N], arrival_offset_samples) -> recording
normalized_correlation(recording[...,L], template[N], threshold=0.1)
    -> SyncResult
```

The audit entry is `python scripts/run_wp1_audit.py --config
configs/cpu_dev.yaml --output runs/wp1-audit`. It is a transmit/synchronization
audit; it is not `smoke_system`, dataset generation or detector evaluation.

## 3. Contracts

- Fixed labels: c=2*b_R+b_I; bits order R then I; classes 00,01,10,11.
  Hard decisions use negative sign bits; zero coordinates choose positive signs.
- Allocation keeps sorted physical q and explicit grid/baseband/passband indices.
  Defaults are 400/64/47/1, q+256, q%8192 and 2048+q. Preserve all 400 independent
  data symbols; no conjugate-symmetric input restriction.
- `FrameWaveforms` retains analytic/real `[B,Nframe]`, grid, OFDM, LFM, layout
  and allocation. Layout intervals are half-open and relative to transmit origin.
  Default LFM is [1920,5760); useful block m starts 9856+m*10240;
  tail silence is [89728,91776). Extra recording offset never changes this layout.
- Frame generation consumes explicit labels/pilots, not global RNG.
  Pilots are separately seeded unit QPSK; no per-frame measured-power normalization.
- OFDM uses ortho FFT and exact tail-copy CP. Its passband carrier uses the
  global transmit-frame time axis; LFM is already passband and uses its declared
  local chirp phase. Do not modulate the LFM twice or reset OFDM carrier per block.
- LFM time/phase and carrier phase are evaluated in float64 then explicitly
  cast for complex64 output. Tukey is symmetric u=n/(N-1); fixed amplitude
  makes analytic mean power equal (Nd+Np)/Nw. Record real RF power separately.
- `SyncResult` contains scores, peak_start/peak_score, detected,
  candidate_arrival/candidate_valid, threshold and epsilon_sync.
  All peak positions are template starts. An invalid candidate is -1 with
  false validity; maximum peak is not necessarily the earliest physical path.
- Correlation uses valid lags only and strict score > threshold. FFT linear
  convolution is cropped starting at N-1. Exact zero-energy windows score zero.
  CPU never probes CUDA; paired precision/device and finite inputs are required.

## 4. Validation & error matrix

| Input | Required behavior |
| --- | --- |
| Algebra fixture passed to physical API | ConfigError, no fabricated waveform |
| Invalid bit/class/QPSK value, shape or finite check | Readable ValueError |
| Incorrect allocation, overlap, wrong pilot positions | Reject before mapping |
| Passband touching DC/Nyquist/negative mirror | Reject unsupported RF recovery geometry |
| Runtime/config dtype or device mismatch | Reject, no implicit migration |
| LFM window with zero mean-square energy | Reject normalization |
| Short recording, empty/zero-energy template, bad threshold | Reject correlation |
| Silent recording/window | Finite zero score; no detection |
| Unrepresentable energy/correlation or lost positive window energy | Readable error, no score clipping |
| Positive sample energy, correlation square or energy product underflows to zero | Reject explicitly; adding epsilon is not evidence that the input was represented |
| Allocation field replaced by a non-tensor or non-integer FFT/carrier metadata | Readable ValueError before accessing tensor attributes |

## 5. Good / base / bad cases

- Base: CPU complex128, 96 kHz, 8192 FFT, 8 OFDM blocks, 91776 transmitted samples.
- Good: prepend 13 zeros to the complete waveform; recovered LFM template start
  shifts by 13 while the original transmitted samples/layout remain unchanged.
- Bad: pass a matched peak to WP2 as an unconditional earliest-path estimate,
  or construct H for an oracle window while taking y from an estimated window.

## 6. Tests required

Run allocation/QPSK/waveform/LFM tests and the full WP0 regression suite.
Assert explicit pilot list, independent four-point labels, Parseval, CP equality,
actual real-waveform FFT recovery, non-integral carrier phase at a changed block
start, LFM power/window/phase, direct-correlation equivalence, boundary lags,
silent windows, pure-noise cases, explicit single precision and conditional CUDA.
Include float64/float32 energy-product underflow and partial sample-energy underflow.
Also retain a small but representable case that verifies the exact additive
epsilon formula: do not replace rejection checks with amplitude normalization
or remove epsilon's legitimate effect at small scales.
Audit PSD integration, band-energy definition, raw correlation arrays and PNGs.
Preserve actual commands/environment/seeds/hashes; CUDA skip is not GPU acceptance.

## 7. Wrong vs correct

Wrong: a native 8192-point FFT has nearly zero outside occupied bins, therefore
the finite waveform has no sidelobes. Correct: inspect a sufficiently dense
periodogram (zero padding only for spectral sampling), normalize with the original
segment length, integrate using Fs/Nfft, and report energy outside both RF bands.

Wrong: FFT roundoff in a silent window divided by epsilon creates a high score.
Correct: identify exact zero windows and return zero, while rejecting genuinely
unrepresentable positive-energy cases instead of hiding them by clipping.

Wrong: check only isfinite after squaring/multiplication; underflowed zero passes.
Correct: check positive-to-zero loss at the sample, numerator and denominator
product boundaries, while accepting mathematically zero correlations/windows.
