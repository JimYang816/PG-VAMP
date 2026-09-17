# WP1 independent check review

Outcome: passed after two local fixes. No unresolved in-scope product finding.
Source: execution specification §§0–7, 9, 10.3/10.5/11, 14.6,
15.2–15.3, 18–24; PRD/design/implement/check manifest and backend/guides specs.
All WP1 product modules, tests and audit script were inspected. The review
stays within §22 WP1; no channel, effective matrix or WP2 receiver was added.

## Findings (fixed)

1. `src/pgvamp_ofdm/receiver/synchronization.py`: nonzero representable sample
   energies could multiply/square to zero before additive epsilon, silently
   returning zero scores for matched tiny signals. Reproduced before repair:
   recording of eight constant samples and four-sample identical template,
   float64 amplitude 1e-100 / float32 amplitude 1e-15, all five scores zero.
   Explicitly reject nonzero sample-energy, correlation-magnitude-squared or
   positive energy-product underflow. Preserve epsilon=finfo(real_dtype).tiny,
   exact-zero window semantics, strict threshold and unmodified scores.
   Added four rejection cases and one representable subnormal-product case
   showing the additive epsilon formula remains intact (`tests/test_lfm.py`).
2. `src/pgvamp_ofdm/modulation/allocation.py`: validate skipped non-tensor index
   fields, allowing AttributeError later; FFT/carrier scalar metadata did not
   enforce integers. All index fields now require tensors, and the two scalar
   fields require actual ints. Four malformed-input regressions added to
   `tests/test_allocation.py`.

Main was informed to sync the WP1 long-term error contract and retrospective.
No shared public signatures, numerical thresholds, interfaces or WP0 code changed.
Final README, IMPLEMENTATION_STATUS, VALIDATION, backend index and
wp1-waveform-contract were subsequently inspected: final 91/166 counts,
implementation-versus-review receipts, repair semantics and WP2/CUDA limits
agree with code and results. No remaining documentation/spec mismatch found.

## Findings (not fixed)

None. No design/public-boundary issue requires a product decision.
Floating-point mathematical peak ties can differ by roundoff; exact computed
ties use first argmax. This is the declared implementation limit rather than
an oracle-assisted earliest-path correction. CUDA execution is unavailable;
it is reported as skipped, not accepted by CPU mock tests.

## Acceptance coverage

| Criterion | Review evidence |
| --- | --- |
| A1 | Explicit independent four-point mapping, all bit dtypes/complex precisions, sign/axis ties, energy and malformed labels. |
| A2 | Handwritten full pilot list, 400/64/47/1 partition, ordered physical/grid/FFT indices, null mapping, independent seed replay and malformed allocation checks. |
| A3 | Noncontiguous multibatch last-axis FFT/Parseval, exact CP copy, 464/8192 power, all 512 recovered bins and 6400 data bits; NumPy FFT from persisted real samples is independent of torch FFT. |
| A4 | Default 91776 samples/0.956 s, exact spans across eight blocks, silence/padding, non-four-divisible leading time 1921, absolute carrier phase and unchanged RNG state. |
| A5 | Independent scalar Tukey at alpha 0/0.1/1, arange endpoint and phase increments, fixed configuration power/normalization, label-invariant LFM; float64 phase then explicit single-precision conversion. |
| A6 | Independent short direct correlation, full-LFM first/interior/last valid shifts, batched inputs, exact silent windows, no-signal/threshold rejection, real/complex noise; cancellation/overflow/underflow errors. Saved long-waveform correlation also checked against NumPy vdot at seven lags for each precision and real/analytic mode. |
| A7 | Both precision audits saved tensors/basic types, PSD/correlation arrays, environment/config/seeds/source hashes and PNGs. All 8 artifact hashes and 21 package hashes per audit verified. PSD independently recalculated from saved samples. All four final PNGs opened and visually checked. |
| A8 | Full CPU suite and both CLI entries, lint/format/type checks; independent full-size CPU double/single audit; all 24 archived WP0 source/config/test fingerprints unchanged; no WP2 path introduced. |

## Verification

- Targeted: **91 passed, 1 skipped** in 1.25 s.
- Full regression: **166 passed, 2 skipped** in 12.31 s.
- Lint: pass. Format: pass (30 files). TypeCheck: pass (21 source files).
- Both inspect-config entry points and git diff --check: pass.
- Fix-focused run: 53 passed in 1.12 s.
- Skips: CUDA hardware unavailable, tests/test_devices.py and tests/test_waveform.py.
- Exact command arguments, stdout/stderr/exit status and final source fingerprints:
  `check-validation.json`, reproducible runner `check-run.py` (fresh directory names
  required before rerunning; prior evidence is intentionally not deleted).
- Artifact verification command (exit 0):
  `$env:MKL_THREADING_LAYER='TBB'; .venv/Scripts/python.exe .trellis/tasks/09-18-wp1-modulation-frame-lfm/research/check-artifacts.py`.
  Numeric output is persisted in `check-artifacts.json`.
- Environment: Windows 11, Python 3.13.9, torch 2.12.0+cpu, NumPy 2.3.5,
  CPU threads 4, deterministic enabled, MKL_THREADING_LAYER=TBB.
- Audit paths: `runs/wp1-check-audit` and `runs/wp1-check-audit-complex64`.

## Numerical evidence and visual inspection

Both audits: transmitted 91776 / recorded 92740 samples, offset 964,
injected and detected template start 2884, no-channel origin error 0,
6400 data bits and zero recovered bit errors. These are transmit-only results.

Double precision saved-waveform independent NumPy recovery max error:
9.524226793575906e-12. Single precision NumPy/float64 replay recovery error:
2.0432709204151548e-7; the production torch/complex64 audit error is
2.6656007889869215e-7. Different FFT precisions explain the distinct metrics;
neither substitutes for the other's precision label.

Double precision PSD integration errors for frame/useful block 0/LFM:
2.776e-17 / 2.082e-17 / 6.939e-18. Corresponding outside ±[21,27] kHz
fractions: 0.0006655853 / 0.0007244081 / 0.0025107553. Nonzero leakage is
explicitly reported, with no OFDM transmit window introduced.

Double direct-correlation max error at sampled lags: real 1.776e-15,
analytic 6.772e-14; single max 3.378e-7. Twelve real and twelve complex
noise recordings per audit have zero detections at 0.1. Double real/complex
maximum noise scores 0.00405995 / 0.00187936; single 0.00352059 / 0.00213683.
These fixture counts are not universal false-alarm probabilities.

PNG inspection: readable three-panel PSD (frame/useful/LFM), original sample
count and zero-padded FFT count, kHz and two-sided dB/Hz axes, positive-half
display with negative mirror included in integration, Tukey/OFDM distinction.
Correlation has sample/second coordinates, threshold and injected template
start annotation. No clipping/layout issue obstructs the WP1 evidence.

## Limits

CUDA hardware not tested. No affine physical channel, CP path support,
effective H, detector, training, BER/SNR sweep or full-system smoke executed.
No claim of earliest-path recovery or realistic ADC frontend. All WP0
historical failures remain in original receipts; this check did not overwrite them.
