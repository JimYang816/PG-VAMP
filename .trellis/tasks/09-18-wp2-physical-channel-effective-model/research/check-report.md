# Independent WP2 review — 2026-09-18

## Findings (fixed)

- `waveform/continuous.py` silently accepted missing or overlapping CP/useful
  spans because `zip` truncated or later segments overwrote earlier results.
  It now verifies canonical frame spans against the physical config.
  `channel/validity.py` also verifies each block's prescribed absolute start,
  rejecting duplicated/overlapping blocks before returning a valid effective H.
  Three new parameterized regressions cover missing, overlapping and wrong LFM spans.
- `scripts/run_wp2_audit.py` retained only summary JSON, preventing independent
  inspection of actual numerical outputs, and omitted source §9 receive SNR.
  It now saves tensor/basic-type frame artifacts containing grid, bits, paths,
  all 20 H/IQ/noise/processed observations; hashes artifacts and resolved config;
  and records each window's SNR_rx using H_DD and the known sigma2.
- Updated model/validation/status docs for actual artifacts and final results.
  Main coordinated the corresponding backend contract update; signatures match.

## Findings (not fixed)

No remaining in-scope code or design blocker. Existing root-wide Ruff errors in
hooks/Trellis/archive tooling remain out of scope (implementation receipt records
192 findings); all product source/tests and WP1/WP2 audit scripts pass. CUDA
numerical tests remain skipped because hardware is unavailable; this is not GPU
acceptance. WP3 dataset/replay/split work and later detectors/training/system smoke
are deferred by the approved work-package boundaries, not silently claimed complete.

## Acceptance review

| Gate | Evidence |
| --- | --- |
| A1 | All five seeded scenarios, sorted delays, sample count/energy, identity and static diagonal tests. Sampling follows source CN power-delay profile and path-dependent uniform epsilon; explicit gain scale is preserved. |
| A2 | Actual-window CP inequalities, positive/negative epsilon, late-frame failure, exact lower inclusion/upper exclusion and layout rejection tests. All reference/matrix windows share the same time offset. |
| A3 | Full 8192/512 Fourier reference versus full H for unequal nonzero paths; 20-window fresh audit max relative error 8.152338913635682e-12. No H/Dirichlet import in independent waveform path, no ICI truncation or normalization. NumPy direct exponentials at four samples per window agree within 2.89e-13 absolute. |
| A4 | Entire default integer frame equals WP1 analytic samples; fractional chirp/CP, segment boundaries, silence/exterior and recording-padding tests. Canonical span validation prevents silent missing/overlapping segments. |
| A5 | Correct centered-grid data rows and pilot columns; independent artifact reconstruction of clean/noisy cancellation for all windows. Pilot leakage energies 6.68–16.32 confirm nontrivial cancellation. Perfect CSI enforced, mismatched batch axes rejected. |
| A6 | Fixed independent RNG states, real/imag means/variances/covariance, 33,554,432 noise samples and 4096 FFT windows; covariance and BER pass declared count-based guards. 11604/204800 bit errors vs theory 0.05649530 at Es/N0=4 dB. SNR_rx recomputed independently from saved H_DD. |
| A7 | Targeted 28 passed/1 skipped; full 194 passed/3 skipped; product Ruff and format pass; mypy all 29 source files pass; fresh CPU audit and safe artifact/hash reconstruction pass. complex64 max relative error 4.6309497747643036e-7 < 2e-6. |

## Verification receipts

- `check-validation.json`: initial independent checks before review fixes, preserved.
- `check-final-validation.json`: final command arrays, stdout/stderr/status and environment.
- `check-physical-audit.json`: tracked copy of final actual audit JSON.
- `check-artifact-verification.json`: per-window independent NumPy/Torch checks,
  product/test source hashes, confirmed audit package/source/config/artifact hashes.
- `check-extra-verification.json`: expected nonempty-output rejection and git diff check.
- `run_check.py`: standalone review runner, using fresh output paths; it intentionally
  refuses to overwrite audit evidence on a second identical invocation.
- `runs/wp2-check-audit-final-01`: final full-size tensors and resolved config (Git-ignored).

No equation or numerical tolerance was relaxed. No commit or archive was performed.
