# WP8 independent review — 2026-09-18

## Findings (fixed)

- File: `docs/REPRODUCING.md`.
  Issue: the sequential CPU recipe resumed a completed run with the unchanged
  `max_steps=300`; actual strict resume rejects a budget already reached.
  Fix: removed that failing command and supplied a concrete `max_steps: 600`
  overlay, explicit resume command and interrupted-run budget constraints.
  Loaded the overlay and verified all other settings equal CPU development defaults.
- Reviewer-only evidence helpers: fixed Windows implicit text decoding and
  formatting, and corrected the review harness's receipt-count expectation to
  the actual split of 15 evaluation commands plus 2 reports. One E501 finding
  and its successful rerun are retained in `check-final-receipts.json`.
  No product code, tests, mathematical references or thresholds changed in review.

## Findings (not fixed)

None outstanding in the approved implementation scope. Main-scale training,
full SNR sweeps, sufficient multi-seed statistics, CUDA numerical/timing results,
and noisy synchronization success-rate experiments remain unexecuted by design.
The review does not convert the bounded CPU checks into performance evidence.

## Verification

- Full regression: **486 passed, 7 CUDA hardware skips**, 111.41 seconds.
  `.venv/Scripts/python.exe -m pytest -q --basetemp=runs/wp8-check-full -ra`.
- Lint: **pass**, Ruff `src tests scripts`; also all three reviewer helpers.
- Format: **pass**, 105 product/test/script files and 3 reviewer helpers.
- TypeCheck: **pass**, mypy, 67 source files.
- Whitespace: `git diff --check` **pass**.
- Exact argv/start/end/exit/stdout/stderr and stream hashes are embedded in
  `check-command-receipts.json` and `check-final-receipts.json`. Ignored log files
  are supplementary copies; the versioned receipts retain their complete text.
- Fresh actual CPU `demo-frame` runs passed in both complex128 and complex64:
  `runs/wp8-check-demo-complex128` and `runs/wp8-check-demo-complex64`.
  Both save 91776-sample TX, 93248-sample RX, eight full 512-grid/400-data systems,
  8192-sample FFT windows and CP2048, 6400 data bits, distinct nonzero path scaling.
- Independent NumPy saved-array audit (safe Torch deserialization only): all
  eight waveform FFT/H checks, pilot cancellation, noise residuals, TX/CP mapping,
  affine CP support endpoints, RF/IQ phase and complete RX length, all-lag direct
  correlation, earliest LFM support and every declared artifact hash passed.
  Max waveform/H relative error is 4.5908142877237e-12 (double reference gate
  remains <1e-9). Single-precision pilot/noise normwise errors are approximately
  1.05e-7 / 2.81e-7, below the existing justified 2e-6 gate.
- Selected LFM template peak is 2592 with score ~0.32656; earliest physical LFM
  support is ~2379.404. Noisy ideal-IQ uses oracle windows, never that peak.
  Random-stream reconstruction/deterministic rerun is separately covered by
  `test_demo.py`, using Torch generators, not claimed as a NumPy RNG replay.
- Direct visual inspection of all four demo PNGs passed: complete waveform,
  noiseless real-RF synchronization, full 400x400 channel and received observation
  are readable, correctly labeled, without blank/clipped or placeholder panels.
  Hashes and numerical audit are in `check-demo-artifact-audit.json` and
  `check-delivery-audit.json`.
- Reused this session's fresh coordinator WP6 11/11 and WP7 15/15 + 2/2 receipts,
  independently rehashed 54 system artifacts, reviewed the saved-system NumPy
  audit (1.1310709079623577e-11), exact two updates/16 parameters, paired inputs,
  counts and checkpoint/inference checks. Reviewed saved-result stdlib recount
  and coordinator inspection of all ten report figures. No expensive harness
  rerun was needed because review changed only documentation.
- Source §§2–17 and 20–23 checked against demo and documentation; PG centered
  residual, safe majorizer, actual divergence/variance, message protection,
  threshold/2T rules and dense complexity remain correctly described.
- Checked delivery matrix categories against actual test definitions and key
  assertions (CP support, FFT/pilot/noise, PSD/Loewner, Jacobian/covariance,
  resume, timing and failure semantics). All 37 fully qualified test anchors
  resolve. 47 local links resolve across README, status, validation, all new
  guides/matrix and WP8/backend specs. The new WP8 interface spec agrees with code.
- All five frozen authority/reference SHA-256 values match baseline. Final source,
  test, document, spec and evidence hashes are recorded in `check-source-hashes.json`.

## Acceptance

A1–A6 pass for bounded software delivery. No remaining implementation blocker.
Source specification and independent oracles remain unchanged. User-owned
`docs/WP7_EXAMPLE_REPORT.md` and `output/` were not edited or included in this review.
No commit or archive performed by this reviewer.
