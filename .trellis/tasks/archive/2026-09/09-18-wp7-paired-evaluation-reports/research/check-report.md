# WP7 independent review — PASS (2026-09-18)

Reviewer scope: approved R1–R9 and source §§16–18, 21.6, WP7, 23.3.
No source specification or independent reference implementation was edited.

## Findings fixed

- `evaluation/runner.py`: the saved synchronization example correlated
  downconverted I/Q against an RF LFM template. The template is now downconverted
  consistently, both tensors use the data precision, the configured threshold is
  honored, and the expected template start includes leading silence. Independent
  identity-channel checks pass in complex128/complex64 with the exact expected
  peak and unit normalized score (2e-6 single-precision tolerance).
- `evaluation/timing.py`: unique measured noise observations previously counted
  the whole generated pool. It now counts `min(repeats, pool) * batch`.
  Timer probes verify factorization occurs inside the measured boundary for cold
  calls and separately measured preparation for cached calls.
- `evaluation/metrics.py`: finite predictions can overflow during energy
  squaring. Nonfinite error/target energy now raises a hard failure before any
  completed metric is published; zero target energy is rejected.
- `evaluation/runner.py`: `main_simulation` now rejects selected PG depth or
  VAMP iterations other than eight before manifest IO. Supplementary development
  experiments retain their explicit label and actual iteration metadata.
- `reporting/validation.py`: the new streamed validator initially rejected all
  negative contractions. WP5 permits finite unresolved small negative `c`;
  reporting now accepts that valid diagnostic without changing detector math.
- Coordinator/reporter review fixed unbounded diagnostic JSONL retention and
  visual metadata/zero/error-bar/template-start issues. Every diagnostic line
  is validated while only small layer moments/ranges and cell counters remain.
  Final reporter tests include a guard forbidding whole-file diagnostic reads.

## Independent verification coverage

- Frame counts retain eight planned blocks and 6400 bits. NaN, Inf and injected
  factorization failures at blocks 0/3/7 preserve exact failed IDs, 5600
  conditional-success bits, unavailable full rates/CI, and no paired conclusion.
- Hand-computed integer counts, aggregate energies and goodput; independent
  fixed-seed bootstrap reconstruction with unequal frame denominators/shared
  indices; true zero rates and independent-frame FER bounds.
- Prepared PG and SVD VAMP independently match DensePGVAMP and Cholesky VAMP on
  multiple observations and a rank-deficient batch. Existing cache tests check
  both dtypes and H/noise/parameter/settings/training-mode invalidation. Invalid
  observations remain rejected. Full regression covers gradients/oracles.
- CPU paths forbid CUDA queries; actual CUDA numerical tests require hardware.
  B1 and B2 timing stay separate. Preparation, amortization, CPU process lifetime
  peak, CUDA allocated peak and estimated matrix storage retain distinct labels.
- Reports consume persisted results with detectors/training disabled; malformed
  counts/provenance/hashes, missing paired coverage, failed populations and
  duplicate/incompatible training seeds have dedicated regression coverage.

## Fresh full-dimensional CLI evidence

`runs/wp7-cli-review-v2/receipts.json`: 15/15 commands exit 0.
`runs/wp7-cli-review-v2/report-receipts.json`: 2/2 commands exit 0.
These contain exact argv, cwd, stdout/stderr and elapsed durations.

The run retains 512 grid/400 data/8192 FFT/2048 CP/eight blocks/T8, uses CPU
complex128 and four threads, and exercises inspect/help, generate, physical
waveform audit, two distinct training seeds (two updates each), two evaluations,
both standalone benchmark modes, unlabeled materialize/infer, single-run report
and multi-seed report. Test cells are moderate Doppler at 0/10 dB, one independent
frame per cell; the same physical frame across SNR is not independent evidence.

The coordinator visually reviewed all 10 single-run figures and representative
combined-report figures: labels/zeros/CI statements/timing metadata and actual
synchronization marker are correct, with no remaining visual finding.

## Receipts and limitations

Final verification: Ruff lint PASS; Ruff format PASS (98 files); mypy PASS
(66 source files); full pytest **478 passed, 7 CUDA skipped in 95.52 s**.
Additional Ruff lint/format checks for `scripts/` PASS (5 files), recorded in
the same final receipts; no code edits were needed.
Independent stdlib audit PASS for both new evaluation bundles, rehashing all
13 published artifacts per bundle and independently recomputing all six cells
from their frame rows. Authority and all four oracle files match the baseline.
No unresolved implementation or visual finding remains.

Final static/full-suite receipts: `check-receipts.json`.
Independent stdlib persisted-artifact audit: `check-artifact-audit.json`.
Source/test/artifact fingerprints and unchanged authority/oracles:
`check-fingerprints.json`.

This validates the software and the small real-dimensional loop. Full main
training, a powered full SNR sweep, CUDA hardware measurements, controlled
ablation and real-world superiority remain unexecuted. Timing used only three
repeats in this acceptance exercise; one-frame bootstrap intervals are explicitly
unstable and cannot establish low BER or a PG performance advantage.
