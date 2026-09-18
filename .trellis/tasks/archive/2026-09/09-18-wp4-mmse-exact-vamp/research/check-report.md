# WP4 independent review — 2026-09-18

Disposition: WP4 A1–A7 pass on the available CPU environment. No remaining
product finding or review blocker. No reference numerical code was modified.
No commit, task metadata change, archive, or later work package was performed.

## Findings fixed

- `tests/test_detector_inputs.py`: the shared physical sample was mutated by
  MMSE's label test before VAMP's baseline was evaluated. VAMP therefore compared
  zero labels with the same zero labels in its changed-target branch. Reload the
  original dataset item at the start of each detector iteration. Final full
  regression includes this fix. An additional independent physical receipt flips
  the original symbols/bits and removes them, comparing hashes of **all** result
  tensors for each detector, including VAMP probabilities.

## Findings not fixed

None. CUDA availability is a verification limitation, not a passing GPU result.
Production PG-VAMP, training, full-system smoke, performance comparison, timing,
and evaluation failure denominators remain the agreed later-WP scope.

## Acceptance review

| Gate | Independent review and evidence |
| --- | --- |
| A1 | Full-H Cholesky MMSE follows the source normal equations and returns raw linear output with None probabilities. Independent NumPy solves across three seeded complex 7x7 systems have maximum error 3.52e-16; suite also covers residuals, diagonal, zero/rank-deficient mix, ties and no mutation. |
| A2 | Reviewed conjugation, centered SVD innovation, separate alpha2/c sums, gamma1 relation, layer-entry diagnostics and final posterior. Existing independent Cholesky comparisons pass at N=8/16/32, T=8/32. Additional NumPy solve plus direct four-point enumeration (no production or reference helper) checks every layer state for three seeds at N=7,T=8; largest error 2.78e-15, with initial linear condition numbers 4.11–4.67. |
| A3 | Read candidate eligibility, representability guard before quotient, pre-reciprocal cap, unchanged candidate mean, true alpha and previous-message retention. Float32/64 underflow, reciprocal, numerator/quotient overflow and finite backward regressions pass. QPSK enumeration and gradcheck pass. |
| A4 | Zero, rank-deficient, mixed and weak-channel branches pass. Positive-sum no-information bound is conditional on normal arithmetic of computed spectral factors, not an SVD error theorem; no fixed epsilon cutoff is introduced. The 0/1e-160/1e-20 cases agree with the untouched Cholesky oracle. Complex scale invariance, one reduced SVD per detect, no cross-call cache, zero learned parameters and forbidden API checks pass. |
| A5 | Shape/dtype/device/finite/noise errors and contextual hard failures pass. complex64 tolerance remains 2e-6/2e-5 on bounded fixtures; complex128 remains 1e-9/1e-8. CUDA skips are explicit. |
| A6 | Fresh pytest-generated physical data supplies 400x400 H and six nonzero unequal path scalings. Independent receipt records original/flipped/absent-label cases, unchanged identical input hashes, sample ID and all output hashes for both detectors. Outputs finite; VAMP protection counters zero for this sample. |
| A7 | Final full regression 313 passed / 5 CUDA skipped in 31.64 s; Ruff lint/format and mypy pass, both config inspections pass. Source SHA unchanged. Final source fingerprint includes the test fix. |

## Executions and artifacts

- Entry: `.venv/Scripts/python.exe .trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-validation.py`.
  `check-validation.json` retains exact subprocess argv, stdout/stderr, exit codes,
  elapsed time and hardware. Final pytest used a fresh ignored `runs/` directory.
- Physical entry: `.venv/Scripts/python.exe .trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-physical.py`.
  Exit 0; `check-physical.json` retains the fresh manifest and full output hashes.
- `check-numerics.py` / `check-numerics.json`: NumPy solve/enumeration audit.
- `check-source-hashes.json`: final SHA-256 of product, tests, source specification
  and supplementary config. Earlier implementation hashes precede the reviewer fix.
- Initial independent pytest also passed 313/5 in 32.83 s. Its root-local temporary
  directory was relocated to `runs/pytest-wp4-check-initial-20260918` after verifying
  both resolved paths remain in this workspace. Final receipts use their original
  fresh runs paths and are unaffected.

Environment: Windows 11 build 26200; AMD64 Family 25 Model 116 Stepping 1;
12 logical CPUs; four test/physical threads; Python 3.13.9; PyTorch 2.12.0+cpu;
MKL_THREADING_LAYER=TBB. CUDA unavailable, zero CUDA devices. Five skips are the
two detector parity cases and prior device/waveform/WP2 CUDA cases. Installed
Ruff executable is used explicitly; no environment failure is relabeled a pass.

Source authority SHA-256:
`A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
Physical input SHA-256:
`4ba3850d51c02f2e41615137424fc0bd687e04f5460f8173b03ed9f3a879c03b`.

Parent owns final spec/status documentation and completion workflow; product
and test snapshot is ready. Evidence establishes baseline correctness on these
fixtures, not BER gains, speed gains, training or complete system smoke.
