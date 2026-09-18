# WP4 implementation evidence — 2026-09-18

Implementation scope: full-H linear MMSE, exact SVD VAMP, common production
QPSK/messages, a 32-iteration configuration and their acceptance tests.
The reference numerics and base configuration are unchanged. Independent review
is pending; no commit/archive or later work package was performed here.

## Actual executions

All product commands ran in D:/SCI/code/PG-VAMP with MKL_THREADING_LAYER=TBB.
Final environment: existing .venv Python 3.13.9; torch 2.12.0+cpu; four CPU
threads in pytest and physical evidence; CUDA unavailable.

| Execution | Outcome |
| --- | --- |
| First targeted pytest, system Python | 58 passed / 2 skipped / 2 fixture failures: wrong gamma dtype and record key; fixed tests |
| Corrected targeted pytest | 60 passed / 2 CUDA skipped |
| Initial full receipt, system Python | 304 passed / 5 skipped / 4 failures: subprocess package imports/CLI; no editable install in that interpreter |
| Finite numerator/quotient overflow reproduction | Finite rejected forward produced NaN gamma/vbar gradients before guard; fixed production message evaluation order |
| QPSK/VAMP targeted pytest after guard and scalar identity test | 33 passed |
| `.venv/Scripts/python.exe -m pytest -q --basetemp=runs/pytest-wp4-receipt` | 313 passed / 5 CUDA skipped, 32.55 s |
| `.venv/Scripts/python.exe -m mypy src` | Pass, 43 source files |
| `.venv/Scripts/python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml` | Pass |
| Same inspect command with configs/vamp_reference_32.yaml | Pass |
| `.venv/Scripts/python.exe -m ruff check src tests` and format | Launcher failed: inherited package searches nonexistent .venv/Scripts/ruff.exe |
| `C:/Software/Anaconda3/Scripts/ruff.exe check src tests` | Pass |
| Same executable `format --check src tests` | Pass, 62 files already formatted |

Full stdout/stderr, exit codes, elapsed times and exact argv are retained in
implementation-validation.json, implementation-lint.json and
initial-system-python-validation.json. Failed environment attempts are not
counted as product passes. run-validation.py reproduces the final environment
using `.venv/Scripts/python.exe`; `--lint-only` uses the installed Ruff launcher.

## Acceptance mapping

- A1: test_mmse.py independently constructs solve/normal-equation and diagonal
  expectations, checks raw output, rank deficiency/zero mix, input immutability,
  class/bits dtype and None probabilities.
- A2: test_vamp.py checks all prescribed layer scalars/vectors and message flags
  against unchanged Cholesky oracle for N=8/16/32, T=8/32. One SVD per detect,
  no cross-call cache, zero parameters and final-posterior output are checked.
- A3: test_detector_qpsk.py uses four-point enumeration, tests unclipped alpha,
  nonpositive/too-low precision, invalid denominators, nonfinite candidates,
  variance/reciprocal boundaries, mean-preserving cap, gradcheck and finite
  rejected/capped backward. Finite numerator and quotient overflow are explicitly
  rejected before division; independent reference was not changed.
- A4: mixed zero/rank-deficient and 0/1e-160/1e-20/I weak-channel tests;
  identity scalar posterior, complex scaling and forbidden numerical API checks.
  SVD positive-sum rounding rationale is in svd-no-information.md.
- A5: shape/dtype/device/finite/noise rejection and contextual hard-failure tests;
  complex64 smoke/gradients; CUDA tests explicitly skip on CPU build.
- A6: test_detector_inputs.py reconstructs a nonzero unequal-time-scaling WP3
  sample at 400 dimensions and tests both actual baselines on identical payload,
  immutability and changed/absent labels. implementation-physical.json records
  actual sample ID, physical epsilon values and payload/output hashes.
- A7: full earlier-WP regression, product static checks passed as above;
  independent review remains a parent-coordinated follow-up.

complex128 uses atol=1e-9 / rtol=1e-8 unchanged. Explicit complex64 tests use
atol=2e-6 / rtol=2e-5 on N=8 systems with H scaled by sqrt(N), noise >=0.5
(MMSE noise >=0.3); this is a single-precision smoke tolerance, not a guarantee
for arbitrarily ill-conditioned channels. No failed numerical assertion was
resolved by relaxing tolerance. SVD full-input gradients are smoke-tested on a
nondegenerate two-by-two fixture; repeated singular-value gradient behavior is
not a claimed baseline training feature. The reusable posterior/messages retain
their finite-gradient protections for WP5.

## Physical evidence

Sample ID: b27666865a33658d825c349e31c201e9cfaeb3e771dae963bc952233308556a9:0x0.0p+0:0.
Shared input SHA-256: 4ba3850d51c02f2e41615137424fc0bd687e04f5460f8173b03ed9f3a879c03b.
Both before/after hashes agree; both outputs are finite and have zero trainable
parameters. VAMP no_information/message_rejected/precision_capped/underflow
counts for this one sample are all zero. This is baseline forward evidence,
not a performance benchmark or complete §15.5 system smoke.

## Remaining follow-ups

Independent checker review; CUDA acceptance requires hardware. WP5–WP8
production PG-VAMP, training/checkpoint, shared three-detector evaluation,
failure-denominator aggregation, honest cold-SVD timing, performance sweeps and
full-system smoke remain unexecuted. The source SHA remains
A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B.
