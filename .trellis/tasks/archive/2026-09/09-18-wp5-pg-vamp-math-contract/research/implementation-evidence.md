# WP5 implementation evidence — 2026-09-18

Implementation authorized by the user after the planning-only turn. This report
records implementer executions; independent `trellis-check` review remains for
the coordinating session. No training, optimizer, checkpoint, full system smoke,
commit or archive was performed.

## Files and behavior

- Added production `algorithms/pg_vamp/{topology,majorizer,linear,model,__init__}.py`
  and exported PGVAMPDetector from algorithms. Only raw_gaps[T]/raw_mu[T] are
  trainable. Default T=8 has 16 real scalars.
- Reused production QPSK/messages without modifying them. Reference modules
  remain unchanged and independent.
- Added `tests/test_pg_vamp.py`, `test_pg_vamp_properties.py`,
  `test_pg_vamp_gradients.py`; expanded detector input/failure and physical
  shared-input tests to include PG-VAMP.
- Detailed diagnostic states remain differentiable. Only routine small logging
  copies detach. Default diagnostics omit the large per-layer states.
- One cholesky_ex per layer/call, with no cross-call cached graph. Explicit
  nonnegative jitter applies only to P; all innovation/W/trace/variance use that
  same factor. A continues to use full H and the original gamma2.

## Executed checks

Environment: Windows, Python 3.13.9, PyTorch 2.12.0+cpu, 4 CPU threads;
CUDA unavailable. Python commands used `.venv/Scripts/python.exe` and
`MKL_THREADING_LAYER=TBB`. The venv has no separate Ruff launcher; actual Ruff
executable was `C:/Software/Anaconda3/Scripts/ruff.exe`.

| Actual command | Result |
| --- | --- |
| `python -m pytest -q --basetemp=.tmp/wp5-full` | **370 passed, 6 skipped in 36.96s**; six CUDA hardware skips |
| `ruff check src tests` | All checks passed |
| `ruff format --check src tests` | 70 files already formatted |
| `python -m mypy src` | Success, 48 source files |
| `python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml` | Exit 0; CPU/complex128, 4 threads, 512 grid / 400 data / 8192 waveform FFT |
| `git diff --check` | Exit 0; only Git LF/CRLF informational warnings |
| `Get-FileHash docs/CODEX_ENGINEERING_SPEC.md -Algorithm SHA256` | A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B |
| `python .trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/audit_physical.py runs/wp5-implementation-tmp/wp5-full/data0/compact/manifest.json` | Exit 0; receipt saved in physical-forward-receipt.json |

After tests exited, the implementer-created `.tmp` directory was moved with
native PowerShell Move-Item to ignored `runs/wp5-implementation-tmp`, after
verifying both absolute paths were within this workspace and destination absent.
Future test commands should use `--basetemp=runs/wp5-<unique>` directly.

Initial checks and fixes (not hidden): the first new-test run had 43 passed,
1 CUDA skip, 1 AST-test failure because Windows read_text defaulted to GBK;
fixed test to read UTF-8 explicitly. The expanded focused run then had
152 passed, 3 skips and a physical fixture setup error because the `.tmp` parent
directory did not exist. Creating that parent resolved the setup error in the
full run. Initial Ruff found 30 formatting/import/line-length findings;
formatting and import ordering fixed them. No tolerance or numerical contract
was changed to make tests pass.

## Acceptance mapping and tolerance

- A1: depths 1/3/8/32, parameter names/count, analytic initialization,
  monotone/range/gaps, nested directed masks, zero columns/edges and diagonals.
- A2: independently looped small-matrix ell, energy diagonal, PSD safety,
  refinement Loewner ordering; 1e10 diagonal / 1e-12 neighbors retains ell=0.02.
- A3–A4: independent torch.linalg.solve construction of B, Hermitian/SPD and
  A^-1-B PSD, full residual equivalence, fixed-layer objective nonincrease,
  complete covariance trace, real-expanded xhat2 Jacobian/(2N). Both jitter=0
  and 0.125 use the same actual P; source tolerances retained.
- A5: N=8/16/32 with nontrivial common raw parameters, every independent
  DensePGVAMP layer state compared; both parameter gradients compared under
  identical weighted-layer real loss. gradcheck eps=1e-6/atol=2e-5/rtol=2e-4.
- A6: forced-full-graph test subclass compares every layer to both production
  SVD VAMP and independent Cholesky VAMP; mu independence. Identity/diagonal
  hard decisions, zero/rank-deficient/mixed batches, joint complex scaling.
- A7: existing production QPSK/four-point and message-guard backward tests
  remained in full regression; PG complete-loop cap/underflow backward tested
  for complex64/128 and amplitudes 25/200/1e6. Zero/subnormal information gives
  uniform probabilities, zero output and finite zero parameter gradients.
- A8: input/config/parameter errors, contextual hard failures, no forbidden
  inverse/CG/random operations/reference imports; one factor per layer across
  repeated backward calls. Explicit single precision state/backward and
  float64 comparison (atol=2e-5/rtol=2e-4 for O(N eps) small well-conditioned
  dense accumulation); dedicated CUDA output/gradient comparison skips.
- A9: full-depth 400-dimensional nonzero unequal path time scaling, same
  sample ID/H/y/sigma2 hash for MMSE/VAMP/PG. Fresh sample per detector;
  original/modified/absent-label complete prediction hashes match within each
  detector. See JSON for exact hashes and epsilon values. PG has 16 learned
  scalars and zero protection counts on this sample. This is a forward audit,
  not an optimizer/checkpoint/full-system smoke result.
- A10: implementer full regression/lint/format/types complete. Independent
  check and documentation/spec/task metadata remain coordinator-owned.

Complex128 forward atol=1e-9/rtol=1e-8, PSD >= -1e-9; no relaxed baseline
tolerance. Independent reviewer should additionally execute the planned NumPy
small-matrix audit; the implementation tests do not substitute for that review.

## Trace-bound audit

The production bound is eta*S with eta=4N eps/(1-4N eps),
S=sum_ij |W_ij||H_ji|/N, floored only at dtype tiny. It is conditional on
computed W/H: no claim of total Cholesky accuracy or finite-dimensional state
evolution. Direct complex elementwise contraction agrees with matrix trace
on phase-rich and rank-deficient fixtures. H=1e-20 I remains informative;
H=1e-160 I and H=0 take no-information with finite zero parameter backward.
The implementation rejects c < -tol and nonfinite W/innovation/tol before
classification; it also rejects 4N eps >= 1. No arbitrary eps signal cutoff,
no unit-scale floor and no hidden jitter was introduced.

## Remaining limitations

CUDA numerical parity is unexecuted because no hardware is available. Dense
operations remain potentially O(N³) with at least O(N²) storage. No sparse
speedup, BER advantage, trained performance or physical experimental validity
is claimed. WP6/WP7 are untouched. Independent checker owns final acceptance.
