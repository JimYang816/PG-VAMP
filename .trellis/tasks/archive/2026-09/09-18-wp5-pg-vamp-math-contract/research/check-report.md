# WP5 independent review — 2026-09-18

Reviewer: dispatched trellis-check. Production/tests reviewed directly, without
another implement/check dispatch. Source SHA-256 independently rechecked:
`A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
Read task PRD/design/implementation plan, curated mathematical/acceptance extracts,
check manifest, applicable backend guides and independent property-review notes.

## Findings (fixed)

- `src/pgvamp_ofdm/algorithms/pg_vamp/linear.py`: a resolved tiny positive c
  could produce finite posterior output but NaN raw-parameter gradients. Minimal
  reproduction: depth=2, complex128 H=1e-140*[[1,.2j],[.3,1]], y=ones, sigma2=1,
  loss=x_soft.real.sum(). Both raw parameter gradients were [NaN,NaN]. Autograd
  anomaly traced the failure to innovation/c. Trying real-component division
  alone still failed: the local x/c² derivative overflowed before its upstream
  weight was applied. Fixed W/c and innovation/c through real-component
  `(x/sqrt(c))/sqrt(c)` evaluation. No channel/noise normalization, detach,
  new cutoff, modified A/P or altered mathematical formula. Added complex128
  1e-140 and complex64 1e-17 mixed weak/zero/ordinary regressions. The weak
  output remains informative and agrees with the independent H^H y/sigma2
  first-order posterior; the batch's gradients agree with its ordinary member
  alone within 2e-6/2e-5 (the weak contribution is below working precision).
- `src/pgvamp_ofdm/algorithms/pg_vamp/model.py`: direct vector norms could
  overflow/underflow when finite G/ell entries were squared for the diagnostic
  safety ratio, yielding NaN or a false zero baseline. Dividing both norm inputs
  by the same finite per-sample maximum preserves the ratio and zero test.
  Added joint H/y/noise scaling tests at 1e-100 and 1e100, with unchanged source
  complex128 tolerances 1e-9/1e-8. These computations only feed log summaries.

- Independent `reference/dense_pg_vamp.py` had the same pre-existing weak
  boundary: the exact minimal reproduction above returns both raw gradient
  arrays [NaN,NaN]. Verified separately after the production fix and reported
  before modifying the reference. The coordinator independently reviewed the
  source-equivalent repair and explicitly authorized a narrowly bounded oracle
  correction. See `check-oracle-correction.md` for §14.5/14.8 justification.
  The reference independently uses real/imaginary component square-root
  divisions, without any production helper/import. Added its own mixed-batch
  float64/float32 weak-boundary regressions and independent weak-channel
  posterior expansion, rather than merely asserting equality to production.
  Normal N=8/16/32 forward and both raw-gradient parity remain mandatory.

## Findings (not fixed)

No known unresolved code findings remain from this review.
- CUDA numerical checks remain unavailable (CPU-only PyTorch). Six explicit
  skips are not CUDA acceptance. No production defect remains known from this
  review. Training/checkpoint/system smoke are outside this WP and unexecuted.

## Verification

Environment: Windows, .venv/Scripts/python.exe, Python 3.13.9,
PyTorch 2.12.0+cpu, NumPy 2.3.5. MKL_THREADING_LAYER=TBB before Python;
OMP_NUM_THREADS=MKL_NUM_THREADS=4; pytest and audits set torch CPU threads=4.

Actual commands (repository root):

```text
.venv/Scripts/python.exe -m pytest -q --basetemp=runs/wp5-check-pytest
.venv/Scripts/python.exe -m pytest -q tests/test_pg_vamp.py tests/test_pg_vamp_gradients.py tests/test_pg_vamp_properties.py --basetemp=runs/wp5-check-targeted
.venv/Scripts/python.exe -m pytest -q --basetemp=runs/wp5-check-final
.venv/Scripts/python.exe -m pytest -q --basetemp=runs/wp5-check-oracle-final
.venv/Scripts/python.exe .trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check_numpy_audit.py
.venv/Scripts/python.exe .trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check_physical_audit.py
C:/Software/Anaconda3/Scripts/ruff.exe check src tests
C:/Software/Anaconda3/Scripts/ruff.exe format --check src tests
.venv/Scripts/python.exe -m mypy src
.venv/Scripts/python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
.venv/Scripts/python.exe .trellis/scripts/task.py validate .trellis/tasks/09-18-wp5-pg-vamp-math-contract
```

Before fixes: 370 passed, 6 skipped, 35.99s. Targeted after fixes: 48 passed,
1 skipped, 1.37s. Production-fix full regression: **374 passed, 6 skipped, 36.93s**.
Final oracle-correction full regression: **376 passed, 6 skipped, 36.10s**,
preserved in `check-oracle-final-pytest.txt`.
Ruff lint/format: pass, 70 files. Type-check: pass, 48 source files.
Configuration inspection and task context validation: pass.

`check-{pytest,targeted,final-pytest}.txt` preserve test stdout.
`check-quality-receipts.json` preserves exact quality commands, UTC start times,
exit codes, stdout/stderr and environment. `check-source-manifest.json` hashes
all final source/tests plus engineering specification and pyproject.toml.
No source numerical tolerance was relaxed.

## A1–A10 evidence mapping

| Gate | Reviewed code and independently executed evidence |
| --- | --- |
| A1 | Parameter registration exactly raw_gaps/raw_mu; 2T at depths 1/3/8/32; default 16. Baselines zero. Threshold initialization/spacing/range, directed nested mask, diagonal one, zero column/edges verified. |
| A2 | d=(1-M²)|H|², exclusive positive prefix/suffix sums, triple-definition ell, energy diagonal, PSD and ordered-mask Loewner tests. Independent NumPy loops reconstruct ell/G/Gbar. |
| A3 | Full-H centered innovation and actual P/B reviewed; one factor/layer test and source AST check pass. NumPy solve-based B is Hermitian positive and bounded by solve(A,I); fixed-input objective decreases. |
| A4 | Conditional real-expanded xhat2 Jacobian fixes H/gamma/M/mu and traces to alpha2. Both tests and separate NumPy complete covariance trace agree with variance, including explicit jitter 0.125. |
| A5 | N=8/16/32, nontrivial identical raw parameters, all independent oracle layer fields and both parameter gradients compare under the same weighted real loss; gradcheck 1e-6/2e-5/2e-4. No forward graph detach beyond log copies. |
| A6 | Forced-full-graph test injection compares exact SVD and separate Cholesky VAMP layerwise and two mu values. Identity/diagonal hard bits, zero/rank/mixed, weak-resolved and subnormal/no-info cases, complex joint scale pass. |
| A7 | Shared production QPSK/message routines unchanged; their enumeration, reject, cap, underflow and unsafe quotient regressions pass. Full PG saturation backward in both dtypes, weak mixed-batch backward and zero gradients pass. Finite/nonfinite trace and factor failures checked before no-info classification. |
| A8 | Invalid shape/dtype/device/noise/raw parameters/settings rejected; CPU default and unavailable CUDA error; explicit complex64 states/forward/gradients pass. No inverse/CG/random trace/cache/hard inference. Reference arithmetic-only fix independently reviewed and tested; common production helpers untouched. |
| A9 | Final fresh EffectiveDataset per detector: N400 WP3 sample16 with nonzero path epsilon; MMSE/VAMP/PG share sample ID and input hashes. Changed/absent labels preserve full prediction hashes. Default PG16 parameters/T8. Diagnostics denominator and scaled safety tests pass. |
| A10 | This independent review, full regression, lint/format/types/config/task validation, NumPy and physical receipts. Parent owns final root docs/spec/metadata synchronization and has reviewed the narrowly scoped reference repair. |

Independent NumPy audit: six n/jitter cases (N=8/16/32, jitter=0/.125), scalar
triple-sum safety, direct solve B, W, centered innovation, c/K/r1 and complete
covariance. Maximum absolute production discrepancy **1.0302869668521453e-13**,
within 1e-9/1e-8; PSD tolerance -1e-9 unchanged. Per-field errors, conditioning
and eigenvalue minima are saved in `check-numpy-receipt.json`.

Final `check-physical-receipt.json` records exact sample/input/prediction hashes,
epsilon values and all three detector parameter counts. Earlier implementer
receipts are historical (before review arithmetic changes). Physical audit is
untrained forward only, not complete system smoke or a performance experiment.

Trace tolerance remains max(gamma_(4N)*sum|W_ij||H_ji|/N,tiny). It bounds the
computed contraction only, not factorization total error or state evolution.
No claims of sparse speedup, BER advantage, training or WP6 completion are made.
