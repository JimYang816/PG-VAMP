# WP0 validation

WP0 CPU validation and independent Trellis review passed. This file records
actual executions only; changes are not yet committed or archived.

## Source and starting state

- Original specification fully read in consecutive ranges 1–350, 351–700,
  701–1050, 1051–1400, 1401–1750 and 1751–2102.
- SHA-256: `A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
- Starting Git HEAD: `2611522` (`first commit`); working tree initially clean.
  Historical planning statements saying no commit existed are superseded by
  this observed execution state.
- Trellis context validation exited 0. Its warning about the 32,768-byte
  injection limit was handled by direct complete source reads.

## Environment preflight

Python: `C:\Software\Anaconda3\python.exe`, 3.13.9.
PyTorch 2.12.0+cpu; NumPy 2.3.5; PyYAML 6.0.3; Matplotlib 3.10.6;
pytest 8.4.2; Ruff 0.12.0; mypy 1.17.1.

The first CPU Cholesky probe **failed with exit 1**, reporting Intel OpenMP
Error #15 (duplicate `libiomp5md.dll`). No product test passed at that point.
With `MKL_THREADING_LAYER=TBB`, a fresh Python process successfully executed
complex128 Cholesky, batched solves, backward with finite gradients and NumPy
matrix multiplication (exit 0). `threadpoolctl` confirmed MKL's TBB backend.
The conflict detection bypass `KMP_DUPLICATE_LIB_OK` was not used.

All subsequent validation commands use `MKL_THREADING_LAYER=TBB` before Python
startup and record four Torch CPU threads. The environment was not upgraded.

Created a workspace-local environment with
`python -m venv --system-site-packages .venv` (exit 0), preserving global packages.
Actual editable installation command:
`.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e '.[test]'`
exited 0 and installed `pgvamp-ofdm 0.1.0`. Its additional flags prevent dependency
downloads and reuse the already available build backend. Subsequent product
validation uses `.venv\Scripts\python.exe` and the console executable beside it.

## Intermediate execution evidence

- Six planned CLI invocations succeeded (exit 0): module and console `cpu_dev`,
  module `main`, `smoke_math`, `smoke_system`, and `cpu_dev --output`.
  Original outputs are in the task's `research/cli-validation.json`.
- Independently compared every §20 default against parsed `configs/base.yaml`;
  all fields matched. See `research/base-source-check.json`.
- Both references executed two-layer identity and zero-channel cases; identity
  posterior maximum absolute error was 0 in this fixture, and zero-channel
  probabilities were exactly uniform. See `research/reference-preflight.json`.
- First implementation pytest: **1 failed, 29 passed, 1 skipped, 31 errors**.
  Errors came from shared temporary-directory permissions; the failed test
  replaced the entire CUDA namespace and intercepted an import-time type
  reference. The test now guards actual CUDA APIs, and later runs use a
  workspace-local `--basetemp`.
- After those fixes: **61 passed, 1 skipped**. Additional exact linear solve
  tests at N=8/16/32 and forbidden-API audit brought targeted/full implementation
  runs to **65 passed, 1 skipped**. These are intermediate results, not the
  independent final acceptance.
- Independent review then reproduced a genuine numerical defect: positive
  subnormal posterior variance could yield NaN backward gradients even when
  the candidate was rejected. A full PG fixture also exposed NaN parameter
  gradients in a precision-cap case. Both independent implementations now
  branch before invalid/capped reciprocals, preserving the original candidate
  mean and precision rules. Float32/float64 boundary and full-loop backward
  regressions passed.
- Review also found that an all-zero/all-no-information PG batch lacked an
  autograd graph. An explicit zero parameter dependency now permits backward
  with zero gradients, without changing outputs. Both regression cases passed.

## Final acceptance commands and results

Set `$env:MKL_THREADING_LAYER='TBB'` before these PowerShell commands.
The final complete command arrays, stdout, stderr and exit codes are preserved
in `.trellis/tasks/09-17-wp0-foundation-references/research/final-validation.json`.

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_devices.py tests/test_cli.py tests/test_reference.py --basetemp=runs/wp0-acceptance-targeted -ra` | 75 passed, 1 skipped; exit 0 |
| `.\.venv\Scripts\python.exe -m pytest -q --basetemp=runs/wp0-acceptance-full -ra` | 75 passed, 1 skipped; exit 0 |
| `python -m ruff check src tests` | Passed; exit 0 |
| `python -m ruff format --check src tests` | All 16 files formatted; exit 0 |
| `.\.venv\Scripts\python.exe -m mypy src` | No issues in 11 source files; exit 0 |
| `.\.venv\Scripts\python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml --output runs/wp0-inspect` | Passed; resolved config and final provenance refreshed; exit 0 |
| `git diff --check` | Passed; exit 0 (Git emits CRLF-normalization warnings) |

Ruff uses the installed `C:\Software\Anaconda3\python.exe` because the inherited
module's launcher is absent from the local venv. A prior venv Ruff invocation
failed for that launcher issue; no lint rule was disabled to resolve it. Mypy
checks the local package without a global missing-import bypass; only the
untyped PyYAML dependency has a targeted exception.

CUDA numerical equivalence is the single skip: `CUDA hardware unavailable`.
CPU-default and unavailable-CUDA error tests did run; mocked error branches
are not evidence of CUDA computation.

The independent check agent separately ran targeted and full suites after its
repairs: **75 passed, 1 skipped** in each (12.08 s / 13.60 s). No remaining WP0
code blockers were reported. The final main-session receipt is the reproducible
execution record, rather than a claim based only on the check agent's summary.

## Numerical contracts and evidence boundaries

- complex128 forward tolerance: `atol=1e-9, rtol=1e-8`; PSD floor `-1e-9`.
- Raw-parameter gradcheck: `eps=1e-6, atol=2e-5, rtol=2e-4`.
- Explicit complex64 check: finite outputs/gradients and `atol=2e-5, rtol=2e-4`
  against double on small well-conditioned fixtures; no single-precision result
  is presented as double-precision acceptance.
- No-information threshold uses the contraction error scale
  `gamma_(4N) * sum_ij(abs(W_ij)*abs(H_ji))/N`, where
  `gamma_(4N)=4N*eps/(1-4N*eps)`, with dtype `tiny` as the normal-precision floor.
  There is no arbitrary absolute floor tied to 1. Tests retain information for
  amplitude `1e-20` and explicitly handle `1e-160`/zero at float64's boundary.
- Full-graph forcing exists only in a test subclass; there is no runtime hard
  mask mode. Exact trace/Frobenius and Cholesky operations remain intact.
- Fixes changed evaluation order and gradient connectivity, not the source's
  message thresholds, precision limits, numerical tolerances or WP scope.

## Artifacts and acceptance mapping

`runs/wp0-inspect/resolved_config.yaml` and `environment.json` contain the full
configuration, environment, source specification hash, actual Git HEAD/dirty
state and sorted package-source hashes. Generated run files are Git-ignored.
The task's `research/final-source-manifest.json` additionally fingerprints
product source, tests, profiles and build configuration at validation.

| Criterion | Actual evidence |
| --- | --- |
| A1 | Editable install, package import, both executable CLI entries |
| A2 | All required physical values independently asserted in CLI receipts |
| A3 | Nested errors, type/enums/derived constraints, profile/roundtrip tests |
| A4 | CPU CUDA-call guards, explicit failure, paired precision tests |
| A5 | Both actual oracles, limits, parameter count, backward and gradcheck |
| A6 | Four-point enumeration, full-graph equivalence, exact solves, AST audit |
| A7 | Source/environment/code hashes, full config, actual command receipts |

Commit/archive remain pending the user's requested diff/results review. No
next work package was created or activated.

## Outside this validation

Full physical waveform/channel tests, production equivalence, complete system
smoke, training, checkpoint recovery, BER sweeps and performance benchmarks
have not been executed. They belong to later work packages.
