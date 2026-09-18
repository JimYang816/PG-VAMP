# WP6 independent check — 2026-09-18

Scope: approved PRD/design/implement, source §§15, 21.1–21.5, 22/WP6 and
WP6 engineering obligations in §23. Role: independently dispatched trellis-check;
no child dispatch, commits, task archive or WP7 expansion.

## Findings (fixed)

1. `config.py`, `data/manifest.py`, `data/materialize.py`, `inference.py`:
   exact persisted-config validation rejected historical WP3 schema-1 data after
   adding `early_stopping_patience` and `early_stopping_min_delta`. Reproduced on
   the actual `runs/wp3-acceptance-data/manifest.json` before fixing. Data reads
   now explicitly accept only the complete historical shape with both these
   keys absent. Original values and their hashes are preserved, not default-filled.
   New checkpoint validation remains strict. Tests reject one missing new key,
   other missing keys and a tampered config hash; historical unlabeled inference
   matches current metadata inference. Actual old manifest (6 records) and dense
   artifact (32 samples) load without byte changes: `check-legacy.json`.
2. `training/checkpoint.py`: Adam's `decoupled_weight_decay` flag was not checked.
   Reject true values; absence remains compatible with PyTorch releases preceding
   that flag. Review tests cover all seven Adam behavior flags, optimizer moments
   and interruption boundaries. No optimizer equations were changed.
3. `inference.py`: detector-raised numerical failures could bypass the output
   finiteness check and omit sample ID/matrix norm. Enrich detector exceptions
   and invalid output errors with sample ID, actual dtype/device, H norm/max,
   retaining the detector's original operation/layer message; no output is saved.
4. `smoke.py`: the explicit system guard checked 512/400/T8/two updates but omitted
   FFT/CP/block count. Enforce 8192/2048/8 as well, with reduced-dimension rejection
   tests. Default physical smoke retains all prescribed dimensions.

Regression additions: `tests/test_wp6_review.py` and legacy inference case in
`tests/test_inference.py`. Formatting/import issues introduced during review were
fixed before final checks. Main coordinator owns README/spec/VALIDATION/status;
the identified stale README/WP5 signature statements were communicated and fixed.

## Findings (not fixed)

No unresolved code or spec-contract findings. CUDA numerical execution remains
unavailable on the installed CPU-only PyTorch build; all seven relevant tests
explicitly skip. This is a recorded hardware limitation, not a passed CUDA test.
Main/full training, SNR sweep, statistical evaluation and performance claims were
not run and remain outside WP6 acceptance.

## Verification

All Python commands use `.venv/Scripts/python.exe` with
`MKL_THREADING_LAYER=TBB`, Windows/Python 3.13.9, torch 2.12.0+cpu. Complete saved
environment and per-source/artifact SHA-256 inventory are in `check-artifacts.json`.
Authority SHA-256 remains
`a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b`.
Independent mathematical references and authoritative specification were unchanged.

| Check | Actual result / receipt |
| --- | --- |
| `C:/Software/Anaconda3/Scripts/ruff.exe check src tests scripts` | pass; `check-lint.txt` |
| `C:/Software/Anaconda3/Scripts/ruff.exe format --check src tests scripts` | pass, 86 files; `check-format.txt` |
| `python -m mypy src` | pass, 55 source files; `check-mypy.txt` |
| `python -m pytest -q --basetemp runs/wp6-check-final` | **430 passed, 7 skipped**, 59.05 s; `check-pytest-final.txt` |
| `python scripts/wp6_acceptance.py --output runs/wp6-check-cli` | **11 subprocess commands exit 0**, exact args/stdout/stderr in `check-cli-receipts.json` |
| `python .trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-artifacts.py` | pass; independent NumPy recomputation in `check-artifacts.json` |

The CLI receipts cover inspect-config for cpu_dev/system, CPU math smoke in both
complex128 and complex64, real full-size CPU system smoke, simulate, audit-data,
two training updates, strict resume to four, no-label materialization and inference.
Math smoke uses N32/T2, two updates, four trainable scalars, exact checkpoint
roundtrip (0 maximum difference) in both dtypes. System uses
512/400/8192/2048/8/T8, two updates and 16 trainable scalars.

Independent saved-artifact audit rehashed checkpoint/input/data/audit files and
shared H/y/sigma2, checked all eight distinct block IDs form one complete frame,
recomputed errors/energies in NumPy, and checked no labels in the inference input.
NumPy FFT of actual saved received waveform versus analytical H has maximum
relative error **1.1310709079623577e-11** (<1e-9); independent FFT/pilot cancellation
also matches saved effective y. Paths have multiple distinct nonzero epsilons.
Three-detector shared-input hash:
`984908ffbc17d3407b7a3de15998e420e17543bb415dd7a1a77b4c4258bb0a09`.

| Algorithm | Bit errors / total | Symbol errors / total | Block errors / total | Frame errors / total |
| --- | --- | --- | --- | --- |
| MMSE | 326 / 6400 | 278 / 3200 | 7 / 8 | 1 / 1 |
| VAMP | 329 / 6400 | 281 / 3200 | 7 / 8 | 1 / 1 |
| PG-VAMP | 335 / 6400 | 286 / 3200 | 7 / 8 | 1 / 1 |

These are smoke counts, not comparative performance evidence. No failed block
is removed from these totals; incomplete-frame behavior has independent artificial
error tests. System saved/reloaded inference agrees at atol=1e-9/rtol=1e-8.

Resume review covers epoch/tail batch, parameter and optimizer tree, sampler state,
Python/NumPy/Torch state plus next draws, scheduled/off-cadence validation and
early-stop state, fresh-output best preservation, stale rewind rejection, CPU CUDA
API prohibition, and failures writing last/best/scheduled-best. Last write failure
resumes previous committed step; later best failure can reconstruct from committed
last. The resume log explicitly invalidates uncommitted forward events.

Historical/check iteration receipts are kept: initial historical read failed with
`persisted resolved config has missing keys`; initial review tests 12 passed;
first independent full run 426 passed/7 skipped preceded four final cases. Initial
`.venv/Scripts/python.exe -m ruff` failed because that system-site-packages venv
has no `Scripts/ruff.exe`; the installed Anaconda ruff executable completed the
final lint/format checks. No dependency upgrade or unsafe runtime bypass was used.
