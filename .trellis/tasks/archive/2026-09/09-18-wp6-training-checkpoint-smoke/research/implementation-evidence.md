# WP6 implementation evidence — 2026-09-18

Status: implementation delivered for independent check; this is not the independent review verdict.

## Environment and authority
- Branch: codex/wp6-training-checkpoint-smoke; no commit performed.
- Python: `.venv/Scripts/python.exe` 3.13.9; PyTorch 2.12.0+cpu; NumPy 2.3.5.
- Windows 11 build 26200, CPU four-thread runtime, `MKL_THREADING_LAYER=TBB` before Python launch.
- Ruff: `C:/Software/Anaconda3/python.exe -m ruff` (installed environment).
- Authoritative specification hash remains a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b.
- No FastCtx tools exposed in this session, so local shell fallback was used.

## Delivered behavior and files
- `algorithms/pg_vamp/model.py`: optional differentiable layer outputs; small detached alpha/precision/d/ell/message diagnostics, with actual T-1 outgoing-message opportunities.
- `training/losses.py`: exact weighted complex error; no detached training path.
- `training/checkpoint.py`: restricted CPU load, complete versioned schema, model/Adam/RNG/sampler validation, mappings/config/manifest/provenance, atomic CPU tensor writes.
- `training/trainer.py`: deterministic tail batches/epochs, strict resume, real validation energy ratio, Adam/clipping, best/last, early stop, JSONL/failure context.
- `training/optimizer.py`: preserve PyTorch Adam math while bypassing its irrelevant CUDA graph-capture availability query for entirely CPU parameters; explicit CUDA retains upstream behavior.
- `inference.py`, `smoke.py`, CLI: label-free provenance-preserving inference; explicit dtype conversion; math/full-size smoke; integer smoke counts and complete-frame grouping.
- Mirrored base configs and strict configuration: patience/min_delta, fixed zero workers/weight decay and clip five.
- `scripts/wp6_acceptance.py`: bounded, repeatable CLI execution receipts; `--output` selects a fresh root.
- New tests: `test_training.py`, `test_checkpoint.py`, `test_inference.py`, `test_smoke.py`.
- Coordinator owns README/status/stable spec edits; implementer did not overwrite them.

## Final checks actually run
All commands ran with the environment above.

| Command | Observed result |
| --- | --- |
| `.venv/Scripts/python.exe -m pytest -q tests/test_pg_vamp.py tests/test_pg_vamp_properties.py tests/test_pg_vamp_gradients.py` | 48 passed, 1 skipped; 7.43 s (after initial layer extension) |
| `.venv/Scripts/python.exe -m pytest -q tests/test_training.py tests/test_checkpoint.py tests/test_inference.py tests/test_smoke.py --basetemp=runs/wp6-tests-atomic-final` | 38 passed, 1 skipped; 17.28 s |
| `.venv/Scripts/python.exe -m pytest -q --basetemp=runs/wp6-tests-implementation-final` | 414 passed, 7 skipped; 52.71 s |
| `C:/Software/Anaconda3/python.exe -m ruff check src tests scripts` | All checks passed |
| `C:/Software/Anaconda3/python.exe -m ruff format --check src tests scripts` | 85 files already formatted |
| `.venv/Scripts/python.exe -m mypy src` | No issues in 55 source files |
| `.venv/Scripts/python.exe scripts/wp6_acceptance.py --output runs/wp6-acceptance` | All 11 CLI subprocesses exit 0 |

`implementation-cli-receipts.json` preserves every exact subprocess argv/stdout/stderr/exit code. It includes inspect-config for cpu_dev and smoke_system, math smoke in complex128 and complex64, full system smoke, simulate, audit-data, two-step train, resume to four, unlabeled materialize, and infer. No full cpu_dev/main training was invoked.

## Final smoke evidence
See `implementation-artifacts.json` for exact file hashes, sizes, software and copied smoke reports. Ignored runtime artifacts remain at `runs/wp6-acceptance/`.

- Math: N32/T2/two updates, four scalars, exact checkpoint prediction roundtrip; both complex128 and explicit complex64 passed.
- System: grid512/data400/FFT8192/CP2048/eight blocks/T8/two updates; full real waveform audit and all three forwards, checkpoint and unlabeled inference passed.
- Independent waveform error: 1.1310698530350202e-11 (<1e-9). Audit path epsilon values are distinct and nonzero; saved in smoke.json.
- Each algorithm uses input hash 984908ffbc17d3407b7a3de15998e420e17543bb415dd7a1a77b4c4258bb0a09.
- Count denominators: 6400 bits, 3200 symbols, 8 blocks, one complete frame, zero incomplete frames. This tiny acceptance sample supports no performance conclusion.
- System last checkpoint SHA256: b8f7e0eb8fdb4f4583520a34af4b5546dd097384a6e977d6357ab4c5c60dfa50.
- CLI resumed training reaches step4; inference produces eight unlabeled-block predictions.

## Adversarial and boundary evidence
- Continuous four updates vs two+resume two: exact parameter, Adam moments/step, sampler permutation/cursor/epoch, ID sequence, RNG states and subsequent Python/NumPy/Torch draws.
- Tail batches and epoch transition; scheduled and off-cadence terminal validation; early-stop counts; new output best preservation; stale rewind rejection.
- Terminal validation remains a real measurement but its event is excluded from scheduled resume state. Terminal best is archived on resume; scheduled-best accompanies earlier best restoration. Resume requires max_steps above the saved step and no reduction from old configured max.
- last is atomically committed before best replacements. Injected best-write failure preserves a usable last and subsequently resumes. New-output resume of an already early-stopped run preserves last even without further updates.
- Unsafe/corrupted/missing/incompatible checkpoint fields, mappings, parameter dtype/shape, sampler, optimizer moments and behavior flags are rejected.
- CPU forbidden-CUDA tests cover train/resume/infer/math smoke; an additional fresh-process patched-CUDA math smoke also passed at `runs/wp6-cpu-forbidden-fresh`.
- Missing labels and changed valid labels give identical inference. complex64 conversion uses atol2e-5/rtol2e-4 for float32 factorization/accumulation on the well-conditioned fixture; complex128 uses exact or source 1e-9/1e-8 checks.
- Loss, gradient and Cholesky injected failures stop with IDs/dtype/matrix norm/key state; second-update failure leaves step1 checkpoint intact.

## Failures found and resolved during implementation
- Plain Python import initially exited on existing duplicate OpenMP runtimes. Used the documented TBB environment; no unsafe duplicate-runtime bypass.
- First tmp_path run hit sandbox permission on system pytest temp root: 2 passed/8 setup errors. Subsequent runs use unique workspace `--basetemp` roots.
- Initial CPU forbidden-CUDA test exposed PyTorch Adam's graph-capture query: fixed through the narrowly scoped optimizer subclass; retained native Adam updates/state.
- Initial mypy issue on NumPy RNG overload and formatting/lint findings were fixed; final checks above are clean.

## Remaining review / limits
- Independent Trellis check is pending; checklist item6 is deliberately open.
- CUDA numerical tests skipped because this PyTorch build has no CUDA; no GPU execution claim.
- Coordinator notified of one historical-data compatibility edge: old WP3 persisted resolved configs lack the two newly added early-stop keys, while config_from_values previously required an exact complete key set. Fresh WP3 regression passes; independent checker should validate narrow compatibility without changing old manifest hash identity.
- Main/full training, SNR sweeps, convergence, GPU performance, WP7 statistics/reporting and WP8 final delivery remain unexecuted/out of scope.
