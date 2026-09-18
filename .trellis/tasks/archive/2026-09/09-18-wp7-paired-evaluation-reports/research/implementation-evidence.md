# WP7 implementation evidence — 2026-09-18

Implementation runs use `D:/SCI/code/PG-VAMP/.venv/Scripts/python.exe`,
`MKL_THREADING_LAYER=TBB`, Windows CPU, torch 2.12.0+cpu. Exact environment,
source/spec hashes and actual argv are persisted in each evaluation's environment.json.
Source specification and independent reference implementations were not edited.

## Actual checks before independent review

- Cache/uncached + existing MMSE/VAMP/PG oracle/gradient regression:
  `python -m pytest -q tests/test_statistics.py tests/test_benchmark.py tests/test_mmse.py tests/test_vamp.py tests/test_pg_vamp.py tests/test_pg_vamp_gradients.py`
  → 77 passed, 1 CUDA skipped (3.04 s).
- First physical evaluation pytest invocation could not access the existing Windows
  Temp pytest root (4 fixture PermissionErrors, before product execution). Retried
  with unique workspace `--basetemp runs/wp7-pytest-eval-1` → 4 passed (14.87 s).
- `python -m pytest -q tests/test_evaluation.py tests/test_statistics.py tests/test_benchmark.py --basetemp runs/wp7-target-tests-v2`
  → 18 passed (18.93 s).
- Ruff check on owned core source/tests/harness passes; format applied. Mypy core
  source passed; reporting was still being implemented at that point.
- Full preliminary suite `python -m pytest -q --basetemp runs/wp7-full-tests-v1`
  → 459 passed, 7 skipped, 1 failed (98.50 s). Failure was the concurrently developed
  reporting semantic-tampering test `[error_energy-0]` (DID NOT RAISE); reporter and
  independent checker notified. This is not claimed as the final quality gate.

Ruff executable is `C:/Software/Anaconda3/Scripts/ruff.exe`. Independent checker
owns final frozen-source regression, lint/type/format and acceptance evidence.

## Actual full-dimensional CLI loop

`python scripts/wp7_acceptance.py --output runs/wp7-cli-acceptance-v1` completed
all 15 non-report subprocess commands with exit code 0. Full command arguments,
stdout/stderr, elapsed time and cwd are retained in `implementation-cli-receipts.json`.
Persistent configuration is copied into `implementation-smoke-config.yaml`.

Commands cover cpu_dev/main inspection; evaluate/benchmark/report help; generate;
independent waveform audit; two distinct training seeds with two updates each;
two checkpoint evaluations; cold-H and same-H benchmarks; unlabeled materialize;
checkpoint inference. Retained dimensions: grid512/data400/FFT8192/CP2048/eight
blocks/T8. The test population is one moderate physical frame at SNR0 and10,
with paired physical identity and independent noise draws. Benchmark batch sizes
1 and2, warmup1/repeats3; evaluation bootstrap retains default2000 repeats.

Physical outputs live under `runs/wp7-cli-acceptance-v1/`. Each evaluation bundle
has manifest/checkpoint/source hashes, exact input lineage, frame/aggregate counts,
paired intervals, diagnostics, failures ledger and a genuine waveform/channel example.
The independent main session recomputed both bundles' count/lineage relationships.

The acceptance commands were run during implementation: source fingerprints may
differ between subprocesses. The standalone benchmark common-preprocessing row was
added after these initial standalone timing receipts; independent check reruns that
latest behavior. One targeted test process briefly overlapped the first evaluation;
these timings are functional evidence, not isolated performance measurements.

## Limits

Main training, full SNR sweep, statistically adequate low-BER comparison and CUDA
numerical execution are 未执行. These smoke timings do not establish a speedup,
convergence, real-world performance or PG-VAMP superiority. Reporting and final
review evidence are tracked separately; WP8 is not started.
