# Reproducing runs

Run commands from the repository root with the package installed. Commands below
are recipes, not evidence that those exact paths were executed. Actual dated
commands, environment, exit status and result limits are in
[VALIDATION.md](../VALIDATION.md). The [configuration guide](CONFIGURATION.md)
explains profiles; [algorithm assumptions](ALGORITHMS.md) apply to every run.

## Environment and output paths

Python >=3.11, PyTorch, NumPy, PyYAML and Matplotlib are package dependencies;
`pip install -e ".[test]"` adds pytest. Ruff and mypy are separate development
tools. Recorded Windows validation used Python 3.13.9, PyTorch 2.12.0+cpu,
NumPy 2.3.5, PyYAML 6.0.3 and Matplotlib 3.10.6; each run records its actual versions.
Fresh installation is distinct from the existing environment's dependency-reuse
installation documented in [README](../README.md).

For this Windows/Anaconda environment, before starting Python:

```powershell
$env:MKL_THREADING_LAYER = 'TBB'
# After activating the intended environment, use python below;
# alternatively replace python with .\.venv\Scripts\python.exe.
```

Use fresh run/data/result paths; change the suffix when rerunning. Generation,
demo, smoke, materialization, new training and evaluation reject conflicting
outputs. Resume intentionally uses its existing training run. A report can be
regenerated from a valid saved bundle. `inspect-config --output` writes config
and environment only and can reuse its destination, so choose a fresh one when
preserving evidence. Large arrays and runs are ignored by git; retain them
alongside command receipts when sharing validation evidence.

## Bounded acceptance

```sh
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml --output runs/example-inspect
python -m pgvamp_ofdm inspect-config --config configs/main.yaml
python -m pgvamp_ofdm smoke --config configs/smoke_math.yaml --device cpu --output runs/example-math
python -m pgvamp_ofdm smoke --config configs/smoke_system.yaml --device cpu --output runs/example-system
python -m pgvamp_ofdm demo-frame --config configs/cpu_dev.yaml --device cpu --dtype complex128 --output runs/example-demo
```

Inspection validates and prints dimensions, frequency endpoints, duration, bits,
runtime and storage estimates. It does not execute a smoke or generate samples.
Math smoke is N=32/T=2 with two updates. System smoke retains 512/400/8192/
CP2048/eight blocks/T8 and includes independent waveform auditing, two updates,
shared-input three-detector predictions, checkpoint loading and label-free infer.
Its `smoke.json`, `shared_predictions.pt`, training checkpoints and inference
artifact support numerical/count checks, not performance conclusions.

`demo-frame` accepts a physical config and optional `--seed`. It requires the full
dimensions and `oracle_timing`. It creates one deterministic sampled
`affine_doppler_strong` frame at Es/N0=10 dB with distinct nonzero path time
scalings. Its dataset count settings do not trigger dataset generation or training.

| Demo artifact | Contents |
| --- | --- |
| `resolved_config.yaml`, `environment.json` | Effective settings, runtime/source provenance |
| `frame.pt` | Full analytic/real TX and RX waveforms, ideal-IQ/noise arrays, physical paths, indices/layout, LFM arrays, all eight grid/effective systems and reference labels |
| `demo.json` | Dimensions, seeds, checks, synchronization summary and artifact hashes |
| `figures/waveform_and_sync.png` | Saved waveform and normalized correlation |
| `figures/channel_and_constellation.png` | Saved channel and receive example |

The LFM correlation uses the noiseless received real passband and is labeled
separately. The noisy ideal-IQ oracle windows produce 400-dimensional H/y/sigma2.
The selected LFM template peak is neither earliest-path truth nor the timing used
to form those windows. This is not noisy synchronization success-rate validation
or `lfm_detect` dataset replay. Independent continuous-waveform FFT vs analytical-H
checks cover every block. Reference labels are audit data, not detector inputs.

Repository acceptance harnesses supply small explicit overlays and durable command
stdout/stderr/exit receipts:

```sh
python scripts/wp6_acceptance.py --output runs/example-training-acceptance
python scripts/wp7_acceptance.py --output runs/example-evaluation-acceptance
python scripts/wp7_acceptance.py --output runs/example-evaluation-acceptance --reports-only
```

WP6 already runs both smokes (and a complex64 math smoke), so do not duplicate
them merely to increase run counts. It covers generate/audit/train/resume/infer.
WP7 uses two training seeds with two updates each, one test frame per SNR cell,
full dimensions and both timing modes. Its combined report exercises multi-seed
plumbing, not statistically sufficient multi-seed evidence.

## Development data, training and evaluation

This larger recipe uses `cpu_dev`: 128 train/32 validation frames and up to 300
updates, plus three scenarios × three SNRs × sixteen test frames. Complete it in
order. Missing data never triggers implicit generation inside training.

```sh
python -m pgvamp_ofdm simulate --config configs/cpu_dev.yaml --output data/example-cpu
python -m pgvamp_ofdm audit-data --manifest data/example-cpu/manifest.json --waveform-frames 2 --output results/example-audit
python -m pgvamp_ofdm train --config configs/cpu_dev.yaml --manifest data/example-cpu/manifest.json --device cpu --output runs/example-pg
python -m pgvamp_ofdm materialize --manifest data/example-cpu/manifest.json --split val --without-labels --output data/example-unlabeled.pt
python -m pgvamp_ofdm infer --checkpoint runs/example-pg/best.pt --input data/example-unlabeled.pt --device cpu --output results/example-inference.pt
python -m pgvamp_ofdm evaluate --config configs/cpu_dev.yaml --manifest data/example-cpu/manifest.json --checkpoint runs/example-pg/best.pt --algorithms mmse vamp pg_vamp --device cpu --output results/example-evaluation
python -m pgvamp_ofdm benchmark --config configs/cpu_dev.yaml --manifest data/example-cpu/manifest.json --checkpoint runs/example-pg/best.pt --device cpu --timing-mode both --output results/example-timing
python -m pgvamp_ofdm report --results results/example-evaluation
```

`generate` aliases `simulate`. The manifest and compact `.pt` shards preserve
hashes, split lineage, physical parameters and independent random streams.
`audit-data` independently reconstructs selected full waveforms; omitted
`--waveform-frames` uses the saved audit count. Dense materialization defaults to
a 1 GiB budget; use `--max-output-bytes` to set a limit and `--allow-large-output`
only when intentionally exceeding it. Omit `--without-labels` for labeled exports.

Training saves `resolved_config.yaml`, `environment.json`, `training.jsonl`,
`run.json`, `last.pt` and validation-selected `best.pt`. Resume restores optimizer,
sampler and RNG state. A completed run rejects resume at the same cumulative
`max_steps`. To extend this completed CPU development run, save the following
overlay as `runs/example-resume.yaml` (all other CPU development defaults remain):

```yaml
training:
  max_steps: 600
```

Then resume toward 600 cumulative updates:

```sh
python -m pgvamp_ofdm train --config runs/example-resume.yaml --manifest data/example-cpu/manifest.json --device cpu --resume runs/example-pg/last.pt --output runs/example-pg
```

For an interrupted run, the configured limit must still exceed the saved step
and must not reduce the previous configured limit.
Changed physics, mapping, optimizer, data identity or precision are rejected.
Inference consumes the no-label materialized file and saves predictions plus
checkpoint/input provenance, inheriting checkpoint precision unless explicitly
converted with `--dtype`.

Evaluation uses the fixed test manifest, not the validation export above. Its
config's seed/scenarios/SNRs/frame count must match that manifest. PG requires
a checkpoint; only `--allow-untrained` allows the clearly named
`PG-VAMP-untrained`. Select checkpoints using validation, never test BER.
For a second training seed, use `train --seed 20260918` in another fresh output
directory; evaluate its checkpoint with the **original test config/seed** into
a second result directory. Then combine compatible bundles:

```sh
python -m pgvamp_ofdm report --results results/example-seed1 results/example-seed2 --output results/example-combined
```

These two paths must first contain actual evaluations from distinct trained seeds.
Repeated runs of one seed do not create independent replicas.

## Reading results

Evaluation bundles contain resolved config/environment, manifest/checkpoint
metadata, `per_frame_metrics.csv`, `aggregate_metrics.csv`, `timing.csv`,
`diagnostics.jsonl` and integrity metadata. `report` validates saved artifacts and
creates applicable `figures/` and `REPORT.md` without rerunning models, data
generation or training. Bundles retain source data needed for report plots.

BER/SER/BLER/FER use integer errors and planned denominators; NMSE uses summed
symbol-error and reference energies before dB. Bootstrap uses independent frames
and paired algorithm resamples. Zero errors remain 0/measured count; log markers
do not change CSV truth. Hard failures preserve IDs and mark the cell
`incomplete_or_failed`, rather than deleting blocks from its denominator.

Timing reports distinguish `per_observation_cold_H` and `same_H_amortized`,
preparation, B1 online latency, fixed-batch throughput and common preprocessing.
PG's message-dependent Cholesky remains per layer. CPU RSS is process-lifetime
high-water memory; matrix work sizes are estimates. Neither is a claim of
per-algorithm allocated memory. Compute throughput is not acoustic-link goodput.

## Main and CUDA recipes — 未执行

Main generation/training, the full seven-SNR/five-scenario evaluation, sufficient
multi-seed statistics and CUDA experiments have not been executed. The following
is a reproducible starting recipe, not a result or promised convergence:

```sh
python -m pgvamp_ofdm simulate --config configs/main.yaml --output data/example-main
python -m pgvamp_ofdm audit-data --manifest data/example-main/manifest.json --waveform-frames 2 --output results/example-main-audit
python -m pgvamp_ofdm train --config configs/main.yaml --manifest data/example-main/manifest.json --device cpu --output runs/example-main
python -m pgvamp_ofdm evaluate --config configs/main.yaml --manifest data/example-main/manifest.json --checkpoint runs/example-main/best.pt --device cpu --output results/example-main
python -m pgvamp_ofdm benchmark --config configs/main.yaml --manifest data/example-main/manifest.json --checkpoint runs/example-main/best.pt --device cpu --timing-mode both --output results/example-main-timing
python -m pgvamp_ofdm report --results results/example-main
```

For explicit CUDA/complex64, first install a compatible CUDA-enabled PyTorch and
verify available hardware; create a separate main overlay with
`runtime.dtype: complex64` (all other main counts/settings retained), then generate
and audit its own manifest with that config. Train/evaluate/benchmark using the
same config, manifest and `--device cuda:0 --dtype complex64` in separate output
directories. Do not reuse the CPU complex128 manifest/checkpoint to silently
change a training/evaluation dtype contract. Materialize a no-label split from
the matching manifest and run `infer --device cpu` with its checkpoint to exercise
CPU inference; this is not retraining. `configs/cuda_example.yaml` demonstrates
runtime selection only and does not inherit the main workload.

## Regression and lower-level audits

```sh
python -m pytest -q --basetemp=runs/example-pytest -ra
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
git diff --check
```

Install Ruff/mypy in the intended development environment if absent. The explicit
pytest base temp avoids this machine's shared-temp permission issue; reserve it
for pytest, which cleans its own contents. CUDA tests may skip without hardware;
record that as skipped, not passed. Lint changed scripts separately when applicable.

For focused physical artifacts, `scripts/run_wp1_audit.py` and
`scripts/run_wp2_audit.py` accept `--config configs/cpu_dev.yaml --output <fresh-dir>`.
WP1 checks transmit/no-channel recovery; WP2 checks full waveform/channel/FFT,
pilot/noise and CP consistency. Neither alone is a three-detector system smoke.
Historical evidence is linked in [README](../README.md); always identify a new
run by its actual environment, source hashes, commands and saved artifacts.
