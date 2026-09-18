# PG-VAMP CP-OFDM

Reproducible CP-OFDM simulation, full-channel linear MMSE, exact VAMP and
trainable PG-VAMP-VC. The authority is [the engineering specification](docs/CODEX_ENGINEERING_SPEC.md).
WP0–WP7 interfaces are implemented; WP8 supplies delivery documentation and the
full-frame demo. Current acceptance results are recorded in
[VALIDATION.md](VALIDATION.md), with functionality and experiment scope in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

The physical chain uses a 512-position grid, 400 unknown QPSK symbols per block,
8192 waveform FFT samples, CP=2048 and eight blocks per frame. Comparison assumes
ideal complex I/Q, oracle timing, perfect CSI, known noise and no coding.
CPU/complex128 is the default. Main training, the complete main SNR sweep,
statistically sufficient multi-seed comparisons and CUDA experiments remain
**未执行 (not executed)**. Small acceptance runs do not establish superiority,
convergence or sparse acceleration.

## Install and run

Python 3.11 or newer is required. In a new environment with the intended CPU
PyTorch installation, install the package and test dependencies:

```sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[test]"
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python -m pgvamp_ofdm smoke --config configs/smoke_math.yaml --device cpu --output runs/quick-math
python -m pgvamp_ofdm smoke --config configs/smoke_system.yaml --device cpu --output runs/quick-system
python -m pgvamp_ofdm demo-frame --config configs/cpu_dev.yaml --device cpu --dtype complex128 --output runs/quick-demo
```

Use fresh output paths. The first smoke is N=32/T=2 algebra; the second retains
full physical dimensions/T=8, audits waveforms, performs two PG updates,
runs all three detectors and verifies checkpoint/inference roundtrips.
The demo produces one physical frame and synchronization evidence without training.
The console entry point `pgvamp-ofdm` accepts the same commands.

The recorded Windows/Anaconda environment needs `$env:MKL_THREADING_LAYER = 'TBB'`
in PowerShell **before starting Python** to avoid its duplicate OpenMP runtime
conflict. It reused preinstalled dependencies with
`python -m venv --system-site-packages .venv` and
`python -m pip install --no-build-isolation --no-deps -e ".[test]"`.
That reuse command does not install missing dependencies. See
[reproduction instructions](docs/REPRODUCING.md) for environment, full CLI workflow,
artifact interpretation and regression commands.

## Documentation

| Document | Purpose |
| --- | --- |
| [Configuration](docs/CONFIGURATION.md) | Defaults, profiles, dimensions, seeds, storage and runtime rules |
| [Algorithms](docs/ALGORITHMS.md) | Physical detection model, exact baselines, PG equations and implementation map |
| [Reproducing](docs/REPRODUCING.md) | Inputs, outputs, prerequisites, bounded acceptance and future main commands |
| [Physical model](docs/WP2_PHYSICAL_MODEL.md) | Detailed waveform/channel/receiver APIs and timing contract |
| [Implementation status](IMPLEMENTATION_STATUS.md) | Functional completion and experiment boundaries |
| [Validation](VALIDATION.md) | Actual dated executions, failures/skips and independent evidence |
| [Delivery matrix](docs/DELIVERY_MATRIX.md) | WP0–WP8 and source §23 requirements mapped to verification |

Historical independent reviews remain available for
[WP6 training/smoke](.trellis/tasks/archive/2026-09/09-18-wp6-training-checkpoint-smoke/research/check-report.md)
and [WP7 evaluation/reporting](.trellis/tasks/archive/2026-09/09-18-wp7-paired-evaluation-reports/research/check-report.md).
They describe those dated runs; current verification is tracked separately.

## Detector API

```python
from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector, PGVAMPDetector

# H: complex [B,N,N]; y: complex [B,N]; sigma2: paired real [B] > 0.
# Default models expect CPU complex128 inputs (float64 real parameters).
linear = MMSEDetector().detect(H, y, sigma2)
vamp = VAMPDetector(iterations=8).detect(H, y, sigma2)
pg = PGVAMPDetector().detect(H, y, sigma2)
```

Detectors receive no target labels. Results expose `x_soft`, `bits_hat`,
`class_hat`, `probabilities` and diagnostics. MMSE returns its raw linear estimate
and `probabilities=None`; VAMP/PG return final QPSK posteriors. PG has exactly
16 learned real scalars at depth 8. Explicitly use `torch.no_grad()` for ordinary
inference; PG forward otherwise retains autograd. Use checkpoint-based `infer`
for trained predictions; a newly constructed PG model is untrained.
