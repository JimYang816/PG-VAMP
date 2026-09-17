# PG-VAMP CP-OFDM

Implementation follows [CODEX_ENGINEERING_SPEC.md](docs/CODEX_ENGINEERING_SPEC.md).
The current work package is **WP0: project foundation and independent mathematical references**.
See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for scope and
[VALIDATION.md](VALIDATION.md) for actual validation evidence.

## Environment and installation

CPU is the default, with complex128 tensors and float64 real parameters. CUDA
requires an explicit request; an unavailable device is an error.

The validation environment is Windows, Python 3.13.9 (Anaconda), PyTorch
2.12.0+cpu, NumPy 2.3.5, PyYAML 6.0.3 and Matplotlib 3.10.6.
This particular Anaconda environment needs the MKL TBB threading backend to
avoid a duplicate Intel OpenMP runtime conflict with PyTorch. In PowerShell,
set this **before starting Python**, including installation and tests:

```powershell
$env:MKL_THREADING_LAYER = 'TBB'
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e ".[test]"
.\.venv\Scripts\python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
.\.venv\Scripts\pgvamp-ofdm.exe inspect-config --config configs/cpu_dev.yaml
.\.venv\Scripts\python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml --output runs/wp0-inspect
.\.venv\Scripts\python.exe -m pytest -q --basetemp=runs/pytest-wp0
```

The recorded installation reuses the preflight-verified installed dependencies.
For a new environment where dependencies are missing, install them with
`python -m pip install -e ".[test]"` using the intended CPU PyTorch environment.
The explicit pytest temporary directory avoids this machine's shared temporary
directory permission problem. Reserve that directory for pytest; pytest manages
and cleans its own temporary contents.

The environment setting selects MKL's threading implementation; it does not
disable OpenMP conflict detection. Other environments may not need it. No
CUDA package or unrelated dependency upgrade is required for CPU operation.

## WP0 boundary

`inspect-config` checks configuration and displays derived quantities and
storage estimates. It does not generate data, train, or evaluate detectors.
Physical profiles retain the 512 grid / 400 data / 8192 FFT configuration.
`smoke_math` describes an explicit algebra fixture. Profile availability does
not mean that a `smoke` command or a physical simulation has been implemented.

The dense PG-VAMP and Cholesky VAMP references are correctness oracles for
small algebra fixtures. They are independent of future production algorithms.
WP1–WP8, physical validation, training and performance comparisons remain
separate work packages. No BER or acceleration claim is made here.

```python
from pgvamp_ofdm.reference.dense_pg_vamp import DensePGVAMP
from pgvamp_ofdm.reference.dense_vamp_cholesky import dense_vamp_cholesky

# H: complex128 [B,N,N], y: complex128 [B,N], sigma2: float64 [B] > 0.
# All inputs and the PG module must use the same device and paired precision.
pg_result = DensePGVAMP(depth=8)(H, y, sigma2, return_diagnostics=True)
vamp_result = dense_vamp_cholesky(H, y, sigma2, iterations=8)
```

Results contain final `x_soft`, four-class `probabilities`, `class_hat`,
`bits_hat`, diagnostic counters and optional layer states. PG learns exactly
`raw_gaps[T]` and `raw_mu[T]`; VAMP has no learned parameters. These APIs take
no target labels. Random matrices used by tests are algebra fixtures only.
