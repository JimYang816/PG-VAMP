# PG-VAMP CP-OFDM

Implementation follows [CODEX_ENGINEERING_SPEC.md](docs/CODEX_ENGINEERING_SPEC.md).
The current work package is **WP3: compact datasets, splits and replay**.
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
WP2 physical-channel validation is described below. WP3 data interfaces are described
at the end of this document. WP4–WP8 production detectors, training and performance
comparisons remain separate work packages.
The algebra references make no physical BER or acceleration claim.

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

## WP1 transmit-frame audit

WP1 adds fixed QPSK mapping, the 400/64/47/1 resource allocation, 8192-point
unitary IFFT, exact CP copies, the 91776-sample transmit frame, real passband
export and normalized LFM matching. The default frame carries 6400 uncoded bits.

```powershell
$env:MKL_THREADING_LAYER = 'TBB'
.\.venv\Scripts\python.exe scripts/run_wp1_audit.py --config configs/cpu_dev.yaml --output runs/wp1-audit
```

The audit saves the resolved configuration, environment and seed provenance,
real/analytic waveforms, frame layout and carrier indices, PSD/correlation arrays,
plots and numerical checks. The PSD reports finite-waveform sidelobes and uses
both positive and negative passbands for its out-of-band energy calculation.
It does not add a transmit window or normalize each frame by measured data power.

Matched-correlation positions denote the **template start**. Recording arrival
padding is separate from transmit-frame duration; a selected peak is not a claim
about the earliest physical path. The audit checks no-channel recovery only.
WP2 validates physical multipath, affine time scaling, effective H and pilot
cancellation separately below. Full three-detector system smoke remains future work.

## WP2 physical-channel audit

WP2 adds five physical channel scenarios, per-path/per-block CP support checks,
complete 512×512 effective H, an independent continuous Fourier/chirp waveform,
ideal-I/Q receive FFT, exact pilot elimination and time-domain complex AWGN.
The resulting detection system has 400 unknown QPSK symbols and uses perfect CSI.
See [WP2_PHYSICAL_MODEL.md](docs/WP2_PHYSICAL_MODEL.md) for API and timing contracts.

```powershell
$env:MKL_THREADING_LAYER = 'TBB'
.venv/Scripts/python.exe scripts/run_wp2_audit.py --config configs/cpu_dev.yaml --output runs/wp2-audit-new
```

Use a fresh output directory. The audit checks two full-size frames, all eight
blocks plus shifted edge windows, noise covariance and identity-AWGN hard-QPSK
BER. It saves actual parameters, seeds, numeric errors, counts and provenance.
Current results and review status are in VALIDATION.md. This is physical-model
acceptance; it does not run datasets, production detectors or training.

## WP3 compact data and replay

Generate a small dataset at the complete 512-grid / 400-data / 8192-FFT / 8-block
physical dimensions, independently audit two complete frames, then explicitly
export dense test inputs:

```powershell
$env:MKL_THREADING_LAYER = 'TBB'
.venv/Scripts/python.exe -m pgvamp_ofdm simulate --config configs/wp3_smoke.yaml --output runs/wp3-data-new
.venv/Scripts/python.exe -m pgvamp_ofdm audit-data --manifest runs/wp3-data-new/manifest.json --waveform-frames 2 --output runs/wp3-audit-new
.venv/Scripts/python.exe -m pgvamp_ofdm materialize --manifest runs/wp3-data-new/manifest.json --split test --output runs/wp3-test-new.pt --max-output-bytes 1073741824
.venv/Scripts/python.exe scripts/verify_wp3_artifacts.py --manifest runs/wp3-data-new/manifest.json --audit runs/wp3-audit-new --materialized runs/wp3-test-new.pt --output runs/wp3-verification-new.json
```

All output paths must be fresh. `generate` aliases `simulate`. Default storage is
versioned manifest plus compact `.pt` frame shards; loaders verify hashes, physical
CP support, mappings, labels, counts and split lineage using restricted CPU loading.
The manifest records full settings, environment/source hashes, seed derivation,
SNR pairing and independent frame versus expanded sample counts. Training and
validation use separate physical frames. Test SNR copies retain their physical
frame, channel, data and pilots with independent noise subseeds.

```python
from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.materialize import load_materialized

dataset = EffectiveDataset("runs/wp3-data-new/manifest.json", "test")
sample = dataset[0]  # H, y, sigma2, x, bits and stable identity metadata
inputs = detection_inputs(sample)  # only H/y/sigma2, each a defensive copy
dense = load_materialized("runs/wp3-test-new.pt", purpose="evaluation")
```

Replay uses no gradients and a bounded matrix cache, with block/time, model,
mapping, record/config hash and dtype/device in its key. Independent consumers
receive the same payload; returned tensors cannot contaminate cached matrices.
Both backends reconstruct independent real/imaginary time-I/Q AWGN streams and
perform the unitary FFT. `waveform_reference` evaluates the physical waveform;
`effective_fast` uses the previously audited analytical grid operator. The data
pipeline requires `oracle_timing`; `lfm_detect` remains a separate frontend demo.

Dense export reports exact tensor payload plus a conservative metadata/serialization
reserve. The default 1 GiB guard requires `--allow-large-output` to exceed it and
also applies to `data.storage=materialized`. `--without-labels` exports inference
inputs; `purpose="train"` and `purpose="evaluation"` reject missing labels. No H
normalization is applied. `audit-data` is explicit, using
`data.audit_waveform_frames` when its frame-count flag is omitted; simulation does
not automatically perform expensive waveform auditing. Full-size main generation,
production detector integration, training and performance sweeps remain unexecuted.
