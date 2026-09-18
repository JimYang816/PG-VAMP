# PG-VAMP CP-OFDM

Implementation follows [CODEX_ENGINEERING_SPEC.md](docs/CODEX_ENGINEERING_SPEC.md).
The current work package is **WP6: training, checkpoint/resume and inference**.
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
`smoke_math` describes an explicit algebra fixture. WP6 supplies the separate
`smoke` command; configuration inspection alone does not execute a smoke run.

The dense PG-VAMP and Cholesky VAMP references are correctness oracles for
small algebra fixtures. They remain independent of production algorithms.
WP2 physical-channel validation is described below. WP3 data interfaces are described
at the end of this document. WP4 adds the production baselines below; WP5 adds
production PG-VAMP. WP6 adds training and smoke; unified performance comparisons
and complete delivery remain WP7–WP8 work.
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
cancellation separately below. WP6 adds the full three-detector system smoke.

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
training and performance sweeps remain unexecuted. Three-detector forward validation
is recorded separately in VALIDATION.md.

## WP4 production baselines

`MMSEDetector` returns the raw full-H Cholesky linear estimate (`MMSE (linear)`),
with nearest-QPSK hard decisions and `probabilities=None`. `VAMPDetector` performs
eight exact SVD linear/QPSK updates and returns the final posterior. It uses one
SVD per detection call and explicit numerical message protection. Neither has
trainable parameters. `configs/vamp_reference_32.yaml` selects the supplementary
32-iteration setting; instantiate `VAMPDetector(iterations=32)` when using it.

```python
from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector
from pgvamp_ofdm.data.dataset import detection_inputs

# sample is an EffectiveDataset item; retain its IDs and labels in the caller.
inputs = {key: value.unsqueeze(0) for key, value in detection_inputs(sample).items()}
linear = MMSEDetector().detect(**inputs)
vamp = VAMPDetector().detect(**inputs, return_diagnostics=True)
```

Inputs are finite complex `[B,N,N]` H and `[B,N]` y, with positive real
`sigma2[B]` on the same device. complex128/float64 is the correctness default;
complex64/float32 is explicit. Invalid inputs raise `ValueError`; numerical
failures raise `FloatingPointError` with batch/dtype/device context. VAMP counts
no-information, rejected messages, precision caps and variance underflow per
sample. Detailed diagnostics add layer vectors/scalars; no large per-layer
matrix copies are retained. Zero and numerically uninformative channels return
zero posterior means and uniform probabilities.

WP4 checks include independent Cholesky-oracle layer comparisons and actual
400-dimensional WP3 input with nonzero path time scaling. This does not establish
three-algorithm performance, full-system smoke, training results or a speed gain.

## WP5 production PG-VAMP

`PGVAMPDetector` implements the source §14 PG-VAMP-VC contract with exactly
`raw_gaps[T]` and `raw_mu[T]` real learned scalars: 16 at default depth 8.
It uses directed soft gates, energy compensation and analytical safety terms,
a centered full-H residual, exact conditional divergence and variance calibration.
The final result is the QPSK posterior, not an extrinsic message.

```python
from pgvamp_ofdm.algorithms import PGVAMPDetector

# inputs is the same batched, label-free H/y/sigma2 payload used above.
model = PGVAMPDetector()  # CPU, float64 parameters for complex128 inputs
pg = model.detect(**inputs)
# model(**inputs) is also supported; both entry points retain autograd.
```

Explicit complex64 inputs require `dtype=torch.float32`; device choices must match.
Use `torch.no_grad()` explicitly for inference. `return_diagnostics=True` retains
large, differentiable layer states for mathematical debugging; routine output keeps
protection counters and small detached layer summaries. Those summaries include both
edge-ratio denominators, thresholds, mu, c and relative safety-term size.

The model shares production QPSK/message protections with VAMP, while the dense
reference remains independent. It reuses one actual Cholesky factor within each layer;
all residuals still use full H. Dense cost may remain cubic and storage quadratic.
Soft-edge counts do not establish sparse acceleration. See VALIDATION.md for the
actual extent of WP5 verification and the WP6 section below for training support.

## WP6 training, resume and inference

Train only PG-VAMP using the prescribed layer-weighted complex MSE, Adam and
gradient clipping. The default remains CPU/complex128; select CUDA or complex64
explicitly. Missing data is an error: generate and audit the manifest first.
These commands describe the development workflow, not completed main experiments:

```powershell
python -m pgvamp_ofdm simulate --config configs/cpu_dev.yaml --output data/cpu_dev
python -m pgvamp_ofdm audit-data --manifest data/cpu_dev/manifest.json --waveform-frames 2 --output results/cpu_dev_audit
python -m pgvamp_ofdm train --config configs/cpu_dev.yaml --manifest data/cpu_dev/manifest.json --device cpu --output runs/pg_cpu_dev
python -m pgvamp_ofdm train --config configs/cpu_dev.yaml --manifest data/cpu_dev/manifest.json --device cpu --resume runs/pg_cpu_dev/last.pt --output runs/pg_cpu_dev
python -m pgvamp_ofdm materialize --manifest data/cpu_dev/manifest.json --split val --without-labels --output data/cpu_dev_unlabeled.pt
python -m pgvamp_ofdm infer --checkpoint runs/pg_cpu_dev/best.pt --input data/cpu_dev_unlabeled.pt --device cpu --output results/cpu_dev_inference.pt
```

`training.max_steps` is the cumulative update limit; increase it explicitly to
continue beyond a completed run. `last.pt` preserves optimizer, sampler and RNG
state; `best.pt` is selected by validation final-layer NMSE, never test BER.
Resolved configuration, environment and JSONL diagnostics accompany checkpoints.
Strict resume checks compatibility and rejects silent changes to physics,
mapping, model depth/mask, optimizer settings, data identity or precision.
Inference accepts no-label materialized data and inherits checkpoint precision;
`--dtype complex64` explicitly converts and records the conversion. CPU loading
uses restricted tensor loading before any explicit device transfer.

```powershell
python -m pgvamp_ofdm smoke --config configs/smoke_math.yaml --device cpu
python -m pgvamp_ofdm smoke --config configs/smoke_system.yaml --device cpu
```

The mathematical smoke uses N=32/T=2 and two updates. The physical smoke retains
512 grid carriers, 400 data carriers, 8192 FFT samples, CP=2048, eight OFDM blocks
and T=8; it includes waveform auditing, three detectors on shared inputs, two PG
updates, checkpoint loading, label-free inference and integer error counts.
These small runs demonstrate integration, not convergence or performance gains.
Full main training, SNR sweeps, unified evaluation and reports remain unexecuted.
