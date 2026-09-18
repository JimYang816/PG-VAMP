# WP8 delivery and complete-frame demonstration

Authority: docs/CODEX_ENGINEERING_SPEC.md §§0.4, 5–9, 18–24. This guide records
the interface without redefining physical equations or reducing acceptance.

## 1. Scope / trigger
Read when editing demo-frame, final reproduction documentation or delivery evidence.
Existing smoke/train/evaluation interfaces retain their own contracts.

## 2. Signatures
```text
python -m pgvamp_ofdm demo-frame --config configs/cpu_dev.yaml
    --device cpu --dtype complex128 --seed 20260917 --output runs/<fresh-directory>
demo_frame(config: Config, output: Path, *, argv: list[str] | None = None)
    -> dict[str, Any]
```
CLI config and output are required; device defaults to CPU; dtype/seed inherit configuration
unless explicitly overridden. API receives an already validated Config.

## 3. Contracts
- One frame, 512 grid/400 data/8192 FFT/CP2048/eight blocks, default 91776 samples
  and 6400 bits. Dataset counts do not control the demonstration workload.
- Deterministic affine_doppler_strong paths with valid CP and at least two distinct
  nonzero epsilons; bounded retries via data.max_channel_attempts. Fixed Es/N0=10 dB,
  sigma2=0.1, recorded explicitly rather than taken from the configured sweep.
- Independent continuous receive recording includes delayed/stretched support.
  All eight windows are sliced from it, unitary-FFT transformed and compared to
  analytical H in complex128 with relative error <1e-9. Runtime H/y/sigma2 use
  requested paired precision after reference checks.
- LFM correlates noiseless real passband and real template. Selected start is
  not earliest path or frame origin, never shifts oracle FFT windows, and does
  not establish noisy RF synchronization or lfm_detect dataset performance.
- Independent real/imag time-IQ noise streams; no received-power normalization.
  Labels are audit-only. Demo performs no detector comparison or training.
- frame.pt contains safe tensors/basic types: layout/allocation/grid/bits/pilots,
  tx_analytic/tx_real, rx_analytic_noiseless/rx_real_noiseless/rx_iq_noiseless,
  iq_noise/rx_iq_noisy, lfm_real/sync/paths/cp_endpoints_s/seeds, and eight windows
  with recording spans, h_grid_reference/observed_grid_reference/H/y/sigma2/x/bits.
  Review with weights_only=True and map_location=cpu.
- resolved_config.yaml, environment.json, figures/waveform_and_sync.png and
  figures/channel_and_constellation.png accompany frame.pt. demo.json is written
  last with dimensions, seeds, error, sync boundary and SHA-256 artifact inventory.
  Environment records actual source hashes. Partial output has no success receipt.
- Windows validation sets MKL_THREADING_LAYER=TBB before Python startup; this
  selects the threading backend without disabling conflict detection.

## 4. Validation and error matrix
| Condition | Behavior |
| --- | --- |
| Algebra profile / reduced required dimensions | ValueError before output |
| sync_mode != oracle_timing | Reject, no silent window shift |
| Existing output path, even empty directory | Reject, preserve prior evidence |
| Invalid config/dtype/device or unavailable requested CUDA | Existing readable error, no silent fallback |
| No valid distinct-nonzero-path sample within retry budget | Reject, no success receipt |
| Nonfinite or >=1e-9 reference discrepancy | Fail with block and discrepancy |
| Plot/write failure after persistence | Retain incomplete artifacts, no demo.json success |

## 5. Good / base / bad cases
Base: CPU complex128/full dimensions/fresh destination. Good: explicit complex64
examples with separate tolerances and unchanged complex128 reference gate. Bad:
using detected LFM peak as oracle origin or two smoke updates as convergence.

## 6. Required tests
test_demo.py covers persisted full-size arrays, all eight FFT/H and pilot/noise
reconstructions, both CPU precisions, direct correlation, random stream replay,
deterministic rerun, config/output errors, CLI and failure-before-success publication.
Run an actual fresh demo CLI, independent saved-array audit and image review;
preserve §23 regression and actual complete-size CPU smoke.

Final delivery records functionality and sufficient performance evidence separately.
Main training, powered sweep and unavailable CUDA remain 未执行 with commands;
historical receipts cannot replace fresh acceptance or justify PG superiority.

## 7. Wrong vs correct
Wrong: H@x is both received signal and physical oracle.
Correct: independently evaluate continuous waveform, then FFT and compare to H.

Wrong: publish passed receipt before plotting or overwrite an existing destination.
Correct: reject old paths and write hashed success receipt last.

Wrong: software acceptance proves main-scale BER or GPU speedup.
Correct: report actual workload, device, precision and unexecuted experiments.
