# WP6 training and persistence contract

Authority: `docs/CODEX_ENGINEERING_SPEC.md` §§10–11, 14–15, 20–23.
This describes implemented interfaces; actual acceptance evidence belongs in VALIDATION.md.

## 1. Scope / trigger
Read before changing PG training outputs, loss, sampling, validation, checkpoint,
resume, infer or smoke. WP7 evaluation/statistics/reporting remains separate.

## 2. Signatures
```text
PGVAMPDetector.forward(H, y, sigma2, *, return_diagnostics=False,
                      return_layer_outputs=False) -> DetectionResult
training.losses.layer_loss(outputs: list[Tensor], target: Tensor) -> Tensor
training.trainer.train(config, manifest: Path, output: Path, *, resume=None) -> dict
training.checkpoint.load_checkpoint(path) -> dict
training.checkpoint.save_checkpoint(path: Path, value: dict) -> None
inference.infer(checkpoint: Path, input_path: Path, output: Path,
                *, device='cpu', dtype=None) -> dict
smoke.smoke(config, output: Path) -> dict
```
CLI: `train --config --manifest --output [--resume --device --dtype --seed]`,
`infer --checkpoint --input --output [--device --dtype]`,
`smoke --config [--output --device --dtype --seed]`. CPU remains the default.
Use existing `materialize --without-labels` to create inference inputs.

## 3. Contracts
- Only raw_gaps[T]/raw_mu[T] are trained. `layer_outputs` retains differentiable
  posterior means; logging summaries detach only small copies. Never use detached
  summaries for loss or request full matrix diagnostics in routine training.
- Loss is sum_t 2t/[T(T+1)] * mean(abs(xhat1[t]-x)^2). Adam defaults lr=1e-3,
  weight_decay=0, grad clip=5; only val final-layer NMSE selects best. Aggregate
  numerator/denominator energies before dividing, never average batch dB values.
- Manifest/split replay is explicit. No missing-data generation in train. Target
  labels remain with the caller and never enter the detector. CPU workers=0.
- Data readers call config_from_values(..., allow_legacy_data=True) for historical
  WP3 schema-1 configs lacking both early_stopping_patience and
  early_stopping_min_delta. Preserve the original values/hash: do not inject
  defaults into persisted payloads. A partially missing pair or any other missing
  key is still rejected. Checkpoints use strict current configuration validation.
- `max_steps` is cumulative successful updates. Persist sampler permutation,
  generator/cursor/epoch, optimizer and Python/NumPy/Torch RNG. CPU seeding uses
  the CPU default generator rather than a helper that also seeds CUDA.
- DeviceExplicitAdam preserves native Adam updates/state and bypasses only the
  irrelevant accelerator graph-capture health query for entirely CPU parameters;
  explicit CUDA uses upstream checks. CPU tests must forbid availability queries
  too, not merely CUDA tensor allocation. Recheck these hooks on PyTorch upgrades.
- Checkpoint schema 1 stores source/config/mapping, model/Adam states, progress,
  RNG, manifest hashes, dtype/training device/mask, software/code/spec provenance
  and run identity. Only tensors/basic types; weights_only CPU load, strict
  validation before applying state; no unsafe pickle fallback.
- Strict resume checks physics/mapping/PG settings, optimizer behavior flags,
  sampler/data identity and dtype; explicitly recorded device and increased
  max_steps are allowed. The new limit must exceed saved step and not reduce the
  previous configured limit; no-op resume is rejected. Validate Adam moments and
  step as well as weights.
- Scheduled validation state is preserved separately in resume_validation_state.
  Extra terminal evaluation must not alter resumed early-stop behavior. Matching
  best/scheduled-best artifacts must accompany a fresh-output resume when that
  best predates the selected last checkpoint; stale older
  resume into a newer run is rejected rather than publishing inconsistent best.pt.
- early_stopping defaults false, patience=10, min_delta=0. Patience counts
  validation events, not optimizer updates. Selection uses strict NMSE improvement.
- JSONL records all §15.4 diagnostics, actual message opportunities (T-1 per
  sample), no-information opportunities (T), gradient norm and elapsed time.
  Checkpoints are separate and atomically replaced after successful updates.
- Inference inherits checkpoint dtype unless explicitly converted and recorded.
  Its input may be a different dataset, but physics/mapping must agree; x/bits
  are optional and never affect prediction. execution_kind distinguishes algebra
  fixtures from physical checkpoints. Output preserves IDs and input/checkpoint hashes.
- System smoke keeps 512/400/8192/2048/8/T8, physical waveform audit, common
  three-detector inputs, two updates, roundtrip and label-free inference. Counts
  retain integer errors/totals and complete-frame grouping; no BER/convergence claim.

## 4. Validation and error matrix
| Condition | Expected behavior |
| --- | --- |
| Missing manifest/labels, empty split, incompatible config | Readable error; no substitute data |
| Unknown/unsafe/malformed checkpoint or incompatible resume | Reject before applying state |
| Output already owned by another run or stale resume | Reject; preserve existing artifacts |
| CPU selected | No CUDA API calls, including RNG/provenance |
| Explicit unavailable CUDA | Error, no CPU fallback |
| Nonfinite loss/gradient/output or failed factorization | Contextual failure with IDs/dtype/norm/state; stop, no hidden sample drop |
| Unlabeled physical materialization | Valid inference input |
| Algebra checkpoint used for physical inference | Reject |
| Explicit inference dtype conversion | Record source/input/actual dtype; test tolerance |

## 5. Good / base / bad cases
Base: CPU complex128, B1, T8 physical manifest, train then resume/infer.
Good: interrupt at epoch/tail-batch or off-cadence validation boundary and compare
with uninterrupted execution; safe CPU inference from CUDA-trained state.
Bad: restore weights only; trust tampered Adam maximize; keep a future best.pt
after rewinding; treat fewer than eight OFDM blocks as a complete physical frame.

## 6. Required tests
Independent weighted loss/gradients; unchanged WP5 predictions; 2T updates;
continuous vs resumed model/Adam/sample sequence/all RNGs; malformed and unsafe
checkpoint rejection; stale/fresh-output best consistency; off-cadence early-stop
equivalence; CPU forbidden CUDA and explicit unavailable CUDA; dtype conversion;
label isolation; contextual failures; manual metric counts; both actual CPU smokes;
full product regression/lint/format/type checks. CUDA numerical tests require hardware.

## 7. Wrong vs correct
Wrong: a successful load_state_dict proves strict resume.
Correct: compare optimizer moments, sampling progress, RNG next draws and later
updates with an uninterrupted run, including validation/early-stopping state.

Wrong: use terminal validation from an interrupted run as an extra patience event.
Correct: resume the scheduled validation state; preserve terminal results separately.
