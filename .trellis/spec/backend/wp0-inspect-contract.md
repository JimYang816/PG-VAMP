# WP0 configuration inspection contract

Authority: [engineering specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§15.2, 19–22. This records the WP0 interface; it does not extend WP scope.

## 1. Scope / trigger

The first executable configuration → runtime → persisted-artifact boundary is
`inspect-config`. It performs static checks and emits evidence without producing
physical data or training a model.

## 2. Signatures

```text
python -m pgvamp_ofdm inspect-config --config PATH [--device cpu|cuda:INDEX]
    [--dtype complex128|complex64] [--output DIRECTORY]
pgvamp-ofdm inspect-config (same arguments)
```

Python configuration entry: `load_config(path=None, *, device=None, dtype=None)`.
`summarize(config)` provides static derived values and storage estimates.

## 3. Contracts

- Packaged `base.yaml` mirrors `configs/base.yaml`; its contents follow source
  §20. Profiles recursively override mappings and replace whole sequences.
- All provided fields are checked before CLI overrides. An override cannot hide
  an invalid supplied field. Unknown nested keys are errors.
- `profile: smoke_math` selects a separate algebra schema with N=32, T=2,
  updates=2. It cannot carry physical fields to evade their validation.
- Physical profiles preserve 512 grid / 400 data allocation. Static CP duration
  checks do not certify per-path/per-block CP support; that requires WP2.
- CPU selection never queries CUDA. Explicit CUDA requests validate availability.
- An output directory contains the complete resolved YAML and environment/code/
  source provenance. Capacity figures are estimates, with scope and exclusions,
  and do not represent measured files on disk.

## 4. Validation & error matrix

| Input | Required behavior |
| --- | --- |
| Unknown key, nested typo, bad type or enum | Readable error; CLI nonzero |
| Contradictory FFT/frequency/allocation parameters | Reject before runtime work |
| Explicit unavailable CUDA | Error without CPU fallback |
| CPU request with GPU present | Remain CPU without CUDA probing |
| Algebra fixture with physical fields | Reject unknown physical fields |
| No output option | Print inspection; no artifact generation |

## 5. Good / base / bad cases

- Base: `configs/cpu_dev.yaml` resolves CPU/complex128, 96 kHz, FFT 8192,
  spacing 11.71875 Hz, 91776 samples/frame and 6400 data bits/frame.
- Good: explicitly selecting complex64 pairs it with float32; it remains a
  distinct precision path with its own tests.
- Bad: a 512-point waveform FFT at the default sample rate is rejected.

## 6. Tests required

Run `tests/test_config.py`, `tests/test_devices.py`, `tests/test_cli.py` and
`tests/test_reference.py`, followed by the full suite. Assert both installed CLI
entries, resolved YAML roundtrip, provenance, nested errors and absence of CUDA
calls on CPU. Numerical reference assertions remain governed by source §§13–14,
23 and the WP0 PRD, not by this CLI contract.

## 7. Wrong vs correct

Wrong: label `inspect-config --config configs/smoke_system.yaml` as a completed
physical smoke test. Correct: record successful parsing only; system smoke is
WP6 and keeps the full physical dimensions.

Wrong: silence the local Anaconda duplicate OpenMP error using
`KMP_DUPLICATE_LIB_OK`. Correct: select a compatible environment before Python
startup. This session verified `MKL_THREADING_LAYER=TBB` with actual complex
Cholesky/backward operations; record that environment in validation evidence.
