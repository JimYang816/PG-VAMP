# Reproducibility and persisted artifacts

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§10, 15.1–15.5, 18, 21.3–21.6, 22/WP0. The source takes precedence.

## Data and random streams
- Separate split, channel, bits, pilots, noise and arrival_offset random streams.
  Derive seeds from the master seed and metadata with a stable hash such as
  SHA-256, never Python hash(). Seed Python, NumPy, Torch CPU and enabled CUDA.
- Split physical frames/channel instances before expanding symbols or SNR
  copies. No shared frame/channel instance across train, validation and test.
  Three algorithms consume identical sample IDs, H, y, sigma2 and labels.
- Default to versioned manifest + compact .pt frame records containing tensors
  and basic Python types. Preserve all fields in §10.3; manifest includes full
  communication parameters, indices, generation distributions, split rules,
  seeds, software versions, checksums, counts and SNR definition.
- Reconstruct effective samples deterministically under no_grad. Cache keys
  include frame, symbol time, model version, mapping and dtype, not just frame.
  Do not normalize H implicitly. Estimate materialization size and enforce the
  configured threshold with explicit --allow-large-output (§10.4).
- Record environment, deterministic settings, threads and device. Promise
  replay in the recorded environment; test CPU/CUDA agreement with justified
  tolerances, not universal cross-platform bitwise identity.

## Checkpoints and resume
- Preserve the complete §15.3 schema: version, model/optimizer states, algorithm
  and waveform config, subcarrier/QPSK mappings, step/epoch/sampler position,
  seeds and Python/NumPy/Torch RNG states, best validation metric, train/val
  manifest hashes, dtype/training device/mask, software and code version/hash.
  WP0 also records the execution specification hash.
- Store RNG states as safely restorable tensors/basic types; no arbitrary pickle
  execution for untrusted files. Load to CPU first, then explicitly move device.
- Strict resume restores optimizer, sampling and RNG state, not just weights.
  Do not silently override waveform, mapping, mask or depth; record allowed
  runtime changes such as device.
- Train only PG-VAMP, using the exact §15.1 loss and settings. Select best
  checkpoint by validation final-layer NMSE, never test BER. Missing evaluation
  checkpoint errors unless --allow-untrained is explicit and the algorithm is
  named PG-VAMP-untrained.

## Logs and evidence
- Training logs are JSONL, checkpoints separate. Record every diagnostic required
  by §15.4, including thresholds, mu, graph/safety scales, precisions, rejection,
  no-information/cap rates, gradients and timing.
- Nonfinite values or Cholesky failure expose sample ID, dtype, matrix norm and
  key state, then stop or enter explicit failure handling; never nan_to_num
  into an apparently normal result.
- Preserve the resolved config, environment, dataset/checkpoint provenance,
  per-frame counts, timing and diagnostics prescribed by §18. Reports read
  results only; they never retrain or tune parameters.
