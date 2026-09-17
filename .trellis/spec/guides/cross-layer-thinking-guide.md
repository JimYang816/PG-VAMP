# Cross-layer contract review

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§3–11, 15–18, 20–23. The source takes precedence.

Trace config → physical frame → waveform/effective model → explicit pilot
elimination → shared H/y/sigma2 → detector → counts/report.

At each changed boundary verify:
- Units, shapes, q/grid/FFT indices, QPSK labels, dtype/device and normalization
  are explicit; 512, 8192 and 400 are distinct dimensions.
- Physical paths/windows satisfy CP support; changing a receive window/front
  end changes the effective operator/noise checks.
- Manifest/config/mapping hashes and sample IDs survive serialization and replay.
  Split separation holds for whole channel/frame instances and SNR copies.
- Detection has no labels or extra truth inputs; reporting uses targets only
  for metrics. Ordinary inference can omit labels.
- Checkpoint roundtrip preserves model/config/mapping; strict resume additionally
  restores optimizer, sampler and RNG state.
- Failure status reaches aggregate results rather than disappearing from counts.
  Recorded timing scopes and precision are identical across algorithms.

Use the full source test list (§23) to select cross-boundary checks; this guide
does not replace public-function input validation or add a new data schema.
