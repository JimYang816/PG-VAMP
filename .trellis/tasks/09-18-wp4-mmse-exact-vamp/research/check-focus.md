# WP4 independent review focus

Scope: all WP4 changes and A1–A7, including files that remain untracked. Source
authority remains docs/CODEX_ENGINEERING_SPEC.md; no WP5 implementation.

## Independent evidence
Read production algorithms and tests separately from implementer conclusions.
Use an independently assembled small system / four-point enumeration to verify
the output and a real WP3 compact sample to confirm the data boundary. Preserve
existing Cholesky reference independence. Matching shared production helpers is
not an independent posterior/message check.

## Numerical risks
- Verify centered SVD delta, complex conjugation, separate alpha2/c sums and the
  exact gamma1 relation on non-diagonal complex H. One SVD per detect call.
- Check normal-equation residual and raw MMSE output; no posterior probabilities
  or disguised nonlinear postprocessing, and no implicit H normalization/jitter.
- Read the SVD no-information bound derivation. Weak representable channels
  remain informative; underflow/zero returns uniform probabilities. Do not
  reinterpret a nonfinite operator or factorization failure as no information.
- Reject invalid nonlinear candidates before undefined arithmetic; preserve
  previous messages. The upper cap keeps the candidate mean, not the old mean.
  Check representability at reciprocal boundaries and finite backward paths.
- Final output is posterior, not extrinsic. Last-layer diagnostics/counters must
  reflect operations actually performed. Mixed batches isolate zero/rank-deficient
  and informed members without corrupting messages or gradients.

## Integration and evidence boundaries
- Verify detector input shape/dtype/device, no label arguments or state, no
  input mutation, zero trainable parameters, and fixed QPSK tie ordering.
- Exercise both real detectors on the same 400-dimensional physical sample,
  preserving IDs and tensor hashes outside the API. Modify/remove targets.
- Verify supplementary VAMP-32 is a distinct profile, while default remains 8.
- No silent cache reuse, automatic device switching, or later-WP CLI features.
- Run appropriate full regression, Ruff/format/mypy, and config inspections.
  CPU/CUDA evidence must state actual hardware; skips are not passes.
- Record command results and source fingerprints. Check README/VALIDATION
  describe only executed results, not promised BER, speed, or system smoke.

Parent handles task status, specs, final commit plan and approval. Reviewer may
fix local product/test findings directly, preserving other agents' edits.
