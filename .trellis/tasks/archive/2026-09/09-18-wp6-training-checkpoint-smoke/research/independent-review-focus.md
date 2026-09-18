# WP6 independent acceptance focus

This checklist supplements the approved PRD; it does not replace source §§15/21/23.

1. Recompute weighted complex loss independently; inspect gradient connectivity of every layer. Compare predictions and gradients with the pre-existing full-diagnostic path. Exactly two parameter vectors / 2T scalars; nondegenerate ICI for actual updates.
2. Strict resume must compare model, optimizer moments/step, sampler permutation/cursor/next IDs, all restored RNG states and next draws. Exercise epoch/tail-batch and validation boundaries. A weight-only roundtrip is insufficient.
3. Interrupted validation/log/checkpoint boundaries must not produce a misleading best checkpoint or double-counted successful step. Verify fresh run directory collision and incompatible resume errors before destructive output writes.
4. Safe checkpoint parsing validates structure, mappings, depth/mask/dtype, tensor finiteness/shapes, optimizer state and manifest identities; no unsafe pickle fallback. Test Python/NumPy RNG safe encoding roundtrip, not only Torch RNG.
5. CPU-only forbidden CUDA checks include seeding, provenance, checkpoint load/resume and infer. Existing utils/random.py uses default_generator.manual_seed on CPU specifically to avoid CUDA access.
6. Inference accepts missing labels and ignores changed labels, inherits checkpoint dtype, records explicit conversion and preserves IDs/input/checkpoint hashes. Data/physics compatibility is checked without incorrectly demanding the inference dataset equal the training dataset.
7. Log completeness: loss/validation event identity, rho/mu, edge ratios and denominators, d/ell, c/alpha1/alpha2, precisions, rejection/no-information/caps, gradient norm and time. Final layer has no outgoing message; denominators must reflect actual opportunities. No JSON NaN/Inf and no retained graphs in logging.
8. Failure injection for nonfinite output/loss/gradient and Cholesky must preserve contextual IDs/dtype/matrix norm and stop; failed observations cannot disappear into success counts.
9. Independent smoke artifact audit: actual 512/400/8192/2048/8/T8 dimensions, nonzero distinct Doppler paths, waveform/reference agreement, three common-input hashes, two PG updates, persisted checkpoint and label-free inference. Verify integer bit/symbol/block/frame totals from saved outputs, with complete-frame grouping; do not mistake fewer than 8 blocks for a complete frame.
10. Run full-scope checks and both CPU smokes with real receipts. complex64 and CUDA have justified tolerances; missing hardware is explicit skip. No WP7 performance or convergence claim.

Coordinator inspected prior WP5 check report and current random/runtime/safe-loading boundaries. No WP6 execution results are asserted here.
