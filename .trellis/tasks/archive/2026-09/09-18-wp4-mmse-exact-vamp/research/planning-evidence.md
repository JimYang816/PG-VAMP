# Planning evidence and review focus

Read-only inspection, 2026-09-18. No product tests run in this planning turn.

## Source anchors
- docs/CODEX_ENGINEERING_SPEC.md:769 — uniform detector API and label isolation.
- docs/CODEX_ENGINEERING_SPEC.md:799 — raw Cholesky linear MMSE.
- docs/CODEX_ENGINEERING_SPEC.md:834 — exact SVD VAMP, 8/32 distinction, protection.
- docs/CODEX_ENGINEERING_SPEC.md:1052 — analytical QPSK and messages.
- docs/CODEX_ENGINEERING_SPEC.md:1863 — WP4 gate.
- docs/CODEX_ENGINEERING_SPEC.md:1920 — algorithm tests and tolerances.
- src/pgvamp_ofdm/data/dataset.py:21 — cloned H/y/sigma2 projection.
- tests/test_reference.py:220 — weak-channel no-information boundary.

## Inspected components
reference/dense_vamp_cholesky.py computes covariance/W through independent Cholesky with layer states and explicit protection; production SVD must not call it.
modulation/qpsk.py owns mapping/tie policy; utils/validation.py validates batched square inputs; config.py already freezes protection constants and accepts positive iteration counts.
Archived WP3 and parent report 248 passed, 3 CUDA skipped: historical only. Formal algorithms directory absent at planning time.

## Review focus
Separate stable alpha2/c sums; exact SVD once per detect; no PG state. Derive/document SVD no-information error bound before completing implementation and verify against equations and weak-channel fixtures. Independently enumerate posterior and preserve oracle separation. Test mean-preserving caps and finite gradients at reciprocal boundaries. Preserve downstream §23 obligations; 400-dimensional baseline forward is not full three-algorithm smoke.

Source SHA-256 verified unchanged: A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B.
