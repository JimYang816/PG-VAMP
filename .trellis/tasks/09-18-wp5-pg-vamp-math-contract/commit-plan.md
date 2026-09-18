# WP5 proposed work commit

## Proposed commit
`feat: 完成 WP5 PG-VAMP 数学合同与独立验收`

One coherent work commit covering the production detector, mathematical/gradient and physical regressions, independently reviewed reference arithmetic repair, documentation/spec contracts, and task evidence. No generated runs/data or unrelated files are included.

## Exact file list
- `.trellis/spec/backend/index.md`
- `.trellis/spec/backend/numerical-contracts.md`
- `.trellis/spec/backend/wp5-detector-contract.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/prd.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/task.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/check.jsonl`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/commit-plan.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/design.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/implement.jsonl`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/implement.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/prd.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/audit_physical.py`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/bug-retrospective.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-final-pytest.txt`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-numpy-receipt.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-oracle-correction.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-oracle-final-pytest.txt`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-physical-receipt.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-pytest.txt`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-quality-receipts.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-report.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-source-manifest.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check-targeted.txt`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check_numpy_audit.py`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/check_physical_audit.py`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/implementation-evidence.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/main-final-verification.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/oracle-fix-review.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/physical-forward-receipt.json`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/planning-evidence.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/property-review-notes.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/source-acceptance-extract.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/research/source-math-extract.md`
- `.trellis/tasks/09-18-wp5-pg-vamp-math-contract/task.json`
- `IMPLEMENTATION_STATUS.md`
- `README.md`
- `VALIDATION.md`
- `src/pgvamp_ofdm/algorithms/__init__.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/__init__.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/linear.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/majorizer.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/model.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/topology.py`
- `src/pgvamp_ofdm/reference/dense_pg_vamp.py`
- `tests/test_detector_inputs.py`
- `tests/test_pg_vamp.py`
- `tests/test_pg_vamp_gradients.py`
- `tests/test_pg_vamp_properties.py`
- `tests/test_reference.py`

## Unrecognized dirty files
`docs/dsh-key-rotation-state.json` appeared during final verification and was not
created or edited by this WP5 session. Excluded from the proposed commit and left
untouched; its contents were not inspected. All files in the exact list above
belong to WP5 planning, implementation, independent check or coordinating documentation.

## Verified result
376 passed / 6 CUDA skipped; Ruff/format/mypy, config and context validation pass. Independent NumPy maximum discrepancy 1.0303e-13; physical 400-dimensional three-detector shared inputs and full prediction label isolation pass. Main independently verified 72 final source/test hashes. Source equations and tolerances unchanged; the PG oracle's two quotient arithmetic sites were corrected after separate review, with no production numerical imports. See research/check-report.md, check-oracle-correction.md and main-final-verification.json.

## Approval boundary
Project workflow.md Phase 3.4 step 5 requires: "Present the plan once, ask for one-shot confirmation". The user confirmed this plan on 2026-09-18, authorizing the work commit followed by normal WP5 archive/journal completion; no push. WP6 remains uncreated and unauthorized. Work is recorded on local branch codex/wp5-pg-vamp-math-contract.
