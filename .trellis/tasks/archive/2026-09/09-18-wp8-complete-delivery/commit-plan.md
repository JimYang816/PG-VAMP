# WP8 concrete commit plan — approved

## Work commit
`feat: 完成 WP8 完整交付与 CPU 验收`

Adds the source-required demo-frame CLI and tests; consolidates configuration, algorithm and reproduction documentation; records fresh CPU acceptance and independent evidence; updates Trellis contracts and parent progress.

Files:
- `.trellis/spec/backend/index.md`
- `.trellis/spec/backend/wp8-delivery-contract.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/prd.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/task.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/check.jsonl`
- `.trellis/tasks/09-18-wp8-complete-delivery/commit-plan.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/design.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/implement.jsonl`
- `.trellis/tasks/09-18-wp8-complete-delivery/implement.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/prd.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/audit-evaluation-artifacts.py`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/audit-system-artifacts.py`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/authority-reference-baseline.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check-command-receipts.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check-delivery-audit.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check-demo-artifact-audit.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check-final-receipts.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check-report.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check-source-hashes.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check_delivery_audit.py`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check_demo_audit.py`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/check_run.py`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/code-evidence.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/coordinator-doc-links.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/delivery-matrix.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/docs-evidence.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/evaluation-artifact-audit.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/evaluation-receipts.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/evaluation-report-receipts.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/implementation-evidence.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/planning-evidence.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/report-visual-review.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/source-excerpts.md`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/system-artifact-audit.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/research/training-cli-receipts.json`
- `.trellis/tasks/09-18-wp8-complete-delivery/task.json`
- `IMPLEMENTATION_STATUS.md`
- `README.md`
- `VALIDATION.md`
- `docs/ALGORITHMS.md`
- `docs/CONFIGURATION.md`
- `docs/DELIVERY_MATRIX.md`
- `docs/REPRODUCING.md`
- `src/pgvamp_ofdm/cli.py`
- `src/pgvamp_ofdm/demo.py`
- `tests/test_demo.py`

## Verification
- 486 passed / 7 CUDA skipped; Ruff check/format src tests scripts and mypy pass.
- Fresh 11-command training/system-smoke loop, 17-command evaluation/timing/report loop and two actual demo precisions pass.
- Independent saved-system/demo waveform, count, checkpoint, hash and visual checks pass; main training/powered sweeps/CUDA remain unexecuted.

## Unrecognized dirty files — excluded
- `docs/WP7_EXAMPLE_REPORT.md`
- `output/pdf/WP7_EXAMPLE_REPORT.pdf`

## Approval boundary
Project .trellis/workflow.md §3.4 step 5 requires: "Present the plan once, ask for one-shot confirmation".
Approval authorizes this single work commit, followed by normal WP8 archive and journal bookkeeping commits. No push or amend. Parent coordination task and unrelated bootstrap task will not be archived automatically.
Before staging, verify the exact file list and source hashes still match the reviewed change; never stage unrelated files or ignored run arrays.


User confirmed this plan on 2026-09-18. Execute work commit, then WP8 archive and journal; no push. Precommit verification matched all 132 reviewed hashes before completion bookkeeping. Only approval/status and archive-destination references were updated afterward; no product code or numerical/test contract changed.
