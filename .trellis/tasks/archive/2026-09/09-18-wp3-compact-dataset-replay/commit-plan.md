# WP3 proposed work commit

Implementation and independent review passed. User confirmed this work commit and normal archive/journal completion.

## Proposed commit
`feat: 完成 WP3 数据集与确定性重放`

One coherent commit contains approved WP3 data/CLI/config/tests, verification helper, product documentation, backend contract, task planning/evidence and parent progress. Prior planning changes belong to this same task; no unrecognized dirty files were found. Generated runs remain ignored.

## Exact file list
- `.trellis/spec/backend/index.md`
- `.trellis/spec/backend/wp3-data-contract.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/prd.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/task.json`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/check.jsonl`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/commit-plan.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/design.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/implement.jsonl`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/implement.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/prd.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/bug-retrospective.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/check-command-receipts.json`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/check-final-command-receipts.json`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/check-followup-command-receipts.json`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/check-independent-manifest.json`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/check-report.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/implementation-evidence.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/inspect-review-manifest.py`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/main-final-verification.json`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/planning-evidence.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/review-focus.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/run-review-checks.py`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/research/source-excerpts.md`
- `.trellis/tasks/09-18-wp3-compact-dataset-replay/task.json`
- `IMPLEMENTATION_STATUS.md`
- `README.md`
- `VALIDATION.md`
- `configs/base.yaml`
- `configs/wp3_smoke.yaml`
- `scripts/verify_wp3_artifacts.py`
- `src/pgvamp_ofdm/base.yaml`
- `src/pgvamp_ofdm/cli.py`
- `src/pgvamp_ofdm/config.py`
- `src/pgvamp_ofdm/data/__init__.py`
- `src/pgvamp_ofdm/data/audit.py`
- `src/pgvamp_ofdm/data/dataset.py`
- `src/pgvamp_ofdm/data/generate.py`
- `src/pgvamp_ofdm/data/manifest.py`
- `src/pgvamp_ofdm/data/materialize.py`
- `src/pgvamp_ofdm/data/records.py`
- `src/pgvamp_ofdm/utils/random.py`
- `tests/conftest.py`
- `tests/test_data_cli.py`
- `tests/test_data_random.py`
- `tests/test_data_records.py`
- `tests/test_dataset.py`
- `tests/test_materialize.py`

## Verified result
248 passed / 3 CUDA skipped; Ruff/format/mypy pass; two complete physical frames / 16 windows, max relative error 1.4435457958165196e-11; 32 dense samples exactly equal compact replay. Current 63 source/test/script/config fingerprints and persisted artifact hashes verified. Details: research/check-report.md and research/main-final-verification.json.

## Approval boundary
User confirmation authorizes this work commit followed by normal WP3 archive/journal workflow; no push. WP4 remains unstarted. Project workflow.md Phase 3.4 step 5 requires: "Present the plan once, ask for one-shot confirmation".
