# WP6 proposed commit plan

## Work commit

`feat: 完成 WP6 训练恢复推理与完整尺寸 smoke`

All paths below were created or edited in this task/session; no unrecognized dirty files. Runtime runs/ artifacts are ignored and excluded; durable receipts/hashes are included.

- `.trellis/spec/backend/index.md`
- `.trellis/spec/backend/wp5-detector-contract.md`
- `.trellis/spec/backend/wp6-training-contract.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/prd.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/task.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/check.jsonl`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/commit-plan.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/design.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/implement.jsonl`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/implement.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/prd.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-artifacts.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-artifacts.py`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-cli-receipts.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-format.txt`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-legacy.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-lint.txt`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-mypy.txt`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-pytest-final.txt`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-pytest.txt`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/check-report.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/implementation-artifacts.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/implementation-cli-receipts.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/implementation-evidence.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/independent-review-focus.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/main-final-verification.json`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/planning-evidence.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/research/source-excerpts.md`
- `.trellis/tasks/09-18-wp6-training-checkpoint-smoke/task.json`
- `IMPLEMENTATION_STATUS.md`
- `README.md`
- `VALIDATION.md`
- `configs/base.yaml`
- `scripts/wp6_acceptance.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/model.py`
- `src/pgvamp_ofdm/base.yaml`
- `src/pgvamp_ofdm/cli.py`
- `src/pgvamp_ofdm/config.py`
- `src/pgvamp_ofdm/data/manifest.py`
- `src/pgvamp_ofdm/data/materialize.py`
- `src/pgvamp_ofdm/inference.py`
- `src/pgvamp_ofdm/smoke.py`
- `src/pgvamp_ofdm/training/__init__.py`
- `src/pgvamp_ofdm/training/checkpoint.py`
- `src/pgvamp_ofdm/training/losses.py`
- `src/pgvamp_ofdm/training/optimizer.py`
- `src/pgvamp_ofdm/training/trainer.py`
- `tests/test_checkpoint.py`
- `tests/test_inference.py`
- `tests/test_smoke.py`
- `tests/test_training.py`
- `tests/test_wp6_review.py`

## Validation

Independent full suite: 430 passed / 7 CUDA skipped; Ruff/format/mypy pass; 11 CLI commands exit 0. Independent NumPy artifact verification and coordinator rehash of 56 source files / 54 artifacts pass. Source specification and independent mathematical oracles unchanged. No main training/sweep or WP7 work.

## After approval

Create the single work commit above, then normal Trellis archive/journal bookkeeping commits. Do not push. Do not start WP7.

## Required confirmation

.trellis/workflow.md Phase 3.4 step 5 requires: "Present the plan once, ask for one-shot confirmation". The user confirmed this concrete plan on 2026-09-18, authorizing the work commit followed by normal WP6 archive/journal completion. No push and no WP7 start.
