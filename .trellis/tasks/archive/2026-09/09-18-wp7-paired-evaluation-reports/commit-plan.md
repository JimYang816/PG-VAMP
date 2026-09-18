# WP7 concrete commit plan — approved by user

## Work commit
Message: `feat: 完成 WP7 配对评测统计计时与报告`

One coherent WP7 change: paired evaluation and frame statistics, failure-safe
counts, prepared inference and fair timing, read-only reports/multi-seed merge,
CLI/tests/acceptance harness, interface specs and task/evidence documents.

Exact included paths:
- `.trellis/spec/backend/index.md`
- `.trellis/spec/backend/wp7-evaluation-contract.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/prd.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/task.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/check.jsonl`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/commit-plan.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/design.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/implement.jsonl`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/implement.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/prd.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/check-artifact-audit.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/check-fingerprints.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/check-receipts.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/check-report.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/check-verify.py`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/coordinator-audit.py`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/coordinator-count-audit.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/coordinator-final-count-audit.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/coordinator-final-source-check.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/coordinator-visual-review.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/implementation-baseline.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/implementation-cli-receipts.json`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/implementation-evidence.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/implementation-smoke-config.yaml`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/independent-review-focus.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/planning-evidence.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/source-excerpts.md`
- `.trellis/tasks/09-18-wp7-paired-evaluation-reports/task.json`
- `IMPLEMENTATION_STATUS.md`
- `README.md`
- `VALIDATION.md`
- `scripts/wp7_acceptance.py`
- `src/pgvamp_ofdm/algorithms/mmse.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/linear.py`
- `src/pgvamp_ofdm/algorithms/pg_vamp/model.py`
- `src/pgvamp_ofdm/algorithms/prepared.py`
- `src/pgvamp_ofdm/algorithms/vamp.py`
- `src/pgvamp_ofdm/cli.py`
- `src/pgvamp_ofdm/evaluation/__init__.py`
- `src/pgvamp_ofdm/evaluation/artifacts.py`
- `src/pgvamp_ofdm/evaluation/metrics.py`
- `src/pgvamp_ofdm/evaluation/runner.py`
- `src/pgvamp_ofdm/evaluation/statistics.py`
- `src/pgvamp_ofdm/evaluation/timing.py`
- `src/pgvamp_ofdm/reporting/__init__.py`
- `src/pgvamp_ofdm/reporting/plots.py`
- `src/pgvamp_ofdm/reporting/report.py`
- `src/pgvamp_ofdm/reporting/validation.py`
- `tests/test_benchmark.py`
- `tests/test_evaluation.py`
- `tests/test_reporting.py`
- `tests/test_statistics.py`
- `tests/test_wp7_review.py`

## Verified evidence
478 passed / 7 CUDA skipped; product + scripts Ruff/format and mypy pass.
17 real CPU full-dimensional CLI commands pass. Independent artifact/visual
checks pass; specification and independent oracles unchanged. Current source
fingerprints match the final checker. Main training/full sweep/CUDA are unexecuted.

## Unrecognized dirty files
None. Pre-existing dirty state was the WP7 planning work from this same session.
Generated runs/results/data remain ignored and are not committed; task receipts
and artifact hashes preserve evidence.

## After approval
Create the work commit above, then normal Trellis WP7 archive and journal
bookkeeping commits. Do not push, do not start WP8, do not alter other tasks.
User explicitly confirmed this plan on 2026-09-18. Execute work commit, then normal archive and journal bookkeeping; no push or WP8.

## Approval requirement
.trellis/workflow.md §3.4 step 5 requires: "Present the plan once, ask for one-shot confirmation".
This plan is the concrete final review for that gate; implementation approval
already covered all development, fixes and validation completed above.
