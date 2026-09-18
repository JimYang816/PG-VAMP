# WP4 proposed work commit

Implementation and independent review passed. User confirmed this work commit
and normal archive/journal completion.

## Proposed commit
`feat: 完成 WP4 线性 MMSE 与精确 VAMP 基线`

One coherent commit contains approved detector modules, 32-layer config, tests,
product documentation, executable backend contract, planning/verification evidence
and parent progress. Planning files created in the preceding turn belong to the
same approved WP4 task. Generated datasets stay under ignored runs/.

## Exact file list
- `.trellis/spec/backend/index.md`
- `.trellis/spec/backend/wp4-detector-contract.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/prd.md`
- `.trellis/tasks/09-17-pgvamp-complete-engineering/task.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/check.jsonl`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/commit-plan.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/design.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/implement.jsonl`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/implement.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/prd.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/bug-retrospective.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-focus.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-numerics.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-numerics.py`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-physical.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-physical.py`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-report.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-source-hashes.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-validation.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/check-validation.py`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/implementation-evidence.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/implementation-lint.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/implementation-physical.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/implementation-source-hashes.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/implementation-validation.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/initial-system-python-validation.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/main-final-verification.json`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/planning-evidence.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/run-validation.py`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/source-excerpts.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/research/svd-no-information.md`
- `.trellis/tasks/09-18-wp4-mmse-exact-vamp/task.json`
- `IMPLEMENTATION_STATUS.md`
- `README.md`
- `VALIDATION.md`
- `configs/vamp_reference_32.yaml`
- `src/pgvamp_ofdm/algorithms/__init__.py`
- `src/pgvamp_ofdm/algorithms/base.py`
- `src/pgvamp_ofdm/algorithms/messages.py`
- `src/pgvamp_ofdm/algorithms/mmse.py`
- `src/pgvamp_ofdm/algorithms/qpsk.py`
- `src/pgvamp_ofdm/algorithms/vamp.py`
- `tests/test_detector_inputs.py`
- `tests/test_detector_qpsk.py`
- `tests/test_mmse.py`
- `tests/test_vamp.py`

## Unrecognized dirty files
None. All listed paths were created or edited by the current WP4 planning,
implementation and review session. No generated dataset is included.

## Verified result
313 passed / 5 CUDA skipped; Ruff/format/mypy and config inspections pass.
Independent NumPy layer maximum difference 2.78e-15; real 400-dimensional input
and full-result label isolation pass. Main session verified all 64 final source
fingerprints and the physical manifest hash. Independent reference and source
specification unchanged. See research/check-report.md and main-final-verification.json.

## Approval boundary
The project workflow Phase 3.4 step 5 requires: "Present the plan once, ask for
one-shot confirmation". Confirmation authorizes this work commit followed by
normal WP4 archive and journal completion; no push. WP5 remains unstarted.
