# Planning validation — 2026-09-17

## Actual checks
- Created the parent and WP0 child using task.py create --no-start; parent/child
  links are reciprocal and both task.json statuses are planning.
- Completed and converged PRD requirements/acceptance/scope, technical design,
  ordered implementation plan, and curated context manifests.
- Ran task.py validate for both task directories after final context edits:
  exit code 0, all validations passed. WP0 has 11 real entries in each manifest;
  parent has 3 in each. No seed/example rows remain.
- Ran a separate Python structural audit: planning status, single child link,
  artifact presence, Markdown file links, JSONL paths/reasons, template residue,
  unchanged execution-spec SHA-256 and absence of product files all passed.
- Reviewed source section coverage and WP boundaries: WP0 reference computations
  are required; production detectors/physical chain/training/evaluation remain
  in later WPs. No source formula or acceptance gate was replaced.

## Known context warning and required handling
The authoritative source is 83,463 bytes; Trellis's per-file injection cap is
32,768. Both validations warn that automatic injection truncates the source.
Validation success does not mean the entire source was injected.
The FIRST entry in every new context manifest is
research/source-reading-contract.md (in this child task), which requires direct
bounded reading of the complete original source and hash verification before
any implementation or check. The original source stays in the manifests for
traceability. No Trellis runtime settings or source specification were changed.

## Non-execution boundary
No task start was run. No product code, package/config/test scaffolding, or
WP1–WP8 task was created. No product pytest, reference computation, physical
smoke, training or evaluation was executed. All product commands in implement.md
are plans, not results. Parent and WP0 await review in planning.
