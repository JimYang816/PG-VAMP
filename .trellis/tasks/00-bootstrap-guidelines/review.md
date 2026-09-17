# Bootstrap review — 2026-09-17

## Scope and evidence
Read all 2,102 lines of docs/CODEX_ENGINEERING_SPEC.md, including §§0–26.
The source remained unchanged, verified by SHA-256 before/after:
`A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
The repository contains Trellis/tooling and documents, with no product source,
tests, configuration profiles or pyproject.toml. Prescribed paths in specs
are explicitly future implementation locations.

## Source consistency review
| User requirement | Spec location | Source |
| --- | --- | --- |
| Python/PyTorch and src layout | backend/directory-structure.md | §§11, 19, 21 |
| CPU default, explicit CUDA, complex128 | backend/runtime-config.md | §§15.2, 20–21 |
| Strict configuration / unknown keys | backend/runtime-config.md | §§10.5, 20 |
| RNG, manifest, checkpoint, safe resume | backend/reproducibility.md | §§10, 15, 18, 21 |
| Numerical prohibitions / physical link | backend/numerical-contracts.md | §§2–14 |
| pytest / independent references | backend/quality-guidelines.md | §§0.3, 6.6, 19, 23 |
| Immutable PG-VAMP contract | backend/numerical-contracts.md | §14 |
| WP0–WP8 order and gates | guides/workflow-and-authority.md | §22 |
| Only actual results | backend/quality-guidelines.md | §§18, 23–24 |
| Source always wins | both indexes and all topic files | §0.1 |

No conflict found in this document review. In particular, checkpoint inference
inherits checkpoint dtype (§21.5), rather than being silently recast by the
general complex128 default; WP0 creates references while WP4/WP5 implement
production algorithms. Mathematical fixtures are not physical smoke tests.
Detailed formulas, parameter tables and the complete mandatory test list remain
in the source. No new formula, tolerance relaxation or WP reordering was added.

Removed frontend/database/error/logging template files; error and logging rules
are consolidated into runtime, reproducibility and verification specs. Replaced
generic reuse/cross-layer guides, including unrelated Trellis-product advice,
with project rules that protect independent oracle calculations.

## Actual checks
- Ran Trellis session, phase and package discovery commands; reused and activated
  the existing docs-only bootstrap task, without creating a WP task.
- Ran a Python document audit: all 10 spec Markdown files have valid relative
  file links and authority references; no checked template residue remains.
- Verified the source SHA-256 and absence of src/, tests/, configs/, pyproject.toml.
- Manually reviewed rules against source sections in the table above.
- Product pytest, physical smoke, training and evaluation: **未执行**; no product
  implementation was requested or created. These document checks are not
  numerical or engineering acceptance tests.

## Handoff
Awaiting user review. The bootstrap task stays unarchived; WP0 has not started.
