# Bootstrap PG-VAMP engineering guidelines

## Goal and authority
Organize durable project rules from docs/CODEX_ENGINEERING_SPEC.md (v2.0).
That document is the sole authoritative engineering specification; Trellis
specs, PRDs, designs and implementation plans cannot alter its contracts.
The user explicitly authorized continuing this existing bootstrap task.

## Scope
Documentation only: reshape .trellis/spec and record this task's review.
The repository has no product source, tests or package manifest yet.
All src/config/test paths in the guidelines are prescribed future paths from
section 19, not claims of existing implementations.
Remove frontend/database and Trellis-product template guidance.
Do not create product code, implement WP0, run training or start evaluation.

## Acceptance
- Cover Python/PyTorch src layout, device/dtype, strict config, reproducibility,
  manifest/checkpoint, numerical prohibitions, independent oracles and pytest.
- Preserve the PG-VAMP contract and WP0–WP8 dependency order.
- Link detailed formulas and parameter/test tables to the source sections.
- Check links, template removal and source consistency; report actual checks only.
- Present the guideline files and pause for user review before WP0.

## Review state
Draft prepared and document checks passed; see review.md for evidence and
source traceability. Awaiting user review. Leave this task unarchived and do not
begin WP0 until the user reviews and authorizes further work.
