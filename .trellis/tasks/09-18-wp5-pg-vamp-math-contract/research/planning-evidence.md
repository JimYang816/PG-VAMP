# WP5 planning evidence — 2026-09-18

Repository main at `5c4e745`, clean before task creation. Source SHA-256 verified:
`A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
User authorizes creation and planning only. No numerical executions or product changes.

## Evidence anchors
- `docs/CODEX_ENGINEERING_SPEC.md:769`: shared label-free API.
- `docs/CODEX_ENGINEERING_SPEC.md:910`: complete PG contract, with parameters at 914,
  majorizer at 951, centered residual at 989, trace/variance at 1016, posterior/guards
  at 1052, differentiation at 1112, diagnostics at 1345, WP5 at 1869 and tests at 1920.
- `src/pgvamp_ofdm/reference/dense_pg_vamp.py`: independent DensePGVAMP implements
  topology, exclusive sums, Cholesky, trace bound, variance, posterior and guards.
  It must not become a production wrapper or import production numerical helpers.
- `src/pgvamp_ofdm/algorithms/{base,qpsk,messages,vamp}.py`: production contracts,
  protected posterior/messages and exact SVD baseline available for integration.
- `tests/test_reference.py:18`: FullGraph; line 119 raw-parameter gradcheck;
  lines 220/235 weak-channel and all-uninformative tests. These do not establish
  future production equivalence, conditional Jacobian or complete WP5 acceptance.
- `configs/base.yaml:91`: PG defaults already exist, with packaged base counterpart.
- `.trellis/spec/backend/wp4-detector-contract.md`: input/results, finite-backward
  protection and fresh-sample label-isolation lessons.

## Predecessor evidence
WP4 archive `research/check-report.md` records independent review, 313 passed/5 CUDA skipped,
Ruff/format/mypy, independent NumPy and physical audit. These are historical results,
not rerun and not claimed as WP5 results. Parent links alone are not dependency verification.

## Decisions and remaining gate
Use production helpers and existing configuration, preserve independent oracle.
Trace threshold follows actual W/H contraction; its limited meaning and audit cases are
explicit in design.md. Full graph is a test injection, not a public hard-mask mode.
Default jitter stays zero; existing explicit nonzero configuration uses one actual P.
Training/system smoke/evaluation remain WP6/WP7. No blocking user-owned scope decision remains.
PRD convergence pass consolidates goal, authority, R1–R6, A1–A10 and exclusions with source mappings.
Technical property checks are future acceptance work, not claims of proven implementation.
At the end of planning, implementation authorization was outstanding. The subsequent
user message explicitly approved this plan and authorized implementation; task status
is now in_progress. The historical planning observations above remain unchanged.
