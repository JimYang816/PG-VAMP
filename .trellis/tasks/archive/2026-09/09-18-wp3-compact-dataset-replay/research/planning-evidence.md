# WP3 planning evidence

Read-only inspection on 2026-09-18; no implementation or new product test run.

- `docs/CODEX_ENGINEERING_SPEC.md:666`: §§10.1–10.5 storage/randomness/replay/materialization/validation.
- `docs/CODEX_ENGINEERING_SPEC.md:769`: §11 excludes labels from detector inputs.
- `docs/CODEX_ENGINEERING_SPEC.md:1228`: §16.2 training mix and paired SNR frames/counts.
- `docs/CODEX_ENGINEERING_SPEC.md:1763`: §21.3 simulate/audit-data, no implicit training-time generation.
- `docs/CODEX_ENGINEERING_SPEC.md:1857`: WP3 acceptance; lines 1915 and 1946: replay/split/shared samples/labels and later integration.
- `src/pgvamp_ofdm/cli.py:69`: only inspect-config exists; no data package yet.
- `src/pgvamp_ofdm/config.py:269`: static size estimate, not an implemented output guard.
- `src/pgvamp_ofdm/channel/parameters.py`: explicit generator and PathParameters validation; caller validates frame CP.
- `.trellis/spec/backend/wp2-channel-contract.md`: implemented interfaces, precision/timing contracts.
- Archived WP2 and parent PRD: WP2 accepted, 194 passed / 3 skipped are historical results, not WP3 validation.

Parent baseline execution-spec SHA-256: A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B.
No external research needed: source specification and prerequisite interfaces resolve planning scope. Actual downstream algorithm checks remain explicit integration obligations.
