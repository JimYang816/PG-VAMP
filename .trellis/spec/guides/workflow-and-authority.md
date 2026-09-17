# Authority and work-package gates

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§0, 22–26. It is the sole self-contained engineering specification.
Trellis specs, task PRDs, designs and implementation plans organize its rules;
they cannot modify, weaken or reinterpret them. When they conflict, the
execution specification wins. Detailed formulas remain in §§14–15; the
separate mathematical derivation is optional audit material, not a dependency.

The repository starts without product code. Treat prescribed paths and target
commands as future deliverables until actual files and execution evidence exist.
Record source sections in task acceptance criteria; check them against the full
source before marking a package complete. Do not import old project behavior.

Proceed in dependency order, passing each §22 gate before advancing:
WP0 → WP1 → WP2 → WP3 → WP4 → WP5 → WP6 → WP7 → WP8.

| Package | Responsibility (full acceptance remains in §22) |
| --- | --- |
| WP0 | Workspace/spec hashes, config/device/layout, independent correctness references, minimal CLI |
| WP1 | QPSK/allocation, OFDM/CP/frame, LFM and basic correlation |
| WP2 | Physical paths, CP validity, analytical H, independent waveform, pilot elimination |
| WP3 | Compact data, manifest/splits/random streams/replay/input validation |
| WP4 | Full-H MMSE and exact VAMP baselines with oracle equivalence |
| WP5 | Production PG-VAMP mathematical contract and gradient/property checks |
| WP6 | Training/checkpoint/resume/inference, math and full-size smoke |
| WP7 | Paired evaluation, metrics/statistics/timing/diagnostics/reports |
| WP8 | Delivery docs, actual CPU full-size smoke, explicit unexecuted experiments |

WP0 reference code is not authorization to skip ahead to WP5 production work.
Physical waveform/effective-model equivalence must pass before effective_fast
main training/evaluation. Mathematical and physical smoke precede substantive
training/comparison; the two-update smoke is itself an acceptance exercise.
Keep all §23 tests, not merely a task's abbreviated checklist.

Only actual executions support results. Record failures and “未执行” honestly;
code completion and adequate performance evidence are separate gates (§§18, 24).
User review boundaries belong in the active task; never infer approval for the
next WP from completion of a documentation bootstrap.
