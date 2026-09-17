# Python/PyTorch engineering guidelines

Authority: [CODEX_ENGINEERING_SPEC.md](../../../docs/CODEX_ENGINEERING_SPEC.md),
especially §§0–1. It is the sole, self-contained engineering specification.
These guidelines organize recurring obligations; they never replace, weaken
or reinterpret its mathematics, communication parameters, tests or WP order.
In any conflict, the execution specification wins. Read the relevant source
sections before implementation; resolve ambiguities explicitly rather than
inventing a contract in Trellis.

Current state: a new project without product source/tests. Paths below describe
the prescribed implementation, not existing or validated code. The backend
directory is Trellis's discovery location for the Python library/CLI; it does
not imply a web server, database or frontend.

| Guide | Long-term responsibility | Source sections |
| --- | --- | --- |
| [Structure](directory-structure.md) | src layout, modules, interfaces, dependencies | 0.3, 11, 19, 21 |
| [Runtime and configuration](runtime-config.md) | CPU/dtype defaults, strict config, validation | 3, 10.5, 15.2, 20–21 |
| [Reproducibility](reproducibility.md) | random streams, manifests, checkpoint, logs | 10, 15, 18, 21 |
| [Numerical contracts](numerical-contracts.md) | physical model, algorithm invariants, prohibitions | 2–14, 23 |
| [Verification](quality-guidelines.md) | independent oracles, pytest, truthful results | 0.3–0.4, 16–18, 22–24 |

Always also read [workflow and authority](../guides/workflow-and-authority.md).
The complete test list and acceptance gates remain in source §§22–23;
this index is not a reduced acceptance checklist.
