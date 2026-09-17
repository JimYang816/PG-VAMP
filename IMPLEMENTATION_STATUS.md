# Implementation status

WP0 implementation and independent Trellis verification are complete. The
changes await user review before commit/archive. See VALIDATION.md and the
task's machine-readable execution receipts. WP1 has not been started.

| Work package | Status |
| --- | --- |
| WP0 foundation, strict configuration, runtime, references, inspect-config | Implemented; CPU verification passed; awaiting review |
| WP1 modulation, frame and LFM | Not started |
| WP2 physical channel and effective model | Not started |
| WP3 datasets and splits | Not started |
| WP4 production MMSE and VAMP | Not started |
| WP5 production PG-VAMP and complete mathematical equivalence | Not started |
| WP6 training, checkpoint, inference and system smoke | Not started |
| WP7 evaluation and reports | Not started |
| WP8 complete delivery | Not started |

The WP0 references do not constitute production algorithm or physical-chain
acceptance. Resolving profiles does not constitute execution of their workloads.

WP0 evidence: both installed CLI entries, all required inspection profiles,
source §20 defaults, complete resolved configuration/provenance, independent
reference algebra/gradient/protection tests, Ruff and mypy. Final targeted and
full suites each report **75 passed, 1 skipped**. The skipped test requires CUDA
hardware unavailable in this CPU build; no CUDA numerical acceptance is claimed.

PG has exactly `2T` real parameters (16 at the default depth 8). Tests include
full-graph/identity/diagonal/zero/rank-deficient limits, independent QPSK
enumeration, N=8/16/32 exact linear solves, gradcheck, scale/dtype checks and
rejected/capped/no-information backward regressions. No tolerance was relaxed.
