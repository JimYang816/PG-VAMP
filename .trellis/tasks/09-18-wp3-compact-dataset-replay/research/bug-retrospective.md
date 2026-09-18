# WP3 review retrospective

## 1. Root cause category
- B — cross-layer contract: compact generator creates canonical IDs and hashes,
  but materialized inputs can be supplied independently. Initial loading checked
  a 64-character string and frame prefix rather than the hash alphabet and full
  SNR-copy/block identity grammar.
- D/E — test coverage gap and implicit assumption: a one-file capacity reserve
  was reused for a request that writes train, validation and test exports. A tiny
  selected split may still carry the entire dataset's split table.
  A fixed per-frame metadata reserve also needs the schema to enforce canonical
  64-character frame/channel IDs, not accept arbitrarily long nonempty strings.
- C — change propagation: compact records chose explicit pilot tensors, so the
  earlier static compact estimate using a pilot seed needed to change too.

## 2. Why initial checks missed them
Happy-path roundtrips use the same producer and consumer. They establish replay
but do not establish rejection of independently malformed metadata. The initial
matrix-payload arithmetic was correct; it did not exercise total serialized
metadata or multiple output files. These were review findings in a new feature,
not recurring failures of previously accepted WP3 code.

## 3. Prevention mechanisms
| Priority | Mechanism | Concrete action | Status |
| --- | --- | --- | --- |
| P1 | Runtime validation | Validate hash alphabet, canonical SNR/block IDs, lineage and compact copy/config consistency | Done; final check-report.md verifies regressions |
| P1 | Budget accounting | Sum per-file estimates and account for full split table/config metadata before generation | Done; final check-report.md verifies regressions |
| P1 | Regression coverage | Mutate valid serialized inputs; test three-export guard and small-split/large-metadata case | Done; final 248 passed / 3 CUDA skipped |
| P2 | Contract documentation | Record persisted-boundary and aggregate-budget rules in WP3 backend spec | Done |

## 4. Systematic expansion
Apply the same boundary reasoning to future checkpoint/manifest linkage and
training/evaluation input adapters. A valid tensor shape or nominal digest length
does not establish metadata consistency. Keep physical identity separate from
runtime/storage choices so replay tests can compare backend/dtype variants.
No change to mathematical equations, numerical tolerances or WP4 scope follows
from these fixes. No template tree exists for this project-specific Python spec;
do not introduce Trellis product templates here.

## 5. Knowledge capture
- Updated `.trellis/spec/backend/wp3-data-contract.md` with executable boundary
  and budget obligations, required regressions and wrong/correct examples.
- Review findings and final command receipts belong in task research, with
  implemented status reported only after the independent check completes.
- Spec changes are included in the WP3 work-commit plan; user approved the
  project's one-shot commit plan and completion workflow.
