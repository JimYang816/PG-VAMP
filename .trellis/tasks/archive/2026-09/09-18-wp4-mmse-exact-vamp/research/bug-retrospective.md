# Bug Analysis: rejected finite-input candidate overflow contaminates backward

## 1. Root cause category
E (implicit assumption) and D (coverage gap): finite message inputs were assumed
to make evaluating the candidate safe. The implementation computed a candidate
division, tested its finiteness, and discarded an infinite result. Forward kept
the previous valid message, but backward could still evaluate 0 * Inf through
the discarded division and produce NaN gradients.

Concrete reproduction: float64 mean=-1e308, r1=1e308, gamma1=1 and
vbar=alpha=0.5. A separate float32/float64 regression distinguishes numerator
overflow from a finite numerator whose quotient overflows.

## 2. Why the initial guard was insufficient
The existing eligibility tests reject nonfinite inputs and unrepresentable
reciprocals; they did not check the arithmetic range of the candidate mean.
Post-hoc finiteness masking checks a forward value, not backward evaluation
safety. No source formula or protection threshold was changed by the fix.

## 3. Prevention mechanisms
| Priority | Mechanism | Specific action | Status |
| --- | --- | --- | --- |
| P0 | Runtime ordering | Validate numerator and per-component quotient representability before division | Implemented |
| P0 | Regression | Both precisions, numerator and quotient overflow, previous-message retention, all involved backward gradients finite | Passed in implementation full regression |
| P1 | Independent review | Review protected backward paths as well as forward outputs and oracle equivalence | Requested in check-focus.md |
| P1 | Executable spec | Record branch-before-unsafe-arithmetic contract in wp4-detector-contract.md | Done |

## 4. Systematic expansion
The same reasoning applies to capped/rejected reciprocal paths; those already
have dedicated tests and remain covered. Future WP5 consumes these production
helpers and must preserve their graph behavior. References remain independent;
do not rewrite a reference to match a production fix. This direct helper fixture
is not evidence of a reachable failure in the existing oracle's complete VAMP
loop. Any actual oracle defect requires its own source-grounded reproduction.

The validation environment also matters: a system interpreter without the
editable project cannot run subprocess/CLI tests. Use the existing project
virtualenv; preserve initial failed receipts and separate corrected results.

Independent review also found that a mutation-based integration test reused the
already-mutated sample for its second detector. Reload a fresh physical sample
for each detector's baseline; otherwise the second changed-label assertion can
compare identical labels and miss its intended intervention. The API itself was
label-free; this was a coverage gap, not evidence of product label leakage.

## 5. Knowledge capture
- Updated `.trellis/spec/backend/wp4-detector-contract.md` and backend index.
- Added float32/64 regression in `tests/test_detector_qpsk.py`.
- Preserved initial interpreter/launcher failures in task research receipts.
- This is an application repository, not Trellis's template source; there is
  no applicable `src/templates/markdown/spec` copy to sync.
- Include these records in the WP4 work commit after the project's one-shot
  commit-plan approval gate; no independent spec-only commit before that gate.
