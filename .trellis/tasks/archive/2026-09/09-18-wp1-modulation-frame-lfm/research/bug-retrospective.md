# Bug Analysis: finite zero is not evidence of a representable correlation

## 1. Root Cause Category
- E (implicit assumption) and D (coverage gap): the initial correlation checked
  finiteness/overflow and prefix cancellation, but finite zero can also mean a
  nonzero square/product underflowed. The epsilon denominator then hid the loss.
- E (input assumption): Allocation.validate assumed dataclass type annotations
  guaranteed actual field types; hand-constructed malformed data escaped with
  AttributeError or accepted non-integer scalar metadata.

## 2. Why fixes failed
There was no repeated failed repair loop. Initial normal-scale tests and zero
window protection passed but did not cover positive-to-zero underflow. Independent
review reproduced float64 amplitude 1e-100 and float32 amplitude 1e-15 energy
products becoming zero, plus non-tensor allocation fields.

## 3. Prevention mechanisms
| Priority | Mechanism | Action | Status |
| --- | --- | --- | --- |
| P1 | Runtime | Reject nonzero sample energy, numerator square or positive energy product lost to zero | Implemented by independent checker |
| P1 | Tests | Float64/float32 product/partial-sample underflow regressions; retain representable small-scale epsilon formula test | Added; final execution recorded in check receipts |
| P1 | Runtime/tests | Validate allocation tensor fields and true integer scalar metadata before use | Implemented with malformed-field regressions |
| P2 | Spec | Capture explicit error behavior and counterexample in WP1 waveform contract | Updated |

## 4. Systematic expansion
Overflow-only checks cannot certify every squared/normalized computation.
Differentiate exact zeros, numerical underflow and precision loss before returning
success. This complements WP0's branch-before-invalid-reciprocal lesson; it does
not justify changing WP0 math or revising the fixed correlation epsilon/threshold.
Keep the changes in WP1 validation boundaries; no generalized numerical framework
or additional WP was created.

## 5. Knowledge capture
- Updated .trellis/spec/backend/wp1-waveform-contract.md with executable errors,
  regression cases and wrong/correct examples.
- No src/templates/markdown/spec tree exists in this Python project, so the
  Trellis-product template-sync instruction is not applicable here.
- Spec changes belong to the WP1 work commit batch, pending workflow §3.4
  one-shot commit-plan confirmation; no independent retrospective commit is made.
