# Bug Analysis: weak informative PG backward and diagnostic scaling

## 1. Root cause category
- **E — Implicit assumption**: representable W/c and innovation/c values were
  assumed to imply representable local quotient derivatives. Very small resolved c
  violates that assumption; local x/c² can overflow before upstream weighting.
- **D — Test coverage gap**: diagonal 1e-20/1e-160 cases covered information
  classification but missed non-diagonal 1e-140 float64 (1e-17 float32) backward.
  The independent reviewer found finite output with NaN raw-parameter gradients.
- A separate diagnostic norm ratio squared extreme finite energy entries, yielding
  inaccurate/nonfinite logging despite unchanged physical detection under scaling.

## 2. Why an initial repair was insufficient
Real-component division avoids complex division's own tiny-denominator arithmetic,
but DivBackward0 can still form x/c². The successful local repair uses two real
divisions by sqrt(c). It preserves the exact source quotient and original trace
threshold, rather than suppressing weak observations or detaching dependencies.
Diagnostic numerator/denominator norms now share one cancelling scale.

## 3. Prevention mechanisms
| Priority | Mechanism | Action | Status |
| --- | --- | --- | --- |
| P0 | Regression | Weak/zero/normal mixed batches in float64/float32; finite gradients, nonzero weak output and ordinary-gradient agreement | Added by independent checker |
| P0 | Independent expectation | Weak-channel leading-order posterior H^H y / sigma2 | Added |
| P1 | Scale regression | Safety diagnostic invariant under joint H/y scale 1e±100 and noise scale 1e±200 | Added |
| P1 | Numerical contract | Document finite-forward vs finite-backward distinction and stable quotient evaluation | Updated |

## 4. Systematic expansion
The same concern applies to precision reciprocals and capped/rejected message
branches; WP0/WP4 already own distinct regression coverage for those paths.
No shared helper or oracle was changed to force agreement. After separate review,
the oracle's independently reproduced identical boundary was locally repaired;
see oracle-fix-review.md and check-oracle-correction.md. Normal-scale oracle
gradient parity and gradcheck still apply; the new extreme boundary also uses
an independent asymptotic and mixed-batch check. CUDA remains a hardware limitation.

## 5. Knowledge capture
Updated `.trellis/spec/backend/numerical-contracts.md` and
`wp5-detector-contract.md`, with executable cases and wrong/correct examples.
This repository has no product template copy of these project-specific specs.
Spec updates belong in the proposed WP5 work commit after workflow §3.4 confirmation.
Final numerical acceptance and command receipts are owned by research/check-*;
this retrospective is not a substitute for those executions.
