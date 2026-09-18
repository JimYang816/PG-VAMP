# WP4 production baseline detector contract

Authority: `docs/CODEX_ENGINEERING_SPEC.md` §§11–13, 14.6, 20, 22/WP4 and 23.
This is an executable-interface guide, not a replacement for the source formulas.
Actual validation status lives in VALIDATION.md and the WP4 task evidence.

## 1. Scope / trigger
Read when consuming physical detector inputs, changing either production
baseline, reusing production QPSK/messages in WP5, or reviewing numerical
protection and diagnostics. Training/evaluation integration remains later work.

## 2. Signatures
```text
MMSEDetector().detect(H, y, sigma2, *, return_diagnostics=False) -> DetectionResult
VAMPDetector(iterations=8).detect(H, y, sigma2, *, return_diagnostics=False)
    -> DetectionResult
DetectionResult(x_soft, class_hat, bits_hat, probabilities, diagnostics)
qpsk_posterior(r, gamma) -> QPSKPosterior(mean, probabilities, variance, vbar, alpha)
nonlinear_message(mean, r1, gamma1, vbar, alpha, previous_r, previous_gamma)
    -> MessageUpdate(r, gamma, rejected, capped, underflow)
```
Detector exports live under `pgvamp_ofdm.algorithms`; posterior and message
helpers live in its `qpsk` and `messages` modules. Both detector modules have zero
trainable parameters. The supplementary config is `configs/vamp_reference_32.yaml`.

## 3. Contracts
- Input H[B,N,N] and y[B,N] are finite complex128/complex64; sigma2[B] is strictly
  positive finite float64/float32 respectively. All tensors share one device.
  No implicit casts, device moves, extra truth arguments, input mutation or cache.
- WP3 items have no batch dimension: project via detection_inputs, then explicitly
  stack/unsqueeze. Keep sample IDs, hashes and targets with the caller.
- x_soft is complex [B,N], class_hat int64 [B,N], bits_hat uint8 [B,N,2]; real bit
  precedes imaginary bit. Use the existing QPSK mapping and lowest-class tie rule.
- MMSE returns the raw full-H Cholesky linear estimate and probabilities=None.
  It does not run the QPSK posterior. Name: `MMSE (linear)`.
- VAMP performs one reduced SVD per detect, reuses it for all fixed iterations,
  and has no cross-call factor cache, learned state, damping or adaptive stopping.
  Compute alpha2 and c separately, and preserve centered full-H innovation.
- VAMP returns final posterior mean and real probabilities[B,N,4], not extrinsic
  messages. Default depth is 8; 32 is supplementary and distinctly named.
- Scalar nonlinear acceptance uses the exact source 1e-6 denominator margin,
  1e-10 precision floor and 1e8 cap. Reject invalid candidates before undefined
  arithmetic; keep previous valid messages. A valid capped candidate keeps its
  candidate mean. Do not form an overflowing reciprocal then mask it away.
  Finite message inputs do not guarantee a representable candidate mean:
  validate the numerator and quotient range before division. A rejected
  overflowing candidate can otherwise leave NaN gradients through zero-weight
  backward paths even when the retained forward message is finite.
- Stable tanh/sech and class probabilities are independently tested against
  four-point enumeration. Production helpers may be shared by future PG-VAMP;
  independent reference implementations must not import these numerical helpers.
- No-information is scale-aware: a weak representable channel must not be
  discarded just because c < eps. The spectral positive-sum bound is conditional
  on computed SVD factors; it is not an SVD accuracy or convergence theorem.
  Zero/subnormal-information cases return zero soft symbols and uniform QPSK.
- Routine diagnostics retain per-sample no_information, message_rejected,
  precision_capped and posterior_variance_underflow counts. VAMP detailed mode
  additionally retains layer vector/scalar states, with r2/gamma2 at layer entry.
  The final layer does not generate an unused nonlinear extrinsic update.
- No hidden jitter, explicit inverse, CG, random trace, forward detach or H
  normalization. SVD cold-start cost belongs in later timing measurements.

## 4. Validation and error matrix
| Condition | Behavior |
| --- | --- |
| Empty/rectangular/mismatched shape, wrong dtype/device, nonfinite inputs, sigma2 <= 0 | ValueError before detection |
| Invalid VAMP iteration count (including bool) | ValueError |
| Nonfinite computed operator or failed factorization | FloatingPointError with algorithm/batch/dtype/device and applicable layer context |
| Invalid nonlinear candidate | Preserve previous valid message, increment rejection count |
| Zero posterior variance | Conservative rejection and underflow count; no divide-by-zero |
| Valid precision over cap | Keep candidate mean, cap precision, increment cap count |
| Zero/numerically uninformative H | VAMP zero soft output, uniform probabilities and no-information count |

Hard numerical failure cannot be relabeled no information or silently removed
from later metric denominators. Detection API does not own sample IDs; caller
must associate them with exceptions.

## 5. Good / base / bad cases
- Base: CPU complex128, B=1, N=400 from physical WP3 data; VAMP-8 and raw MMSE.
- Good: mixed zero/rank-deficient/informative batch; separate weak H=1e-20 I
  and H=1e-160 I float64 cases; altered targets with identical detector inputs.
- Bad: use fixed eps as a signal cutoff, add a posterior to MMSE, reuse a
  production helper inside the reference, or silently replace failed samples.

## 6. Tests required
Normal-equation residual / independent solve / diagonal MMSE; 8/16/32 dimensional
layerwise SVD-vs-Cholesky VAMP; four-point posterior enumeration; rejection,
cap, underflow and reciprocal-boundary backward; dtype/device and input rejection;
finite-input numerator/quotient overflow rejection with finite backward;
zero/rank-deficient/mixed systems; scale invariance; zero learned parameters;
one SVD per call; forbidden API; physical 400-dimensional shared-input and target
isolation. Use source complex128 tolerances and independently justified
complex64 tolerances. A two-baseline physical forward is not full system smoke.
Reload a fresh sample for each detector's mutation-based label-isolation test;
otherwise a previous detector's mutated labels can make the next comparison
vacuous. Verify both changed-label and absent-label predictions.

## 7. Wrong vs correct
Wrong: `1 / vbar` followed by masking/clamping guarantees finite derivatives.
Correct: branch before unsafe reciprocals; rejected/capped paths must also have
finite backward behavior. Finite forward alone does not establish this.

Wrong: matching the production QPSK helper with another call to that helper
establishes independent VAMP correctness.
Correct: use four-point enumeration and the separate Cholesky reference, while
production detectors share only the intended production mapping/posterior APIs.
