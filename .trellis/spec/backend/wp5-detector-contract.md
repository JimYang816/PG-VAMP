# WP5 production PG-VAMP contract

Authority: `docs/CODEX_ENGINEERING_SPEC.md` §§11, 14, 17.4, 20, 22/WP5 and 23.
This documents the implemented interface; VALIDATION.md records actual test evidence.

## 1. Scope / trigger
Read when consuming or changing production PG-VAMP, its gradients, layer diagnostics,
or its integration with WP3 data and WP6 training. Independent DensePGVAMP
is an oracle, not the production path. See wp6-training-contract.md for training,
checkpoint and inference; unified evaluation remains WP7 work.

## 2. Signatures
```text
PGVAMPDetector(depth=8, *, dtype=torch.float64, device="cpu",
    rho_hi_db=0, rho_lo_db=-60, min_gap_db=0.5, temperature_db=3,
    init_mu=0.8, jitter=0, mask_mode="soft")
model(H, y, sigma2, *, return_diagnostics=False,
      return_layer_outputs=False) -> DetectionResult
model.detect(H, y, sigma2, *, return_diagnostics=False) -> DetectionResult
model.thresholds() -> (rho[T], mu[T])
```
Export: `pgvamp_ofdm.algorithms.PGVAMPDetector`. Both forward and detect preserve
autograd. Inference callers can explicitly use `torch.no_grad()`.

## 3. Contracts
- Only raw_gaps[T] and raw_mu[T] are learned: exactly 2T real scalars, 16 by default.
  Parameter dtype/device must match sigma2 and H; no silent conversion.
- H[B,N,N] and y[B,N] use complex128/complex64, sigma2[B] uses paired float64/float32
  and is positive finite. Inputs share a device. Labels and identities stay with caller.
- Output is final posterior x_soft, class_hat, bits_hat and probabilities; reuse WP4
  production QPSK/messages. No truth input, oracle import or checkpoint state in baselines.
- Strict source threshold parameterization, directed soft masks, diagonal one, zero
  off-diagonal edges, column energy normalization without normalizing H itself.
- d uses (1-M²), ell uses nonnegative exclusive row sums; complete H remains in A
  and the centered residual. No zero-centered estimator or added noise compensation.
- One actual Cholesky factor per layer supports innovation and W. c=Re tr(WH)/N,
  alpha2=1-c, and gamma1 derives from the exact squared-Frobenius variance.
  Explicit jitter changes only P and is recorded; all actual operators share that P.
- The no-information tolerance is max(gamma_(4N)*sum_ij|W_ij||H_ji|/N, tiny),
  with gamma_(4N)=4N*eps/(1-4N*eps). It bounds the computed contraction, not the
  entire factorization error. No unit-scale epsilon cutoff. Significant negative c
  is a hard failure; a finite unresolved contraction produces zero r1/gamma1.
- Resolved tiny c still requires stable backward arithmetic. Evaluate W/c and
  innovation/c through real components and two divisions by sqrt(c), preserving
  c and the source formulas. A direct quotient can produce an unrepresentable
  local x/c² derivative even with finite output and finite total parameter derivative.
  Real component division alone does not fix that second failure mode.
  DensePGVAMP owns the same algebraic repair locally after a separately reviewed
  reproduction; it does not import this production implementation.
- An entirely uninformative batch retains zero parameter dependence for valid zero
  gradients. Invalid nonlinear messages preserve the prior valid message; valid caps
  preserve candidate means. Final layer does not generate unused extrinsic messages.
- Routine counters are per sample. layer_summaries contains detached small log copies:
  rho/mu/c/c_tolerance/jitter, candidate_edges/all_off_diagonal_edges, both effective
  edge ratios, safety_relative and safety_zero_baseline. Empty denominators are explicit
  and yield zero ratios. safety_relative is ||ell||_2/||diag(G)||_2, zero if both vanish.
  Calculate both norms after a common cancelling scale to avoid overflow/underflow
  in the logging-only ratio; this does not normalize physical H or change detection.
- Detailed layers retain differentiable states and large matrices for math tests;
  routine output omits these. return_layer_outputs=True adds only differentiable
  posterior means for WP6 loss, without retaining the full detailed state list.
  Detaching a log copy must not detach forward quantities.
- Exact dense work can remain O(N³) per layer; 2T parameters and soft-edge ratios are
  not sparse-computation or low-memory claims. No cross-call old-parameter graph cache.

## 4. Validation and error matrix
| Condition | Behavior |
| --- | --- |
| Bad input shape/dtype/device/nonfinite values/nonpositive noise | ValueError |
| Illegal depth, threshold span, temperature, mu, jitter, mask mode | ValueError |
| Parameter dtype/device mismatch or nonfinite raw parameters | ValueError |
| Explicit unavailable CUDA | Existing readable runtime error; no fallback |
| Invalid computed P/W/innovation/trace/variance or failed factorization | Contextual FloatingPointError; no hidden jitter |
| Finite unresolved trace | Count no_information; zero soft estimate and uniform posterior |
| Invalid nonlinear candidate or underflow | Prior valid message retained; protection counted |

## 5. Good / base / bad cases
- Base: CPU, complex128, T=8, B=1, real physical N=400 inputs from detection_inputs.
- Good: nondegenerate ICI for gradient tests; mixed zero/rank-deficient/informative
  batch; explicit complex64; H=1e-20 I vs H=1e-160 I in float64.
- Bad: fixed eps discards weak information; mask or gamma detached; labels enter
  detect; random trace or exact-VAMP precision replaces the source PG contract.

## 6. Required tests
Independent forward and raw-parameter gradients; finite-difference gradcheck;
threshold/mask/energy/PSD/Loewner; B order/Hermitian and single-layer objective;
conditional real-expanded Jacobian of xhat2 wrt r2; complete covariance trace;
full-graph exact-VAMP and mu independence; identity/diagonal/zero/rank-deficient;
scale invariance; protected finite backward; dtype/device and one factor per layer;
diagnostic denominators; fresh physical sample label isolation for all three detectors.
Include non-diagonal weak channels at float64 amplitude 1e-140 and float32 1e-17
mixed with zero and ordinary samples: weak output remains informative, all gradients
are finite, and ordinary-sample gradients are not contaminated. Safety diagnostics
must remain finite and scale invariant for joint H/y/sigma2 scaling by 1e±100/1e±200.
Source tolerances apply. Degenerate/full-graph gradients may naturally be zero.

## 7. Wrong vs correct
Wrong: freeze precision or mask in production because conditional divergence holds
them fixed. Correct: freeze only the local Jacobian test's conditions; differentiate
all actual dependencies in the multilayer loss.

Wrong: call PG-VAMP's r1 Jacobian the source alpha2. Correct: the linear posterior
xhat2 has Jacobian I-WH and normalized real trace alpha2=1-c.

Wrong: production and oracle share the same numerical routine so parity proves it.
Correct: production reuses production helpers; oracle owns independent numerics and
matrix properties/covariance have independent checks beyond implementation parity.

Wrong: finite forward and diagonal weak-channel tests prove stable training.
Correct: exercise non-diagonal weak information with a nonzero-upstream-gradient
loss; inspect backward intermediates and mixed-batch contamination before accepting.
