# Independent reference boundary correction

This correction followed discovery, separate source-backed explanation and explicit
coordinator review authorization; it was not made to force a failing parity test
to agree. Before the change, DensePGVAMP(2) on
H=1e-140*[[1,.2j],[.3,1]], y=[1,1], sigma2=1 (complex128/float64)
returned finite x_soft approximately [1.3e-140, (1-.2j)e-140]. Backward of
x_soft.real.sum() returned raw_gaps=[NaN,NaN], raw_mu=[NaN,NaN]. Production
initially exhibited the same defect. Autograd anomaly located innovation/c;
real-component division alone still produced NaN in DivBackward0 because the
unweighted local quotient derivative overflows at this tiny positive c.

Source §14.5 defines K=W/c and r1=r2+innovation/c on resolved positive c.
Source §14.8 requires full actual-dependency differentiation, so detaching c,
raising the no-information threshold or suppressing NaNs would change the
contract. For c>0 the identity x/c=(x/sqrt(c))/sqrt(c) preserves the exact
formula while separating its derivative evaluation into representable stages.
The production implementation uses a real view; the oracle implements its own
explicit real/imaginary components. Neither imports or calls the other's
numerics. Only the two quotient evaluation sites changed in the reference.

Independent evidence beyond production/oracle parity:

- Both implementations' tests compare the tiny-channel posterior divided by
  amplitude to the analytic vanishing-channel limit H0^H*y/sigma2, with a
  non-diagonal complex H0 and both float64 amplitude=1e-140 and float32=1e-17.
- Mixed weak/zero/ordinary batches preserve information classification,
  uniform zero-channel probabilities and finite gradients. Total raw gradient
  agrees with ordinary member alone (weak contribution below precision).
- N=8/16/32 ordinary full-layer/parameter-gradient comparisons and gradcheck
  are retained at the original source tolerances, not weakened.
- Separate NumPy matrix/covariance/solve audits do not depend on either oracle
  or production computational helper.

No mathematical contract, parameter, threshold, jitter, message logic, API,
source tolerance or physical configuration changed. This repairs a numerical
evaluation-order defect within the approved finite-backward contract.
