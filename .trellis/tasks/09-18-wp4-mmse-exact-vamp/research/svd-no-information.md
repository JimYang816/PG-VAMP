# SVD no-information arithmetic boundary

For computed singular values and positive gamma2, each term
q_i = lambda_i / (lambda_i + gamma2) is nonnegative. The computed
c = sum(q_i)/N therefore has no subtractive cancellation. In normal arithmetic,
forming lambda, denominator, quotient and the sum incurs a conservative relative
bound gamma_(4N) = 4N eps / (1 - 4N eps). Its absolute scale is c itself:
tau = max(gamma_(4N) c, finfo.tiny). This is a rounding bound for evaluation
of the spectral sum, conditional on computed SVD factors, not a universal bound
on SVD backward error. SVD/alpha/delta nonfiniteness remains a hard failure.

The independent Cholesky oracle computes trace(W H)/N and uses the same
gamma_(4N), but its absolute scale is sum_ij |W_ij||H_ji|/N to account for
cancellation in complex matrix products. These scales need not coincide.
Both regard subnormal c as numerically uninformative (normal-arithmetic relative
error bounds no longer apply); neither uses eps as an absolute signal cutoff.

For H=a I, sigma2=gamma2=1: c=a²/(1+a²), and the oracle scale equals c.
a=0 yields c=0; a=1e-160 yields approximately 1e-320 < float64 tiny and is
uninformative; a=1e-20 yields 1e-40 >> tiny, so remains informed despite c<eps.
Tests compare these decisions and resulting outputs directly to the unchanged
Cholesky oracle. No arbitrary fixed epsilon is added to a denominator.
