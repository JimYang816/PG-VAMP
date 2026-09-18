# Independent property review notes — implementation coordination

These are algebraic review assertions, not numerical test results. The source
specification remains authoritative. Prepared by the coordinating session while
the implementation agent owns production code and tests.

## B order and objective
For the actual P >= A > 0, write C=P^(-1/2) A P^(-1/2), so 0<C<=I.
Then B=P^(-1/2)[(1+mu)I-mu C]P^(-1/2). For 0<=mu<=1 and 0<lambda<=1,
lambda[(1+mu)-mu*lambda] lies in (0,1], since
1-lambda[(1+mu)-mu*lambda]=(1-lambda)(1-mu*lambda)>=0.
This yields 0<B<=A^-1, including consistent nonnegative jitter in P.
Tests should form reference inverse actions through solve, not call inverse.
For q(r)=0.5*r^H A*r-Re(b^H*r), the centered update r+B(b-Ar) cannot
increase q. Keep A/b/r fixed for that single-layer assertion; no BER theorem follows.

## Conditional Jacobian and covariance
Fix H, sigma2, gamma2, M and mu in a single linear layer. The derivative of
xhat2=r2+W(y-H*r2) with respect to r2 is I-WH; its real-expanded normalized
trace is 1-Re(trace(WH))/N=alpha2. Differentiating r1 instead would test a
different map. The ordinary full-network gradient must still propagate through
all these quantities' actual dependencies; freezing them is test-local only.
With K=W/c and r2=x+v, r1-x=(I-KH)v+Kw. Independent circular v and w with
the source covariances produce Cov=(I-KH)(I-KH)^H/gamma2+sigma2*K*K^H.
The normalized trace equals the sum-of-squares variance formula.

## Loewner test construction
For two entrywise ordered real masks 0<=M<=M'<=1 with unit diagonal and
the same H, G and G' retain the same diagonal. The increase of each off-diagonal
Gram contribution has magnitude bounded by the corresponding increase in
sum_k m_ki*m_kj*|H_ki|*|H_kj|. The analytical ell decrease supplies the row sum
of those bounds, so Gbar(M)-Gbar(M') is Hermitian diagonally dominant with
nonnegative diagonal, hence PSD. Compare Gbar (not merely G) in this test.

## Numerical boundary review
The trace contraction bound controls summation of computed W/H products, not
all errors in forming W. Include weak-but-normal information, cancellation,
rank deficiency and explicit jitter fixtures. A significant negative c is a hard
failure; an uninformative finite contraction is counted separately. Acceptance
does not permit raising a fixed epsilon cutoff until normal weak inputs disappear.
