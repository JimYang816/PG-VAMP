# Independent oracle arithmetic repair review — 2026-09-18

## Finding and evidence
The independent checker reproduced nonfinite raw_gaps/raw_mu gradients in the
unchanged DensePGVAMP on H=1e-140*[[1,.2j],[.3,1]], y=ones, sigma2=1, depth=2,
with the real-sum posterior loss. Production originally had the same defect.
Normal N=8/16/32 forward/gradient parity did not reveal this shared boundary bug.
The reference is therefore not suitable as the sole expectation on this case.

## Source contract
Source §14.5 defines W/c and innovation/c; §14.8 requires real multilayer
dependencies to differentiate without arbitrary detach. Re-evaluating each
quotient as two real-component divisions by sqrt(c) for c>tol preserves these
identities and the original no-information decision. No new physical scaling,
noise change, trace approximation, tolerance relaxation or learning parameter.
Autograd's direct quotient can overflow local x/c² even when the weighted total
parameter derivative is representable. Componentwise division alone was insufficient.

## Coordinating review decision
The parent reviewed the source equations, production patch, failed simpler repair,
mixed-batch/asymptotic regression and independent checker reproduction. It approved
only the corresponding W/c and innovation/c arithmetic change inside the reference,
implemented locally with no imports from production numerical helpers.
This follows design.md's explicit exception for independently justified oracle
defects; it is not permission to change the oracle to match unexplained output.

## Required verification
The checker owns separate reference float64/float32 weak-channel mixed-batch
regressions against the leading-order H^H y/sigma2 expectation, finite backward
and ordinary-sample gradient agreement; then normal-scale layer/gradient parity,
full tests, lint/types and a refreshed source manifest. Final results belong in
check-report.md and command receipts, not inferred from this approval record.
