# WP7 independent review focus

Scope: full approved R1–R9 and original execution specification, not just final edits.
Read the complete source excerpts, and original referenced sections when needed.

1. Pairing: identical H/y/sigma2/sample IDs/hashes for every algorithm; target
   mutation must not change predictions. Reject incomplete/duplicate frames,
   mismatched scenario/SNR/frame count, unsafe or incompatible checkpoint.
2. Independently hand-compute counts and energy ratios. A two-frame toy with
   two blocks/frame, two symbols/block (16 bits total), one wrong bit in frame 0
   block 0 and both bits wrong in one symbol of frame 1 block 1 has BER=3/16,
   SER=2/8, BLER=2/4, FER=2/2. This is a metric algebra fixture only, never a
   substitute for physical eight-block frame acceptance. Physical goodput must
   use the real frame's 6400/0.956 coefficient and actual FER.
3. Bootstrap resamples entire independent frames and recomputes ratios, never
   bits or block means; all algorithms share the paired resample. Audit zero
   counts, F=1/very few frames, unequal denominators, deterministic seed, and
   no pollution of data/training RNG. Report paired intervals, not only point
   differences. Do not pretend same-frame SNR copies are independent.
4. Inject NaN, Inf and factorization failures at distinct block positions and
   in a batch. Persist exact affected IDs and planned denominator. A failed
   cell has unavailable full-population metrics, not a success-only BER with
   a nominal full denominator. Legal message rejection remains valid output.
5. Probe decomposition/graph calls and timer boundaries, not just nonnegative
   durations. Cold-H includes all preparation; same-H excludes only genuinely
   invariant work and separately reports first preparation plus amortization.
   Changed H, dtype/device, sigma2 or PG parameters invalidates relevant cache.
   No private cache can silently bypass finite-input validation or autograd.
   Compare prepared/unprepared outputs against independent mathematical oracles.
6. B1 online latency is not batch duration/B; actual fixed-batch throughput is
   separate. Device synchronization, threads, warmup/repeats and diagnostics
   exclusion are explicit. CPU peak RSS and CUDA allocated memory have different
   meanings; process high-water marks must not be falsely attributed per algorithm.
7. Check actual message/no-information opportunity denominators including early
   failure, two edge-proportion denominators, full-grid versus effective ICI,
   learned versus initial thresholds/mu, and finite small diagnostic copies.
8. Recompute persisted aggregate counts from per-frame rows independently.
   Inspect every applicable §18 figure visually, including zero-error log markers.
   REPORT must work with detectors/training disabled and source datasets absent;
   tampered artifacts rejected. Missing evidence explicitly says 未执行.
9. Multi-seed merge checks compatible experiment/sample hashes and disallows
   duplicate seeds/runs masquerading as replicates; retain per-seed values and
   do not multiply the same baseline into independent observations. Untrained
   PG is unmistakably labelled, and a single seed has no invented across-seed SD.
10. Run real full 512/400/T8/8-block CPU CLI loop plus target/full regression,
    lint/format/type checks. Preserve actual argv, environment, exit codes,
    artifacts and fingerprints. CUDA hardware absence is skip, not success.
    Confirm source spec/oracles unchanged. Main training/sweep remain unexecuted
    unless actually run; no performance superiority conclusion from smoke.

Prior verified runtime: .venv/Scripts/python.exe with MKL_THREADING_LAYER=TBB;
Ruff at C:/Software/Anaconda3/Scripts/ruff.exe. These are historical hints, not
current execution receipts. The current CLI names must be checked from --help.
