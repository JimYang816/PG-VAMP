# Independent review focus

This is a review checklist, not a result. Main prepared it while implementation runs.

1. Trace one physical frame from explicit QPSK/pilots through piecewise Fourier/chirp,
   affine transmit-time selection, absolute-time IQ downconversion, actual receive FFT,
   and pilot elimination. Verify the independent path imports no effective-matrix kernel.
2. Check that source-grid q is used in path frequency shift, receiver-grid k only in
   FFT inner product. Verify N=8192 and all 512 rows/columns remain represented.
3. Independently derive actual-window phase: for R=T_m+Delta, envelope exponent is
   q*df*((1+epsilon)*(Delta+n/Fs)-tau_m) and residual carrier phase is
   fc*(epsilon*(T_m+Delta+n/Fs)-tau0). Therefore shifted H per path/column gains
   exp(j*2*pi*(q*df+epsilon*(fc+q*df))*Delta). Both waveform and H must use the
   same receive time convention after removal of any external recording offset.
4. CP support: include -Tcp, exclude Tu; positive/negative epsilon, first/last block,
   lower/upper endpoint, and a window that was valid before shifting but is now invalid.
   A finite analytical matrix is not proof that a window is physically valid.
5. Continuous evaluator: CP belongs to its own block's Fourier series, LFM uses local
   chirp phase but OFDM carrier uses global time; inter-segment and outside-frame samples
   are zero. Test arbitrary times, not merely the exact sample lattice.
6. Noise: time-domain complex variance sigma2, half variance per real component,
   orthonormal FFT covariance and selected-bin cross covariance. Monte Carlo hard QPSK
   uses bit count plus binomial tolerance; no per-frame measured-power normalization.
7. Pilot leakage test must be nontrivial: assert nonzero/significant H_DP p before
   cancellation. H_DD/y use centered grid positions from allocation, not FFT bins.
8. Review malformed public inputs, dtype/device mismatch, unsupported CSI, invalid
   frame timing and CP; avoid acceptance of NaN or silently unrepresentable parameters.
9. Read actual audit files and receipts, compare source fingerprints after final fixes,
   preserve failures, distinguish historical WP1 results from newly executed WP2 tests.
10. Source/config/receiver labels must remain ideal_complex_iq, perfect CSI and oracle
    timing. No detector/manifest/training claim is implied by physical-chain acceptance.
