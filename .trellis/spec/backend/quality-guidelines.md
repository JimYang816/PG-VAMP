# Verification and truthful reporting

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§0.3–0.4, 6.6, 16–18, 19, 21–24. Full §23 tests and §22 gates remain mandatory;
the following principles do not reduce them.

## Independent correctness evidence
- Use pytest. Target command after WP0 supplies the package/tests:
  python -m pytest -q. This is not a claim that tests currently exist or pass.
- Independently readable dense_pg_vamp and dense_vamp_cholesky oracles live in
  the project. Compare PG-VAMP layerwise forward and parameter gradients;
  compare production SVD VAMP with Cholesky layerwise outputs.
- Do not wrap production code as its own oracle or change both together merely
  to claim equivalence. Corrections require a source-contract explanation and
  independent evidence. No inherited external/historical oracle test counts.
- Validate waveform samples through actual FFT against analytical H, with
  nonzero distinct path time scaling and valid CP at the default 8192 length.
  §6.6 requires relative error <1e-9 for nondegenerate complex128 tests.
- Algebra fixtures (N=8/16/32) cannot replace physical tests. Full system smoke
  retains 512 grid/400 data/T=8, all three forwards, two PG updates,
  checkpoint roundtrip and counts (§§15.5, 21.2).
- Preserve all §23 property, Jacobian, gradient, limit, protection, scale,
  device, data and CLI tests. Full/diagonal cases may legitimately have zero
  gradients; test learning gradients on nondegenerate ICI fixtures.
- Use §23's numerical benchmarks: complex128 forward atol=1e-9/rtol=1e-8,
  small-matrix PSD minimum eigenvalue >= -1e-9, gradcheck eps=1e-6,
  atol=2e-5/rtol=2e-4. Record conditioning-based tolerance changes; never loosen
  tolerances merely to pass. Statistical tests use fixed seeds and
  sample-count-justified tolerances. complex64 checks finite output/gradients
  and explicitly justified single-precision error.

## Fair evaluation and failure semantics
- Identical sample IDs/hashes, preprocessing, H/y/sigma2, device, dtype, threads
  and fixed frame counts for all algorithms. No target-dependent inference,
  test tuning, cherry-picked samples or asymmetric cost accounting (§16).
- Aggregate counts before rates/dB; use frame-cluster bootstrap and paired
  resampling for algorithm differences. Keep zero errors as 0 / measured count;
  log-plot display values never overwrite CSV truth (§17).
- Message rejection is a valid algorithm branch. NaN/Inf/factorization failure
  is a hard failure: preserve sample IDs and mark incomplete_or_failed instead
  of removing failed blocks from denominators. Never silently drop missed frames.
- Timing follows §17.3, including declared decompositions, warmup/repeats,
  CUDA synchronization, batch/device/dtype/thread metadata and distinct cold-H
  versus same-H amortized results. Do not confuse compute throughput with
  physical-link goodput.

## Evidence and completion
- Report only commands actually run and their actual outcomes, environment and
  output paths. Failed/skipped/unavailable checks must be identified as such.
  Never infer executed test counts, BER, speedups or convergence from formulas.
- Source §24 describes earlier equation checks, not this project's acceptance.
  Untested main training/SNR sweeps remain “未执行” with reproducible commands.
- Separate implementation completion from sufficient performance validation.
  smoke_math, smoke_system and development_only do not establish main results.
  A single training seed cannot support invented mean/std across seeds.
- Reports state ideal-I/Q/timing/perfect-CSI/no-coding assumptions and limitations;
  PG-VAMP may lose or show no gain. No promised real-world performance (§18).
