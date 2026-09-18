# Independent WP3 review focus

Prepared by coordinator during implementation; these are review obligations, not passed results.

1. Validate source §§5.4, 6.6, 10, 11, 16.2, 21.3, 22 WP3 and applicable §23 directly. Do not accept implementation tests as the only requirement inventory.
2. Reconstruct persisted frame/shard checksums, mapping, noise variance, counts and split identities independently. Distinguish per-scenario physical frame counts from SNR-copy/sample counts.
3. Exercise out-of-order and repeated access, cache eviction, consumer mutation and cross-process replay. A cache keyed only by frame or a global RNG-driven __getitem__ can pass sequential tests while failing the contract.
4. Replay time-domain independent quadrature noise then unitary FFT for §5.4. Compare effective and independent waveform observations with the same saved seeds; no measured-power scaling or hidden H normalization.
5. Verify artifact audit saves actual full physical waveforms/tensors sufficient for an independent NumPy FFT and pilot cancellation. Select nonzero unequal epsilon frames; do not call H@X an independent waveform oracle.
6. Validate all input routes, not just compact records: materialized inference permits absent labels, training/evaluation require both, labels present must be valid; check dimension/dtype/finite/positive sigma2/QPSK/config/mapping/count/split metadata.
7. Capacity check must precede matrix construction and output, and apply to CLI and config-selected materialized storage. Tensor payload must not be mislabeled exact serialized bytes. Use tiny outputs to test explicit override.
8. Preserve runtime CPU default and explicit CUDA; reject unsupported synchronization/backend behavior rather than silently switching modes. Document full-size complex64 tolerance separately.
9. Rerun product lint/format/mypy and full regression. Existing Trellis tooling lint findings are not WP3 failures; scope lint to product src/tests/audit scripts.
10. Actual three-detector shared-input and prediction/target-isolation integration remains WP4–WP7. WP3 consumer/hash tests validate data contracts, not completed downstream algorithms.

Verified environment preflight: existing .venv/Scripts/python.exe with MKL_THREADING_LAYER=TBB set before startup loads PyTorch 2.12.0+cpu and performs matrix multiplication. Default interpreter without this setting hit duplicate OpenMP initialization. No dependency upgrades or KMP_DUPLICATE_LIB_OK workaround used.
