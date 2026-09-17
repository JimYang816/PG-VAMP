# Runtime and configuration

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§3.3, 4.2, 8.4, 10.5, 15.2, 15.5, 20–21. The source takes precedence.

- Default training and inference device is CPU even when a GPU exists.
  CUDA requires explicit selection; unavailable requested CUDA is an error,
  never an automatic CPU fallback. CPU-only operation must not invoke CUDA.
- Default correctness precision is complex128 with float64 real parameters.
  complex64/float32 is an explicit efficiency mode with separate error checks
  and experiment labels. No AMP/float16. Constants, buffers, identities and
  intermediate tensors follow the input device/dtype.
- Default CPU batch size is 1, num_workers is 0; set and record startup threads
  as min(4, os.cpu_count() or 1). main also defaults to CPU.
- configs/base.yaml follows §20; profiles override only necessary fields.
  Reject unknown keys, including nested keys, rather than silently dropping
  misspellings. Save the fully resolved configuration in each run directory.
  Unsupported coding and csi_mode=estimated must error.
- Validate derived frequency/FFT/carrier relationships, resource counts,
  disjoint sorted mappings, CP support for every path/block, positive threshold
  range and supported dtype using §§3, 6.4, 20. Do not merely check raw fields.
- Validate tensor shapes, square system dimensions, real/complex dtype pairs,
  finiteness, positive sigma2, labels when required, mapping/config hashes and
  split separation (§10.5).
- smoke_math is explicitly an algebra fixture, exempt from physical configuration
  checks; smoke_system retains the real physical dimensions. Never relax the
  latter's dimensions to make a smoke test cheaper.
- inspect-config exposes resolved device/dtype, physical derived values and
  storage estimates (§21.1). Generating data does not silently train; training
  without data does not secretly generate a different dataset.
- Checkpoint inference inherits checkpoint dtype unless explicitly converted;
  record and test conversions. Device switches are explicit (§21.5).
