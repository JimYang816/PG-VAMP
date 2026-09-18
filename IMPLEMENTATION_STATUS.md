# Implementation status

The implementation follows [the engineering specification](docs/CODEX_ENGINEERING_SPEC.md).
WP0–WP7 are implemented, independently reviewed and archived. WP8 delivery is
implemented and independently verified on 2026-09-18. Completion bookkeeping
is recorded in the Trellis task archive and developer journal.

| Work package | Functionality | Verification |
| --- | --- | --- |
| WP0 configuration/runtime/references | Complete, archived | Independent CPU acceptance; historical record in VALIDATION.md |
| WP1 QPSK/allocation/frame/LFM | Complete, archived | Full-size waveform and synchronization audits |
| WP2 physical channel/receive model | Complete, archived | Independent waveform/H, CP, noise and pilot-cancellation checks |
| WP3 compact data/replay/splits | Complete, archived | Replay, hashes, isolation, storage budgets and independent audit |
| WP4 linear MMSE/exact VAMP | Complete, archived | Cholesky/SVD oracle and full-dimensional forward checks |
| WP5 PG-VAMP-VC | Complete, archived | Mathematical properties, layerwise oracle, gradients and 2T parameter contract |
| WP6 train/resume/checkpoint/infer/smoke | Complete, archived | Fresh WP8 11-command CPU acceptance and independent artifact audit passed |
| WP7 paired evaluation/timing/reports | Complete, archived | Fresh WP8 17-command CPU acceptance and independent saved-result/visual audits passed |
| WP8 delivery documentation/demo-frame/final CPU validation | Implemented and independently verified | 486 passed / 7 CUDA skipped; Ruff/format/mypy; fresh full-size smoke, 28 training/evaluation/report commands and both demo precisions passed |

See [VALIDATION.md](VALIDATION.md) for dated actual execution results. The
[delivery matrix](docs/DELIVERY_MATRIX.md)
maps work packages and every source §23 category to concrete test coverage.
Historical results are not new execution results; current final acceptance
is recorded in the dated WP8 section, separate from previous work packages.

## Evidence boundaries

The fresh WP8 system smoke retains 512 grid points, 400 unknown QPSK symbols,
8192 FFT samples, CP2048, eight OFDM blocks, T8 and two training updates.
Independent saved-array checks verify the common detector inputs, integer counts,
16 trainable real scalars, checkpoint roundtrip and waveform/H relative error
1.1310709079623577e-11. This establishes the bounded software path, not convergence
or a BER advantage. Source specification and independent references are unchanged.

Main/full training, statistically powered full SNR sweeps, adequate multi-seed
performance comparisons, CUDA numerical/latency experiments, controlled
ablations, unknown-channel estimation and real-world validation are **未执行**.
CUDA tests skipped on this CPU-only installation do not count as numerical
acceptance. Main comparison assumes ideal timing/IQ, perfect CSI, known noise,
valid CP and uncoded QPSK. LFM synchronization demonstration is separate.

No PG-VAMP speedup or superior BER is claimed. Its current soft-gated dense
implementation still performs dense factorizations. Configuration inspection
is not workload execution, and a two-update checkpoint is not a converged model.
