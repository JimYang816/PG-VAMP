# Implementation status

WP0 is committed and archived. WP1 implementation and CPU validation are complete;
independent review fixed numerical/input boundary cases and reran the full suite.
The user approved the WP1 work commit and normal completion workflow. See VALIDATION.md and
the task's machine-readable execution receipts. WP2 implementation and its CPU
audit now pass independent review and final verification.

| Work package | Status |
| --- | --- |
| WP0 foundation, strict configuration, runtime, references, inspect-config | Completed, committed ecbda13, archived |
| WP1 modulation, frame and LFM | Complete; CPU verification and independent review passed |
| WP2 physical channel and effective model | Implemented; CPU tests/audit and independent review passed |
| WP3 datasets and splits | Implemented; CPU validation and independent review passed |
| WP4 production MMSE and VAMP | Implemented; CPU validation and independent review passed |
| WP5 production PG-VAMP and complete mathematical equivalence | Not started |
| WP6 training, checkpoint, inference and system smoke | Not started |
| WP7 evaluation and reports | Not started |
| WP8 complete delivery | Not started |

WP4 final evidence: **313 passed, 5 CUDA skipped**, product Ruff/format/mypy
and both baseline configuration inspections pass. Independent NumPy layer audit
maximum discrepancy is **2.78e-15**; actual 400-dimensional shared physical inputs
and complete-output label isolation pass. The user approved the work commit and normal archive/journal completion. No baseline performance or full-system smoke claim is made.

The WP0 references do not constitute production algorithm or physical-chain
acceptance. Resolving profiles does not constitute execution of their workloads.

WP2 final independent evidence: **194 passed, 3 skipped** in the complete suite;
product-scope Ruff/format and mypy pass. Two physical frames and 20 full-size
windows have maximum waveform-versus-H relative error **8.15e-12** in complex128.
The noise audit checks 33,554,432 time samples, 4096 FFT windows and a 204800-bit
identity-AWGN experiment. The three CUDA skips are explicit. Root-wide Ruff
includes pre-existing errors in tooling/archive files; those remain outside this
product change. Production PG-VAMP, training and system smoke remain
unimplemented. WP3 supplies dataset generation and replay; independent review passed
with 248 passed / 3 CUDA skipped, product Ruff/format/mypy and fresh artifact verification. See docs/WP2_PHYSICAL_MODEL.md for the physical interfaces.
The checker fixed malformed frame-span validation and added three regressions.
The final audit saves hashed tensor artifacts and receive SNR; safe CPU loading
and independent NumPy Fourier/FFT/cancellation reconstruction passed all 20 windows.

WP1 final evidence: **91 passed, 1 skipped** targeted and **166 passed, 2 skipped**
full regression; Ruff/format/mypy and both inspection CLI entries pass. CUDA
hardware is unavailable. CPU complex128 and explicit complex64 audits each
recover 6400 uncoded bits from the 91776-sample no-channel frame with zero bit
errors and zero template-start timing error. PSD, out-of-band power and matched
correlation plots retain their raw arrays and provenance. This is transmit-chain
acceptance, not physical-channel validation, detector BER or full-system smoke.

WP0 evidence: both installed CLI entries, all required inspection profiles,
source §20 defaults, complete resolved configuration/provenance, independent
reference algebra/gradient/protection tests, Ruff and mypy. Final targeted and
full suites each report **75 passed, 1 skipped**. The skipped test requires CUDA
hardware unavailable in this CPU build; no CUDA numerical acceptance is claimed.

PG has exactly `2T` real parameters (16 at the default depth 8). Tests include
full-graph/identity/diagonal/zero/rank-deficient limits, independent QPSK
enumeration, N=8/16/32 exact linear solves, gradcheck, scale/dtype checks and
rejected/capped/no-information backward regressions. No tolerance was relaxed.
