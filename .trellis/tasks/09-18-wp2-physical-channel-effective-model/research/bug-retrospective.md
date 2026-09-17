# WP2 boundary and evidence review

## 1. Root cause categories
- E / implicit assumption: distinct Generator objects were assumed to imply
  independent noise streams. Equal states actually generate equal quadratures.
  The implementation now rejects shared/identical-state generators; fixed-seed
  covariance checks test the circular complex-noise contract.
- B / cross-layer contract: a FrameWaveforms dataclass can be manually replaced
  with missing or inconsistent layout spans. Trusting it merely because WP1
  normally constructs it correctly can silently omit OFDM blocks in continuous
  evaluation. Independent review added rejection at the physical boundary.
- D / evidence coverage: JSON summary numbers alone cannot be recomputed from
  the original time waveform. Independent review added safe-loadable tensor
  artifacts and SNR_rx so validation can independently reproduce the claim.

## 2. Why earlier checks were insufficient
The initial normal fixtures used different RNG seeds and valid WP1 layouts, so
they did not exercise those malformed boundaries. The waveform/H tests genuinely
compared independent paths, but the audit initially persisted only their summary.
These were coverage gaps, not reasons to change physical equations or tolerances.

## 3. Prevention mechanisms
| Mechanism | Action |
| --- | --- |
| Runtime | Reject identical RNG state and malformed frame layout before evaluation. |
| Regression | Exercise degenerate RNG pairs and replaced layout spans, plus valid cases. |
| Statistics | Check quadrature means/covariance and individual selected-bin covariance entries. |
| Evidence | Save tensor inputs/outputs with hashes; safe-load and independently recompute FFT/cancellation/SNR. |
| Documentation | Keep executable error contracts in backend/wp2-channel-contract.md and source model docs. |

## 4. Systematic expansion
For WP3, preserve independent stream derivation and physical frame/window identity
in data records; creating a new object is not sufficient separation. Carry error
status across boundaries rather than allowing malformed metadata to produce a
plausible shorter waveform. No dataset implementation is added in WP2.

## 5. Knowledge capture
The WP2 backend contract captures RNG/state, physical timing, layout and artifact
rules. Independent review passed after three new layout regressions; final full
suite is 194 passed/3 skipped. This repository has no Trellis product template tree to synchronize;
its project-local backend specification is the durable contract.
