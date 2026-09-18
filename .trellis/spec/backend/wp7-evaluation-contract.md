# WP7 evaluation and reporting contract

Authority: docs/CODEX_ENGINEERING_SPEC.md §§16–18, 21.6, 22 WP7, 23.3.
Implementation and independent review are complete; acceptance evidence belongs in
VALIDATION.md and the WP7 task research, not this interface guide.

## 1. Scope / trigger
Read before changing evaluation population selection, metric denominators,
frame statistics, detector preparation, timing, result bundles or reports.
WP7 consumes existing physical data and validated checkpoints; it does not
select parameters on test data or change the detector mathematics.

## 2. Signatures
```text
evaluation.runner.evaluate(config, manifest: Path, output: Path, *,
    checkpoint=None, algorithms=None, allow_untrained=False,
    bootstrap_seed=None, argv=None) -> dict
evaluation.runner.benchmark(config, manifest: Path, output: Path, *,
    checkpoint=None, algorithms=None, allow_untrained=False,
    modes=("per_observation_cold_H", "same_H_amortized"), argv=None) -> dict
evaluation.metrics.block_counts(bits_hat, x_soft, bits, x) -> dict
evaluation.metrics.frame_counts(blocks, symbols) -> dict
evaluation.metrics.aggregate(rows) -> dict
evaluation.statistics.bootstrap(rows_by_algorithm, *, seed, repeats=2000)
    -> (intervals_by_algorithm, paired_comparisons)
algorithms.prepared.PreparedDetector(model, H, sigma2)
PreparedDetector.detect(H, y, sigma2) -> DetectionResult
reporting.report(results: list[Path], output: Path | None = None) -> dict
```
CLI evaluate/benchmark: --config --manifest --checkpoint --algorithms --device
--dtype --seed --allow-untrained --output. evaluate additionally accepts
--bootstrap-seed; benchmark accepts --timing-mode with per_observation_cold_H,
same_H_amortized or both (default). report accepts --results with one or more
evaluation directories and optional --output. Test --seed must match the fixed
pre-generated manifest; different training seeds belong to separate checkpoints.

## 3. Contracts
- The manifest fixes test scenario/SNR/frame count. Complete physical frames
  have eight blocks at one SNR. All selected detectors receive the same
  H/y/sigma2 and sample IDs/hashes; targets never enter a detector.
- main_simulation enforces PG depth eight and VAMP iterations eight for selected
  algorithms. Supplementary development experiments retain their actual depth
  and explicit development label instead of posing as the main protocol.
- A frame row owns integer bit/symbol/block/frame errors, planned denominators,
  success/failure counts and error/target energies. Aggregate sums before ratios
  or logarithms. Physical goodput uses 6400/0.956 * (1 - FER).
- A hard failure makes complete-population rates unavailable and the cell
  incomplete_or_failed. Preserve planned totals and affected IDs. Any
  success-only rate must be explicitly conditional with its own denominator.
  Message rejection/holding is valid algorithm behavior, not a discarded block.
- Bootstrap resamples independent frames with replacement, recalculating total
  count ratios. All paired algorithms share resample indices. Preserve the
  bootstrap seed and repeats; zero errors remain zero in CSV. A zero bootstrap
  CI does not prove zero true BER; frame-zero FER bounds follow source §17.5.
- Prepared detection is eval/inference-only, bound to unchanged H/noise/model
  settings/parameters/dtype/device. Cache no observations or training graph.
  PG may reuse masks and majorizer terms, never all gamma2-dependent factors.
  Ordinary public forward keeps autograd and oracle equivalence.
- Cold timing includes required preparation and hard decisions. Amortized
  timing separately reports preparation, subsequent observations and the total
  amortization denominator. Different physical blocks may have different H.
  B1 online latency is separate from batch latency/B. Exclude IO/logging and
  report common preprocessing separately. CUDA timing synchronizes only when
  explicitly selected; CPU operation must not query CUDA.
- CPU process lifetime peak RSS is not per-algorithm allocation. CUDA peak
  allocated memory and matrix workspace estimates have distinct labels.
- Result bundles preserve source/config/environment/checkpoint/manifest
  provenance, per-frame/aggregate counts, timing, diagnostics, sample lineage,
  failures, paired statistics and actual examples. Reports consume these
  persisted artifacts only; they must not need source datasets or run models.
- Squared-error energy can overflow even when each prediction is finite. Check
  energy sums explicitly and record that block as a hard failure rather than
  serializing an infinite metric. Keep diagnostics streaming with bounded
  summaries so main-profile reports do not retain every layer's Python objects.
- Diagnostic c may be a finite unresolved small negative value under the WP5
  numerical protection contract. Report validation must preserve that legitimate
  no-information diagnostic, not impose a stricter detector equation afterward.
- Physical synchronization examples must correlate matched frequency domains:
  affine_waveform defaults to downconverted I/Q, whereas make_lfm.analytic is
  a passband template. Downconvert the template consistently, pair precision,
  and use configured threshold. Label the expected LFM template start including
  leading silence; it is different from the recording's frame arrival offset.
- Multi-seed reports retain per-seed results and compatibility checks. Repeated
  runs of the same training seed are not independent replicas. Reused baseline
  results must not inflate independent sample counts. Single-seed results have
  no invented cross-seed variance. PG without checkpoint needs explicit
  allow-untrained and the PG-VAMP-untrained algorithm label.

## 4. Validation and error matrix
| Condition | Required behavior |
| --- | --- |
| Missing checkpoint for PG | Error unless explicitly allow-untrained |
| Incompatible physical/PG/dtype contract | Reject before evaluation |
| Empty, duplicate, missing or mixed-SNR frame population | Reject |
| NaN/Inf/factorization failure | Persist IDs and incomplete status; no hidden drop |
| Changed prepared H/noise/model/device/dtype or training mode | Invalidate/reject |
| Missing, tampered or unsafe result artifact | Reject report input |
| Existing unrelated output/pending run | Preserve it and reject overwrite |
| Missing performance evidence | State 未执行; no invented curves/conclusions |
| Unavailable explicitly selected CUDA | Readable error; no CPU fallback |

## 5. Good / base / bad cases
Base: CPU complex128, full physical dimensions, trained checkpoint, fixed test
manifest, frame metrics, paired intervals and a report from saved results.
Good: inject failure in one block, retain the original eight-block opportunity;
copy a result bundle without data/checkpoint sources and regenerate its report.
Bad: average per-block dB, resample bits, drop a failed block, mix dtypes in a
speed claim, or reuse one time-varying H as if it represented every frame block.

## 6. Required tests
Independent hand counts and energy sums; frame-cluster/paired bootstrap with
fixed seed and unequal denominators; zero/few-frame bounds; target isolation;
fault injection preserving planned populations; preparation parity against
independent oracles and cache invalidation; decomposition and timer-boundary
probes; B1/batch/CUDA metadata; persisted count/hash verification; report with
model execution disabled; missing/tampered/multi-seed inputs; visual figure QA;
real CPU 512/400/T8/eight-block CLI loop; complete regression/lint/type checks.

## 7. Wrong vs correct
Wrong: successful blocks alone produce a normal BER while the CSV still calls
the planned population complete.
Correct: retain the planned totals and failure IDs, make full rates unavailable,
and label any success-only statistic as conditional.

Wrong: all-zero bootstrap bars prove zero real BER.
Correct: preserve 0 / measured bits, explain frame correlation and limited
evidence, and optionally report the valid frame-level zero-event FER bound.
