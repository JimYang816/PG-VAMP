# Validation record

## WP7 independent final validation — 2026-09-18

Final full regression: **478 passed, 7 CUDA skipped in 95.52 s**. Product Ruff
lint/format and mypy pass (66 source files). Fresh full-dimensional acceptance
completed **17/17 CLI commands**, including two training seeds, paired evaluation,
both timing modes, unlabeled inference, single-run and multi-seed reports.

Independent review fixed mismatched baseband/RF synchronization diagnostics,
actual measured-noise draw counts, finite-output squared-energy overflow,
main_simulation fixed-eight-round enforcement, and report handling of legitimate
unresolved small negative c. Diagnostic reporting is streamed into bounded
summaries. Failures at blocks 0/3/7 retain planned denominators and exact IDs.
Prepared inference matches separate PG and Cholesky VAMP oracles; ordinary
training gradients and previous WP regressions remain covered.

Artifacts: `runs/wp7-cli-review-v2/`, with 512-grid/400-data/8192-FFT/2048-CP,
eight-block frames and T8. Both result bundles passed independent hash and
per-frame-to-aggregate recount. Coordinator visually inspected all ten final
single-run figures and five representative merged-report figures. The source
specification and independent reference files retain their pre-WP7 hashes.

Evidence in `.trellis/tasks/09-18-wp7-paired-evaluation-reports/research/`:
`check-report.md`, `check-receipts.json`, `check-fingerprints.json`,
`check-artifact-audit.json`, `coordinator-final-count-audit.json`, and
`coordinator-visual-review.json`. Exact CLI receipts live with the run and are
included in the task evidence. Historical implementation-stage evidence follows.

This is software acceptance with one physical frame per SNR cell and two-update
training checkpoints. Full main training, statistically powered SNR sweeps,
CUDA numerical execution, ablations and performance superiority are **未执行**.
The three-repeat timings and one-frame intervals do not establish an advantage.

## WP7 implementation validation — 2026-09-18

Core targeted evaluation/statistics/timing tests: **18 passed**. Prepared-state
equivalence plus existing baseline/PG oracle/gradient regression: **77 passed,
1 CUDA skipped**. The preliminary full suite was **459 passed, 7 skipped,
1 reporting semantic-validation failure** while reporting was still being developed;
the independent review owns the final frozen-source quality gate.

All 15 non-report subprocess commands in `scripts/wp7_acceptance.py` completed
successfully under `runs/wp7-cli-acceptance-v1`. They include physical generation,
independent waveform audit, two distinct training seeds with two updates each,
paired three-detector evaluation at two SNRs, both benchmark protocols, unlabeled
materialization and checkpoint inference. Dimensions remain 512/400/8192/CP2048,
eight blocks/frame and depth/iterations8. Exact receipts and limitations are in
the active WP7 task's `research/implementation-evidence.md` and
`research/implementation-cli-receipts.json`.

This is a one-frame-per-cell functionality exercise, not adequate BER/performance
evidence. Main training, full sweep and CUDA numerical execution remain 未执行.
Final report/check results are recorded by the independent review.

## WP6 independent final validation — 2026-09-18

Final full regression after review fixes: **430 passed, 7 CUDA skipped** in
59.05s. Ruff check/format (86 files) and mypy (55 source files) pass. Independent
CLI acceptance reran all 11 commands successfully, including both mathematical
precisions, full physical smoke and train/resume/unlabeled-infer integration.

Review reproduced and repaired historical WP3 schema-1 data loading: only the
exact pair of missing new early-stop fields is accepted for data reads, without
changing the persisted configuration or its hash. Checkpoints remain strict.
Additional repairs validate all supported Adam behavior flags, add contextual
inference failures, and enforce full 8192/2048/eight-block physical smoke dimensions.
Regression coverage includes interrupted last/scheduled-best writes, legacy
manifest/materialized inputs and malformed optimizer states.

Independent artifact verification recomputes bit/symbol/block/frame counts from
saved predictions, confirms identical three-detector H/y/sigma2 hashes, checks
checkpoint hashes and performs NumPy FFT/cancellation reconstruction. Maximum
relative waveform error is **1.1310709079623577e-11**. Count denominators are
6400 bits / 3200 symbols / eight blocks / one complete frame per detector.
Both mathematical smokes perform two updates at N32/T2; system smoke performs
two at 512-grid/400-data/T8 and includes checkpoint loading and unlabeled inference.

Evidence: WP6 task `research/check-report.md`, `check-pytest-final.txt`,
`check-lint.txt`, `check-format.txt`, `check-mypy.txt`, `check-artifacts.json`
and its independently written artifact verifier. The implementation-stage
receipts below are retained as history. CUDA hardware is unavailable; no GPU
numerical acceptance, full main training, BER sweep, convergence or speedup is claimed.

## WP6 implementation validation — 2026-09-18

Implementation-stage full regression: **414 passed, 7 CUDA skipped** in 52.71s.
Ruff check/format (85 files) and mypy (55 source files) pass. This is the
implementation receipt; independent review is in progress and may add fixes.

The final implementation CLI exercise ran 11 commands successfully: two config
inspections, complex128/complex64 mathematical smokes, complete physical smoke,
simulate, audit-data, two-update training, resume to four updates, unlabeled
materialize and infer. Artifacts are in `runs/wp6-acceptance/`; durable command
receipts/hashes are under the WP6 task's `research/implementation-*.{md,json}`.
CPU environment remains Python 3.13.9 / PyTorch 2.12.0+cpu / NumPy 2.3.5,
four threads and MKL_THREADING_LAYER=TBB.

System smoke retains grid512/data400/FFT8192/CP2048/eight blocks/T8/two updates.
Nonzero distinct time-scaling paths have waveform/effective-model relative
error **1.1310698530350202e-11**. All three algorithms consume identical input
hashes and count one full frame: 6400 bits, 3200 symbols and eight blocks.
Checkpoint roundtrip and label-free inference pass. These tiny acceptance runs
do not establish convergence, BER advantage or computational speedup.

Targeted checks cover complete model/Adam/sampler/RNG resume equivalence,
off-cadence validation, early stop, best-write failure recovery, strict safe
checkpoint schema, no-label/changed-label inference and CPU forbidden-CUDA paths.
Historical WP3 configuration compatibility is explicitly under independent review.
No main/full training, SNR sweep, CUDA numerical run or WP7 report was executed.

## WP5 independent final validation — 2026-09-18

Final full regression after review repairs: **376 passed, 6 CUDA skipped** in
36.10s. Ruff lint/format (70 files), mypy (48 source files), cpu_dev inspection,
task context validation and Git whitespace checks pass. No known unresolved
WP5 code finding remains. Source §23 tolerances and specification SHA are unchanged.

Independent review found and fixed finite-forward/NaN-backward behavior for an
extremely weak non-diagonal channel. Production W/c and innovation/c now use
two real-component sqrt(c) divisions, preserving the source equations and trace
threshold. float64 1e-140 and float32 1e-17 weak/zero/normal mixed batches retain
informative outputs and finite gradients without contaminating normal samples.
Safety diagnostic norms now share a cancelling scale, preventing overflow or
underflow under joint H/y scaling by 1e±100 (noise by 1e±200).

The checker separately reproduced the same pre-existing defect in DensePGVAMP.
After a source-backed coordinating review, only its two quotient evaluation
sites were repaired independently, without production numerical imports.
Separate weak-channel asymptotic/mixed-batch tests and ordinary layer/gradient
parity pass; this was not an unexplained change to make two implementations agree.
See `research/oracle-fix-review.md` and `check-oracle-correction.md` for the audit.
MMSE/VAMP and shared production QPSK/message implementations are unchanged.

Independent NumPy audit reconstructs ell via scalar sums, G/P, solve-based B/W,
innovation and full covariance at N=8/16/32, jitter=0/0.125. Maximum absolute
discrepancy is **1.0303e-13**, with positive matrix-order checks and fixed-layer
objective checks. A freshly loaded physical 400-dimensional sample supplies the
same input hash and identity to all three detectors; original/modified/absent
labels produce identical complete prediction hashes within each detector.

Final evidence under `.trellis/tasks/archive/2026-09/09-18-wp5-pg-vamp-math-contract/research/`:
`check-report.md`, `check-oracle-final-pytest.txt`, `check-quality-receipts.json`,
`check-numpy-receipt.json`, `check-physical-receipt.json`, `check-source-manifest.json`.
The coordinating session independently verified all 72 final source/test hashes.
Earlier 370/374-test receipts are historical, before all review fixes.
Environment remains Python 3.13.9, PyTorch 2.12.0+cpu, NumPy 2.3.5, four threads,
MKL_THREADING_LAYER=TBB. CUDA numerical acceptance is unavailable, not passed.
No training, optimizer update, checkpoint/resume, full-system smoke, BER sweep
or performance benchmark was run. The physical check is a forward audit only.

## WP5 implementation validation — 2026-09-18

Implementation-stage full regression: **370 passed, 6 CUDA skipped** in 36.96s.
Ruff check/format (70 files), mypy (48 source files), cpu_dev configuration
inspection and Git whitespace checks pass. Independent final review is pending;
these results are not its acceptance record.

The production PG-VAMP module has exactly 16 learned real scalars at T=8.
New tests cover directed nested gates, energy/safety/Loewner properties,
operator order and fixed-layer objective, conditional real Jacobian, complete
covariance trace, N=8/16/32 layerwise independent-oracle parity and both raw
parameter gradients, gradcheck, full-graph/diagonal/weak/rank-deficient limits,
explicit jitter consistency, protected backward and same-layer factor reuse.
Existing baseline and QPSK/message tests remain in the full regression.

A real 400-dimensional WP3 sample with six unequal nonzero path time scalings
was passed to MMSE/VAMP/PG-VAMP using identical input hashes and sample identity.
All three produce finite outputs and unchanged complete prediction hashes when
labels are changed or removed. The sample's PG protection counters are zero.
This is a physical forward audit, not training, checkpoint or full-system smoke.

Evidence: `.trellis/tasks/archive/2026-09/09-18-wp5-pg-vamp-math-contract/research/`
`implementation-evidence.md`, `audit_physical.py`, `physical-forward-receipt.json`.
The report preserves initial Windows encoding/temp-directory and formatting
failures and their fixes. Source specification, independent reference numerical
code and production QPSK/messages are unchanged. No numerical tolerance was relaxed.
Environment: Python 3.13.9, PyTorch 2.12.0+cpu, four CPU threads,
MKL_THREADING_LAYER=TBB. No GPU numerical acceptance is claimed.
Training, checkpoint/resume, full-system smoke, BER sweeps and timing remain unexecuted.

## WP4 independent final validation — 2026-09-18

Final full regression: **313 passed, 5 CUDA skipped** in 31.64 seconds.
Product Ruff/format, mypy (43 source files), and cpu_dev/vamp_reference_32
configuration inspections pass. No remaining WP4 review blocker. The independent
Cholesky reference and authoritative engineering specification are unchanged.

The reviewer repaired a test-coverage gap: each detector now reloads its original
physical sample before label mutation. Independent original/flipped/absent-label
checks compare all result tensors, including VAMP probabilities, on the same
400-dimensional physical input. Input hashes remain unchanged and equal.
Independent NumPy solves and direct four-point enumeration audit every VAMP
layer on three seeded 7x7 complex systems: maximum discrepancy **2.78e-15**;
MMSE maximum error is **3.52e-16**. Source complex128 tolerances were not relaxed.

Final receipts under `.trellis/tasks/archive/2026-09/09-18-wp4-mmse-exact-vamp/research/`:
`check-report.md`, `check-validation.json`, `check-numerics.json`,
`check-physical.json`, and `check-source-hashes.json`. The final source hashes
include the reviewer's test fix; implementation-stage receipts below are historical.
Environment: existing Python 3.13.9 virtualenv, PyTorch 2.12.0+cpu, four threads,
MKL_THREADING_LAYER=TBB. CUDA is unavailable; no GPU acceptance is claimed.
Production PG-VAMP, training, evaluation/performance sweeps and full-system smoke
remain unexecuted later-WP work.

## WP4 implementation validation — 2026-09-18

Full regression: **313 passed, 5 skipped** in 32.55 seconds using the existing
`.venv` (Python 3.13.9, PyTorch 2.12.0+cpu, four CPU threads,
`MKL_THREADING_LAYER=TBB`). All skips require CUDA. Product Ruff/format and
mypy (43 source files) pass. Both `cpu_dev` and `vamp_reference_32` configuration
inspection commands pass. This is the implementation-stage record; the completed
independent review is recorded above.

New checks cover raw linear MMSE, SVD/Cholesky VAMP equivalence at every layer
for N=8/16/32 and 8/32 iterations, QPSK enumeration, true divergence, mixed
zero/rank-deficient/weak channels, scale invariance, label-free inputs, invalid
inputs and hard failures, and rejected/capped-message backward. Finite candidate
overflow required a guard before division: filtering the infinite quotient
afterwards left NaN gradients. Four float32/64 regressions now cover numerator
and quotient overflow without modifying the independent reference.

A fixed-seed WP3 sample with six nonzero, unequal path time scalings supplies
the same 400×400 H/y/sigma2 to both real baselines. Inputs remain unchanged and
changing/removing labels leaves predictions unchanged. Both outputs are finite,
both parameter counts are zero, and this VAMP sample has zero protection events.
No BER, speedup, training, three-algorithm or full-system smoke claim is made.

Task evidence: `.trellis/tasks/archive/2026-09/09-18-wp4-mmse-exact-vamp/research/` contains
`implementation-validation.json`, `implementation-lint.json`,
`implementation-physical.json`, `implementation-source-hashes.json`,
`implementation-evidence.md`, and `svd-no-information.md`.
The initial system-Python run had four package-import/CLI failures because that
interpreter lacks the editable install; its failed receipt is retained.
The virtualenv's inherited `python -m ruff` launcher could not find its executable;
the successful lint receipt uses the installed Anaconda Ruff executable.

## WP2 independent final validation — 2026-09-18

The checker reviewed the full physical scope and fixed malformed frame interval
acceptance (missing/overlapping spans), adding three regression cases. The audit
now records source §9 receive SNR and hashed tensor artifacts for every window.
No physical equation or numerical tolerance was changed.

Final targeted tests: **28 passed, 1 skipped**. Full regression: **194 passed,
3 skipped**. Product Ruff/format and mypy (29 source files) pass. CUDA skips
remain hardware limitations. Commands, stdout/stderr and exit codes are in
`research/check-final-validation.json` under the WP2 task; earlier checker and
implementation runs remain preserved.

Fresh audit: `runs/wp2-check-audit-final-01/`; tracked JSON copy:
`research/check-physical-audit.json`. Two frames / 20 windows retain the maximum
complex128 error **8.152338913635682e-12**. The checker safely loaded both tensor
artifacts on CPU with `weights_only=True`, verified artifact/config/source hashes,
and independently recomputed all 20 FFTs, data/pilot row selection, clean/noisy
cancellation, receive SNR and complex64 conversion checks. NumPy direct Fourier
evaluation also agrees at four samples in each actual window. Detailed numerical
cross-checks and final source fingerprints are in `research/check-artifact-verification.json`.
The independent review found no remaining WP2 blocker. Later work packages and
the full-system smoke remain unexecuted.

## WP2 implementation validation — 2026-09-18

The implementer added the five physical path scenarios, per-block CP validation,
independent continuous waveform/affine propagation, full analytical grid H,
actual ideal-I/Q FFT, true pilot cancellation and time AWGN. The following is
the preserved implementation-stage record; final independent evidence is above.

Complete regression: **191 passed, 3 skipped** (CUDA unavailable). Product-scope
Ruff, format and mypy pass. Exact commands, stdout/stderr, exit codes and source
fingerprints are saved in the WP2 task's `research/implementation-validation.json`.
The implementation physical audit is retained in `runs/wp2-audit-final-01/`, with a tracked
copy at `research/physical-audit.json`. Earlier `runs/wp2-audit/` is preserved.

Two independently seeded physical frames cover all eight blocks, plus shifted
first/last-block windows: 20 windows, 8192 time samples, 512×512 full H and 400
data rows. Distinct nonzero path time scales yield maximum complex128 relative
error **8.152338913635682e-12**, below 1e-9. Explicit complex64 conversion is
checked separately against 2e-6. Pilot leakage is measured before cancellation;
both noiseless and noisy residuals are checked against the physical waveform.

At Es/N0=4 dB, sigma2=0.3981071705534972. The 33,554,432-sample noise audit gives
real/imaginary variances 0.19905049/0.19903189 and FFT complex variance 0.39795522.
Means, component covariance and every entry of an eight-bin covariance matrix
pass predeclared sample-count tolerances. Identity IFFT/time-AWGN/FFT hard QPSK
gives **11604/204800 bit errors** (0.05666016), versus theory 0.05649530; the count
is within the predeclared six-standard-deviation tolerance (626.89 errors).

The first full pytest command hit 31 setup errors because the system pytest temp
directory was inaccessible. A fresh workspace `--basetemp` resolves this without
changing tests. The virtualenv Ruff launcher is absent, so the installed
`C:/Software/Anaconda3/Scripts/ruff.exe` was used. Literal `ruff check .` reports
192 existing hook/Trellis/archive issues; the final lint scope covers all product
source/tests and WP1/WP2 audit scripts. No unrelated tooling was modified.

This is ideal-I/Q/perfect-CSI/oracle-timing/uncoded physical validation. Full
system smoke, datasets, production detectors, training and performance sweeps
remain **未执行** and are separate later-package gates.

## WP1 independent final validation — 2026-09-18

The independent Trellis checker reviewed the full WP1 scope and repaired two
input/numerical boundary defects: positive correlation squares/products could
underflow to finite zero, and malformed Allocation fields could bypass readable
validation. Nine new regression cases verify explicit rejection, including a
small representable case that preserves the original additive epsilon formula.
No source equation, correlation threshold or forward tolerance was changed.

Exact final commands and source fingerprints are in
`.trellis/tasks/archive/2026-09/09-18-wp1-modulation-frame-lfm/research/check-validation.json`
(retained at this location by the approved completion workflow).

| Final check | Result |
| --- | --- |
| WP1 targeted pytest | **91 passed, 1 skipped**, exit 0 |
| Full pytest, including unchanged WP0 tests | **166 passed, 2 skipped**, exit 0 |
| Fresh complex128 and complex64 CPU transmit audits | Both passed, exit 0 |
| Ruff / format check / mypy | Passed, exit 0 |
| Both inspect-config entry points / git diff --check | Passed, exit 0 |

The two skips are unavailable CUDA hardware, not failed CPU tests. Final audit
directories are `runs/wp1-check-audit/` and `runs/wp1-check-audit-complex64/`.
Both reproduce the zero-bit-error and zero-timing-error observations below, with
maximum recovery errors 9.524068763971754e-12 and 2.6656007889869215e-7.
The primary session opened both final complex128 figures, validated all eight
hashed files in each audit directory, and confirmed the 24 WP0 product/test/config
fingerprints remain unchanged. See `research/main-artifact-review.json` in the
WP1 task for that additional evidence. No product changes followed these checks.

The checker's separate `research/check-artifacts.json` records exact replay from
the saved seeds for both precisions, independent NumPy FFT recovery from persisted
real samples, direct-inner-product correlation comparisons, PSD integration and
source/artifact fingerprints. Its status is passed. Double NumPy recovery maximum
error is 9.524226793575906e-12; the tiny difference from PyTorch is FFT rounding.

Implementation and check receipts remain separate so the original 157-test
implementation result is not confused with the 166-test final regression result.
The user confirmed the work commit and completion workflow on 2026-09-18; WP2
was not created or started. Commit/archive metadata is retained with the task.

## WP1 implementation-stage validation — 2026-09-18

WP1 implements the reviewed §22 modulation/frame/LFM scope. Independent final
review is recorded above; these are actual implementation
stage executions, not an assumed acceptance based on WP0 results.

Machine-readable commands, stdout, stderr, return codes and source fingerprints:
`.trellis/tasks/archive/2026-09/09-18-wp1-modulation-frame-lfm/research/implementation-validation.json`.
The accompanying `research/validate_implementation.py` records the command sequence.
All commands use `MKL_THREADING_LAYER=TBB` before Python startup and the existing
workspace `.venv`; no dependencies were upgraded. Branch: `codex/wp1-modulation-frame-lfm`.

| Executed check | Actual result |
| --- | --- |
| WP1 pytest: test_qpsk, test_allocation, test_waveform, test_lfm | 82 passed, 1 skipped; exit 0 |
| Full pytest regression, including WP0 | 157 passed, 2 skipped; exit 0 |
| CPU complex128 full-size transmit audit | Passed; exit 0 |
| CPU explicit complex64 full-size transmit audit | Passed; exit 0 |
| Ruff check / format --check | Passed; 30 files formatted; exit 0 |
| Mypy src plus audit script | Passed; 21 source files; exit 0 |
| Module and console inspect-config | Both passed; exit 0 |
| git diff --check | Passed; exit 0 |

Both skipped tests require unavailable CUDA hardware. CPU behavior and explicit
unavailable-CUDA handling were exercised; there is no GPU numerical acceptance.
Earlier lint findings were formatting/line-length issues, fixed without rule suppression.

### WP1 audit observations

Double-precision output: `runs/wp1-audit-final/`.
Single-precision output: `runs/wp1-audit-complex64/`.
Each directory contains resolved_config.yaml, environment.json, waveforms.pt,
psd.npz, correlation.npz, summary.json, artifact_hashes.json and two figures.
Generated run directories are Git-ignored; tracked receipts retain the command output.

| Measurement | complex128 | complex64 |
| --- | --- | --- |
| Transmit frame length / duration | 91776 samples / 0.956 s | Same |
| Data bits recovered without channel/noise | 6400, zero incorrect bits | Same |
| Maximum grid recovery absolute error | 9.524068763971754e-12 | 2.6656007889869215e-7 |
| Known LFM template-start timing error | 0 samples | 0 samples |

The double run injected 964 recording-padding samples: recording length 92740,
template start 2884, transmit frame length still 91776. OFDM useful analytic power
and LFM analytic power agree with 464/8192 = 0.056640625. Twelve fixed-seed real
and twelve complex noise recordings produced zero detections at threshold 0.1;
this finite fixture result is not a universal zero-false-alarm claim.

The double-run two-sided periodogram used >=4x zero padding, original segment
length in the power denominator, and both ±[21,27] kHz bands. Its largest power
integration error was 2.776e-17. Estimated out-of-band energy ratios were
0.0006655853 (whole frame), 0.0007244081 (first useful OFDM block), and
0.0025107553 (LFM). These are one recorded realization and a declared spectral
estimator, not a strict-bandlimit or channel-performance claim.

The primary session opened the initial PSD and correlation PNGs and inspected
their axes, ranges, labels and curves. Final plot titles distinguish the absence
of an OFDM transmit window from the LFM's intentional Tukey envelope.

Double forward tolerances remain atol=1e-9, rtol=1e-8; single precision uses
atol=2e-5, rtol=2e-4. Labels/indices/CP copies/lengths use exact assertions.
Time and carrier/chirp phase use float64 then explicit output conversion.

WP2 physical channel, CP path support, effective H, pilot cancellation, production
detectors, training, detector BER sweeps and full-system smoke are **not executed**.
Zero incorrect bits here means no-channel transmit recovery, not measured detector BER.

## WP0 historical validation

WP0 CPU validation and independent Trellis review passed. The following is its
historical execution record. WP0 was subsequently committed as ecbda13 and archived
under `.trellis/tasks/archive/2026-09/09-17-wp0-foundation-references/`.
Older pending-review statements below describe their original recording time.

## Source and starting state

- Original specification fully read in consecutive ranges 1–350, 351–700,
  701–1050, 1051–1400, 1401–1750 and 1751–2102.
- SHA-256: `A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
- Starting Git HEAD: `2611522` (`first commit`); working tree initially clean.
  Historical planning statements saying no commit existed are superseded by
  this observed execution state.
- Trellis context validation exited 0. Its warning about the 32,768-byte
  injection limit was handled by direct complete source reads.

## Environment preflight

Python: `C:\Software\Anaconda3\python.exe`, 3.13.9.
PyTorch 2.12.0+cpu; NumPy 2.3.5; PyYAML 6.0.3; Matplotlib 3.10.6;
pytest 8.4.2; Ruff 0.12.0; mypy 1.17.1.

The first CPU Cholesky probe **failed with exit 1**, reporting Intel OpenMP
Error #15 (duplicate `libiomp5md.dll`). No product test passed at that point.
With `MKL_THREADING_LAYER=TBB`, a fresh Python process successfully executed
complex128 Cholesky, batched solves, backward with finite gradients and NumPy
matrix multiplication (exit 0). `threadpoolctl` confirmed MKL's TBB backend.
The conflict detection bypass `KMP_DUPLICATE_LIB_OK` was not used.

All subsequent validation commands use `MKL_THREADING_LAYER=TBB` before Python
startup and record four Torch CPU threads. The environment was not upgraded.

Created a workspace-local environment with
`python -m venv --system-site-packages .venv` (exit 0), preserving global packages.
Actual editable installation command:
`.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e '.[test]'`
exited 0 and installed `pgvamp-ofdm 0.1.0`. Its additional flags prevent dependency
downloads and reuse the already available build backend. Subsequent product
validation uses `.venv\Scripts\python.exe` and the console executable beside it.

## Intermediate execution evidence

- Six planned CLI invocations succeeded (exit 0): module and console `cpu_dev`,
  module `main`, `smoke_math`, `smoke_system`, and `cpu_dev --output`.
  Original outputs are in the task's `research/cli-validation.json`.
- Independently compared every §20 default against parsed `configs/base.yaml`;
  all fields matched. See `research/base-source-check.json`.
- Both references executed two-layer identity and zero-channel cases; identity
  posterior maximum absolute error was 0 in this fixture, and zero-channel
  probabilities were exactly uniform. See `research/reference-preflight.json`.
- First implementation pytest: **1 failed, 29 passed, 1 skipped, 31 errors**.
  Errors came from shared temporary-directory permissions; the failed test
  replaced the entire CUDA namespace and intercepted an import-time type
  reference. The test now guards actual CUDA APIs, and later runs use a
  workspace-local `--basetemp`.
- After those fixes: **61 passed, 1 skipped**. Additional exact linear solve
  tests at N=8/16/32 and forbidden-API audit brought targeted/full implementation
  runs to **65 passed, 1 skipped**. These are intermediate results, not the
  independent final acceptance.
- Independent review then reproduced a genuine numerical defect: positive
  subnormal posterior variance could yield NaN backward gradients even when
  the candidate was rejected. A full PG fixture also exposed NaN parameter
  gradients in a precision-cap case. Both independent implementations now
  branch before invalid/capped reciprocals, preserving the original candidate
  mean and precision rules. Float32/float64 boundary and full-loop backward
  regressions passed.
- Review also found that an all-zero/all-no-information PG batch lacked an
  autograd graph. An explicit zero parameter dependency now permits backward
  with zero gradients, without changing outputs. Both regression cases passed.

## Final acceptance commands and results

Set `$env:MKL_THREADING_LAYER='TBB'` before these PowerShell commands.
The final complete command arrays, stdout, stderr and exit codes are preserved
in `.trellis/tasks/09-17-wp0-foundation-references/research/final-validation.json`.

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_devices.py tests/test_cli.py tests/test_reference.py --basetemp=runs/wp0-acceptance-targeted -ra` | 75 passed, 1 skipped; exit 0 |
| `.\.venv\Scripts\python.exe -m pytest -q --basetemp=runs/wp0-acceptance-full -ra` | 75 passed, 1 skipped; exit 0 |
| `python -m ruff check src tests` | Passed; exit 0 |
| `python -m ruff format --check src tests` | All 16 files formatted; exit 0 |
| `.\.venv\Scripts\python.exe -m mypy src` | No issues in 11 source files; exit 0 |
| `.\.venv\Scripts\python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml --output runs/wp0-inspect` | Passed; resolved config and final provenance refreshed; exit 0 |
| `git diff --check` | Passed; exit 0 (Git emits CRLF-normalization warnings) |

Ruff uses the installed `C:\Software\Anaconda3\python.exe` because the inherited
module's launcher is absent from the local venv. A prior venv Ruff invocation
failed for that launcher issue; no lint rule was disabled to resolve it. Mypy
checks the local package without a global missing-import bypass; only the
untyped PyYAML dependency has a targeted exception.

CUDA numerical equivalence is the single skip: `CUDA hardware unavailable`.
CPU-default and unavailable-CUDA error tests did run; mocked error branches
are not evidence of CUDA computation.

The independent check agent separately ran targeted and full suites after its
repairs: **75 passed, 1 skipped** in each (12.08 s / 13.60 s). No remaining WP0
code blockers were reported. The final main-session receipt is the reproducible
execution record, rather than a claim based only on the check agent's summary.

## Numerical contracts and evidence boundaries

- complex128 forward tolerance: `atol=1e-9, rtol=1e-8`; PSD floor `-1e-9`.
- Raw-parameter gradcheck: `eps=1e-6, atol=2e-5, rtol=2e-4`.
- Explicit complex64 check: finite outputs/gradients and `atol=2e-5, rtol=2e-4`
  against double on small well-conditioned fixtures; no single-precision result
  is presented as double-precision acceptance.
- No-information threshold uses the contraction error scale
  `gamma_(4N) * sum_ij(abs(W_ij)*abs(H_ji))/N`, where
  `gamma_(4N)=4N*eps/(1-4N*eps)`, with dtype `tiny` as the normal-precision floor.
  There is no arbitrary absolute floor tied to 1. Tests retain information for
  amplitude `1e-20` and explicitly handle `1e-160`/zero at float64's boundary.
- Full-graph forcing exists only in a test subclass; there is no runtime hard
  mask mode. Exact trace/Frobenius and Cholesky operations remain intact.
- Fixes changed evaluation order and gradient connectivity, not the source's
  message thresholds, precision limits, numerical tolerances or WP scope.

## Artifacts and acceptance mapping

`runs/wp0-inspect/resolved_config.yaml` and `environment.json` contain the full
configuration, environment, source specification hash, actual Git HEAD/dirty
state and sorted package-source hashes. Generated run files are Git-ignored.
The task's `research/final-source-manifest.json` additionally fingerprints
product source, tests, profiles and build configuration at validation.

| Criterion | Actual evidence |
| --- | --- |
| A1 | Editable install, package import, both executable CLI entries |
| A2 | All required physical values independently asserted in CLI receipts |
| A3 | Nested errors, type/enums/derived constraints, profile/roundtrip tests |
| A4 | CPU CUDA-call guards, explicit failure, paired precision tests |
| A5 | Both actual oracles, limits, parameter count, backward and gradcheck |
| A6 | Four-point enumeration, full-graph equivalence, exact solves, AST audit |
| A7 | Source/environment/code hashes, full config, actual command receipts |

Commit/archive remain pending the user's requested diff/results review. No
next work package was created or activated.

## WP3 implementation validation (2026-09-18; historical implementation run)

Environment: Windows, `.venv/Scripts/python.exe`, Python 3.13.9, PyTorch 2.12.0+cpu,
`MKL_THREADING_LAYER=TBB` set before launch. CPU uses four threads; no CUDA
acceptance is claimed. Commands and artifacts are recorded in the WP3 task's
`research/implementation-evidence.md`.

- Initial WP3 targeted tests: **44 passed** in 15.23 s.
- Full WP0–WP3 regression: **238 passed, 3 skipped** in 29.40 s. Three skips require CUDA.
- After restricted-load error handling and two added regressions: materialization
  suite **18 passed** in 6.94 s (includes unsafe pickle rejection and failed export publication).
- Ruff product checks and mypy passed. No numerical tolerances were relaxed.
- Actual simulate/audit/materialize commands completed with `configs/wp3_smoke.yaml`.
  This config reduces only frame counts: full 512/400/8192/8 dimensions remain.
- `runs/wp3-acceptance-data/manifest.json`: 1 train frame, 1 validation frame,
  2 independent test frames paired over 2 SNR values, 48 total expanded samples.
- `runs/wp3-acceptance-audit/`: two complete physical frames, 16 windows;
  maximum independent waveform/H relative error **1.4435423701359176e-11**.
- `scripts/verify_wp3_artifacts.py` independently loaded the saved arrays and used
  NumPy FFT/cancellation/SNR reconstruction, verifying all 16 windows and nonzero
  unequal path epsilons. Maximum NumPy error **1.4435457958165196e-11**.
- `runs/wp3-acceptance-test.pt`: all 32 samples exactly match compact replay.
  Tensor payload 82,355,456 bytes, reserved estimate 83,469,568 bytes, actual file
  82,376,190 bytes. The 8192-matrix complex128 example is unit-tested as
  **20,971,520,000 bytes for H alone**; no such large file was generated.
- `runs/wp3-artifact-verification.json` preserves manifest/file and sample hashes.
  Tests also cover three independent consumers, cross-process replay, cache
  mutation/eviction/order, split/channel leakage, SNR pairing, random stream/retry
  isolation, training mixture, dtype, corrupt inputs and label-free inference.

Actual three-detector integration belongs to WP4–WP7. This WP3 evidence does not
claim production detector execution, main training, BER curves or full-system smoke.

## WP3 independent review (2026-09-18)

Final product Ruff lint/format and mypy pass (55 formatted files; 37 typed source
files). Complete regression: **248 passed, 3 CUDA skipped in 33.01 s**. All Python
commands use the environment above with `MKL_THREADING_LAYER=TBB` before startup.
Exact argv, timestamps, stdout/stderr and final source fingerprints are preserved
in the WP3 task's `research/check-final-command-receipts.json`.

Review fixes enforce canonical frame/channel SHA-256 identities, source hash
syntax, sample block/SNR identity and compact SNR-copy consistency. Capacity
guards now account for three independent file reserves and the full split table
and resolved configuration stored with each export. Regression tests verify
rejection before generation and compare actual file size against its estimate.
Single precision has a separate full-size effective/waveform comparison with
`atol=2e-6, rtol=2e-5`; double precision retains the source's unchanged 1e-9 bound.

Final artifacts use `runs/wp3-review-final2-{data,audit,inspect}`, plus
`runs/wp3-review-final2-test.pt` and `runs/wp3-review-final2-verification.json`.
The independent array verifier reconstructs quadrature seeds and noise samples,
FFT, pilot cancellation, SNR and compact/dense parity. See the task's
`research/check-report.md` for acceptance mapping, precise hashes and limitations.

## Later-package validation not executed

WP2 physical waveform/channel tests and WP4/WP5 production algorithm checks are
recorded above at their actual review stages. WP6 bounded training, checkpoint
recovery and complete system smoke are recorded separately above. Main/full
training and statistically powered BER/performance sweeps remain unexecuted.
WP7 bounded statistical reports and both benchmark protocols are now validated
separately above; WP8 delivery remains unstarted.
