# WP3 independent review — passed, 2026-09-18

Reviewed by the dispatched trellis-check agent directly. The source specification
remains authoritative (SHA-256 `a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b`).
Read the complete saved native hook, task artifacts, curated manifests and actual
source §§5–11, 15.2–15.3, 16.2, 20–23. No WP4 code, numerical relaxation, commits,
archive or task-metadata changes were made by the checker.

## Findings (fixed)

- `src/pgvamp_ofdm/data/records.py`: nonempty arbitrary frame/channel IDs and
  inconsistent SNR-copy names could enter the schema. Enforced the existing
  generator's canonical lowercase SHA-256 identities and split/SNR copy grammar.
  Tests reject nonhex IDs, a 10,000-character ID and invalid copy names.
- `src/pgvamp_ofdm/data/materialize.py`: source hashes were length-only checked,
  and sample IDs only checked their frame prefix. Enforced hexadecimal hash
  syntax, canonical lineage IDs, integer sample count and the configured SNR-copy
  and block identity. Added malformed hash/block/SNR regression cases.
- `src/pgvamp_ofdm/data/generate.py` and `materialize.py`: config-selected export
  reserved overhead once although three files are written; a tiny selected split
  still stores the complete split table. Sum per-file estimates and reserve full
  configuration/lineage overhead, bounded by canonical generated IDs. Budget
  failure remains before record/dense generation and output creation. Tests cover
  the old aggregate boundary and actual export size below the estimate.
- `tests/test_dataset.py`: single precision previously checked only finite y
  and H casting. Added actual 8192-FFT waveform/effective observation parity in
  complex64 with separate `atol=2e-6, rtol=2e-5`, allowing single-precision FFT and
  matrix rounding. The double physical 1e-9 criterion is unchanged.
- `scripts/verify_wp3_artifacts.py`: made double-precision scope explicit and
  independently rederived SHA-256 quadrature seeds and time-domain noise samples.
  The initial extra check used sqrt(sigma2/2), differing by up to 3.1e-16 relative
  from the implementation's underflow-safe sqrt(sigma2)/sqrt(2). Matching that
  documented scalar evaluation order restores bitwise real/imag equality; no
  tolerance was loosened. The failed first check and successful rerun receipts
  are retained for transparency.

## Findings (not fixed)

No remaining in-scope blocker or unresolved code finding. CUDA cannot be accepted
without hardware (three tests skipped). Actual three-detector shared-input and
target-isolation integration, training/checkpoints, full-system smoke and main
BER/performance sweeps remain WP4–WP8 work, not WP3 claims. Materialize is a CPU
storage export; explicit CUDA consumer replay remains an EffectiveDataset option.
Audit is intentionally CPU complex128, independent of consumer precision.

## Verification

- Lint: pass, `ruff check src tests scripts`.
- Format: pass, `ruff format --check src tests scripts`, 55 files.
- TypeCheck: pass, `.venv/Scripts/python.exe -m mypy src`, 37 source files.
- Tests: **248 passed, 3 skipped in 33.01 s**, complete pytest on final source.
- Fresh simulate, two-full-frame audit, test materialization, independent saved
  array verification and inspect-config: all exit 0.
- `git diff --check`: pass; normal Windows line-ending conversion notices only.
- Windows, Python 3.13.9 / PyTorch 2.12.0+cpu, CPU four threads,
  `MKL_THREADING_LAYER=TBB` set before Python startup. No dependency upgrades or
  unsafe duplicate-OpenMP override.

Exact argv, UTC timestamps, output, exit status and src/tests/scripts/config
fingerprints: `check-final-command-receipts.json`. The executable receipt driver
is `run-review-checks.py --final`; use fresh paths for a future independent run.
Earlier receipts are historical (`check-command-receipts.json`,
`check-followup-command-receipts.json`), including the first independent noise
assertion failure. They do not override final green results.

## Final persisted artifacts and independent checks

- `runs/wp3-review-final2-data/manifest.json`: SHA-256
  `f0eabb8465f07c5c2eef4d2f2fb60a01d5ed1fbb1b86e9b5d25cf9c78bad42eb`.
- Package source SHA-256:
  `5450d72a21f5b1c582e5a4a3bcb5fcc2bab5e2bc42ac33186a276e22d4db0a89`.
- `runs/wp3-review-final2-audit/audit.json`: SHA-256
  `9197447fa756a8b179ae88f95ad2dfdd11d91a4ec9a816cf46bd05a73a1f9712`.
- `runs/wp3-review-final2-test.pt`: SHA-256
  `a2b4c170596b220876c72c28ce4d15db0c22723434ffe485db616cf78c012446`.
- `runs/wp3-review-final2-verification.json`: 16 windows, nonzero unequal epsilon
  verified, maximum independent NumPy relative error `1.4435457958165196e-11`.
  Production audit maximum `1.4435423701359176e-11`, both below 1e-9.
- H payload 81,920,000 bytes, other tensors 435,456 bytes, metadata/serialization
  reserve 1,125,168 bytes; estimated 83,480,624 bytes, actual 82,376,223 bytes.
  All 32 materialized samples equal compact replay bitwise for H/y/sigma2/x/bits.
- Separate checker `inspect-review-manifest.py runs/wp3-review-final2` imports no
  product manifest loader. It independently checks raw file/config hashes,
  frame/channel disjointness, counts, frequency-index separation, paired SNR
  paths/bits/pilots, distinct noise seeds and current package source fingerprints.
  Output: `check-independent-manifest.json`. Train and val each have one frame /
  eight samples; test has two independent frames, four copies / 32 samples.

## Acceptance mapping

| Acceptance | Evidence |
| --- | --- |
| A1 | Safe load/schema/tamper/label tests; independent hashes/mapping/count checks |
| A2 | Cross-split frame/channel tests, raw manifest audit, paired SNR and retry/stream tests |
| A3 | Out-of-order access, cache eviction/mutation, cross-process hashes, no_grad and per-block H tests |
| A4 | Fixed-seed 400-frame mixture/SNR test, configured counts and paired-frame audit |
| A5 | Both dtype payload formula tests; 20,971,520,000-byte H-only example; pre-generation guards; actual dense size/parity |
| A6 | Unlabeled inference/missing-label errors; three independent consumer hashes; H/y/sigma2-only projection |
| A7 | Fresh complete-size two-frame physical audit; NumPy FFT/cancellation/noise/SNR; full WP0–WP3 regression |

The coordinator owns final task/spec/status reconciliation and user-facing review.
