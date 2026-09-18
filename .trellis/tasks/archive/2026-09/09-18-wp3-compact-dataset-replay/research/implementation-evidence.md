# WP3 implementation evidence — 2026-09-18

Implemented directly by dispatched trellis-implement; independent check pending.
No commits, archive, task-metadata edits, WP4 implementation or main-scale experiments.

## Commands actually executed

All Python commands use `$env:MKL_THREADING_LAYER='TBB'` before interpreter startup.
Ruff uses `C:/Software/Anaconda3/Scripts/ruff.exe`: `python -m ruff` failed initially
because this system-site-packages venv does not contain the ruff.exe launcher.
No unsafe duplicate-OpenMP override was used.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_data_records.py tests/test_data_random.py tests/test_dataset.py tests/test_materialize.py tests/test_data_cli.py --basetemp=runs/pytest-wp3-targeted-2
# 44 passed in 15.23s
.venv/Scripts/python.exe -m pytest -q --basetemp=runs/pytest-wp3-full
# 238 passed, 3 skipped in 29.40s (CUDA hardware)
.venv/Scripts/python.exe -m pytest -q tests/test_materialize.py --basetemp=runs/pytest-wp3-safe-load
# 18 passed in 6.94s, after restricted-load error wrapper and 2 additional tests
.venv/Scripts/python.exe -m pytest -q tests/test_config.py --basetemp=runs/pytest-wp3-summary
# 33 passed in 0.25s after aligning inspect-config compact estimate to explicit pilots
C:/Software/Anaconda3/Scripts/ruff.exe format --check src tests scripts
# 55 files already formatted (final)
C:/Software/Anaconda3/Scripts/ruff.exe check src tests scripts
# All checks passed
.venv/Scripts/python.exe -m mypy src
# Success: no issues found in 37 source files
.venv/Scripts/python.exe -m pgvamp_ofdm simulate --config configs/wp3_smoke.yaml --output runs/wp3-acceptance-data
.venv/Scripts/python.exe -m pgvamp_ofdm audit-data --manifest runs/wp3-acceptance-data/manifest.json --waveform-frames 2 --output runs/wp3-acceptance-audit
.venv/Scripts/python.exe -m pgvamp_ofdm materialize --manifest runs/wp3-acceptance-data/manifest.json --split test --output runs/wp3-acceptance-test.pt --max-output-bytes 1073741824
.venv/Scripts/python.exe scripts/verify_wp3_artifacts.py --manifest runs/wp3-acceptance-data/manifest.json --audit runs/wp3-acceptance-audit --materialized runs/wp3-acceptance-test.pt --output runs/wp3-artifact-verification.json
.venv/Scripts/python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml --output runs/wp3-inspect
```

## Actual artifacts

- `runs/wp3-acceptance-data/manifest.json`, SHA256
  `4eb54088d7e2f556c2a1f55d934cd96d5f620c1aa77cf73de65e165536f488f3`.
- Train=1 independent frame/8 samples; val=1/8; test=2 independent frames,
  4 SNR copies/32 samples; static + moderate test scenarios, SNR 0/10 dB.
- `runs/wp3-acceptance-audit/audit.json`: complete saved transmit analytic/real and
  received IQ frames (including arrival offset), records, 16 windows, H, noise,
  FFT observations, pilot contributions, processed H/y, sigma2 and receive SNR.
  Nonzero unequal epsilon paths present. Worst double waveform/H relative error
  `1.4435423701359176e-11`; independent saved-array NumPy FFT error
  `1.4435457958165196e-11`, below unchanged 1e-9 bound.
- `runs/wp3-acceptance-test.pt`, SHA256
  `95650de0797f7f4bc125af2eff5094632e5aeab92b36e1c9fb044204f6416e98`.
  H-only bytes=81,920,000, other tensors=435,456, payload=82,355,456,
  reserve=1,114,112, estimate=83,469,568, actual=82,376,190.
- `runs/wp3-artifact-verification.json`: all 32 dense samples bitwise equal to
  compact replay; 16 independently reconstructed FFT/cancellation/SNR windows;
  per-sample payload fingerprints retained.

Artifacts preserve the package-source fingerprint at generation time. Subsequent
safe-load error handling does not change their physical values; final checker may
generate fresh artifacts against its reviewed final source.

## Scope and implementation choices

- Generation IDs hash physical generation settings, excluding runtime/backend,
  storage and detector settings, so dtype/backend changes retain paired frames.
- CPU-first compact records retain double physical precision; explicit device/dtype
  replay obeys consumer runtime. Six required streams plus scenario/SNR domains use
  stable SHA256 JSON derivation. Channel retries do not consume other streams.
- Storage `materialized` uses a pending manifest until every requested export
  succeeds. Failed output preserves diagnostics but has no final manifest.json.
- `audit-data` is explicit; omitted frame count uses `data.audit_waveform_frames`.
  `simulate` records this policy and does not run an implicit expensive waveform audit.
- `lfm_detect` is rejected by the data pipeline; no silent oracle timing fallback.
- Label-free inference, split lineage, safe loading, count/config/index validation,
  no-grad/mutation-safe cache and preallocation budget guard are implemented.

## Remaining follow-ups

Independent Trellis check and its final source/artifact receipts remain pending.
CUDA unavailable. Actual production three-algorithm input integration/target isolation
belongs to WP4–WP7. Main generation, training, BER sweeps and system smoke unexecuted.
