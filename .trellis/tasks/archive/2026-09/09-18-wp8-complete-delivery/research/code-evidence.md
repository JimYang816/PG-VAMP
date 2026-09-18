# WP8 demo-frame implementation evidence — 2026-09-18

Ownership: `src/pgvamp_ofdm/demo.py`, `src/pgvamp_ofdm/cli.py`,
`tests/test_demo.py`. No physical formulas, detectors, oracles, data/checkpoint
formats or other agents' docs were changed.

## Boundary and contract

Read the native hook's saved context, task PRD/design/implementation plan,
trellis-start and trellis-before-dev, backend/guides indexes and applicable
runtime/reproducibility/structure/WP1/WP2/quality/reuse/cross-layer contracts.
The behavior gap is the missing §21.6 CLI delivery entry, despite reusable
waveform/channel/receiver implementations. Added a package-level orchestrator,
CLI dispatch and focused tests; no dependency on repository audit scripts.

`demo-frame --config <physical YAML> --device cpu --dtype complex128
--output <fresh directory> [--seed N]` generates one frame. CPU default and
explicit runtime precision follow existing resolver. Algebra fixtures,
non-full dimensions, `lfm_detect` reception and all existing output paths are
rejected. Full dimensions are 512 grid / 400 data / 8192 FFT / CP2048 / eight
blocks; this is independent of a profile's dataset/train budget.

Seeded strong affine paths must contain distinct nonzero epsilon and pass CP
support for all blocks. The complete continuous physical recording includes
its delayed/time-stretched tail. Each reference window is sliced from that
recording; actual FFT is checked against independently constructed analytical
H at relative error <1e-9. Example H/y use selected precision. Es/N0 is fixed
at 10 dB, sigma2=0.1; independently seeded real/imag AWGN is injected into IQ.

LFM correlation uses noiseless real receive RF and is marked as such (there is
no real-noise front-end claim). Its selected template peak, injected frame
origin and earliest physical LFM support are separately named. The peak is
never used to shift oracle FFT windows. No detector or training runs in demo.

## Artifacts

- `frame.pt`: safe tensor/basic-type dictionary with schema_version, layout,
  allocation, full grid/bits/pilots, tx_analytic/tx_real,
  rx_analytic_noiseless/rx_real_noiseless/rx_iq_noiseless/iq_noise/rx_iq_noisy,
  lfm_real, sync result, paths, CP endpoints, arrival offset, independent seeds,
  Es/N0/variance and eight windows. Each window contains full reference grid H,
  reference FFT, runtime H/y/sigma2, x/bits/pilots and recording start/end.
- `resolved_config.yaml`, `environment.json`: actual runtime and source hashes.
- `figures/waveform_and_sync.png`, `figures/channel_and_constellation.png`:
  rendered by reloading saved arrays, not a new simulation.
- `demo.json`: last-write success receipt with argv, UTC timestamps, dimensions,
  seed streams, resampling count, independent errors, explicit sync limitations,
  and hashes of every other output. Plot/other failure leaves no success receipt.

## Checks actually run

Environment: `.venv/Scripts/python.exe`, `MKL_THREADING_LAYER=TBB`.

1. `python -m ruff check src/pgvamp_ofdm/demo.py src/pgvamp_ofdm/cli.py tests/test_demo.py`
   passed; matching Ruff format --check passed.
2. `.venv/Scripts/python.exe -m mypy src/pgvamp_ofdm/demo.py src/pgvamp_ofdm/cli.py`
   passed (2 source files).
3. `.venv/Scripts/python.exe -m pytest -q tests/test_demo.py tests/test_cli.py
   --basetemp=runs/wp8-demo-tests-v3 -ra`: **11 passed in 26.64 s**.
   Tests cover both precisions, complete dimensions, independent FFT and pilot
   cancellation/noise accounting, direct selected-lag correlation, independent
   random stream reconstruction, full rerun determinism, hash verification,
   failed plot/no success receipt, fresh-dir/config errors, CLI dispatch and
   subprocess rejection. Existing CLI tests are included.
4. `git diff --check` passed (unrelated README CRLF advisory only).
5. Opened both complex128 PNGs from v3 output: complete waveform/correlation,
   thresholds, axes/legends, matrix and receive/reference constellation legible.

Concrete v3 complex128 artifact:
`runs/wp8-demo-tests-v3/demo0/frame/demo.json`: 91776 transmit samples,
93248 recording samples, 6400 bits, 8 windows; max independent relative error
4.590823433761398e-12; arrival offset 267; selected peak 2592;
earliest physical LFM start 2379.4039961580875. This demonstrates why selected
peak must not be called the first path. No performance conclusion follows.

## Failed attempts retained and numerical explanation

`runs/wp8-demo-tests` initial tests: 2 failed / 4 passed. Independent indexing
gave a different BLAS layout, so bitwise comparison of pilot cancellation was
inappropriate (errors about 6e-16 double / 3e-7 single).
`runs/wp8-demo-tests-v2`: 1 failed / 10 passed. A componentwise single-precision
comparison of `y-Hx` to near-zero noise failed at one element (2.39e-6).
Final tests apply double 1e-9 and WP2's normwise single 2e-6 threshold, with
the latter relative to observation energy before cancellation. Dense 400-term
single-precision reduction accumulates rounding, and a near-zero residual is
not an appropriate relative scale. The product's independent double 1e-9
gate and all physical/oracle formulas were unchanged. Failed artifacts remain.

## Remaining work for root/checker

Source is ready for independent review, actual CLI acceptance with root's
command receipts, final full regression/type/lint, and document/status updates.
No full suite or costly acceptance harness was duplicated here. CUDA is not
tested by this focused CPU run; no main training or SNR sweep was run.
