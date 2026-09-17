# Planning evidence — 2026-09-18

## Inspected authority
docs/CODEX_ENGINEERING_SPEC.md:294 (§5), :355 (§6), :508 (§7), :547 (§8),
:610 (§9), :1446 (§19), :1601 (frame/channel/receiver defaults), :1851 (WP2),
:1895 (§23) establish this child scope and numerical gates.
Parent PRD defines WP0→WP1→WP2 dependency order; WP1 work is committed at 414de3f,
archived by a5ae406. Historical 166 passed / 2 skipped is the parent record,
not a fresh run and not WP2 acceptance.

## Existing interfaces inspected
- src/pgvamp_ofdm/waveform/frame.py: FrameLayout half-open intervals;
  FrameWaveforms retains grid/layout/allocation; build_frame consumes explicit bits/pilots,
  no RNG; global OFDM carrier and separate local LFM phase.
- src/pgvamp_ofdm/waveform/lfm.py: symmetric Tukey u=n/(N-1), discrete power
  normalization and float64 phase; independent continuous evaluator must match integer samples.
- src/pgvamp_ofdm/config.py: initial delay bound exists but does not replace per-block
  dynamic CP support; identity/static inspection semantics already distinguish channel types.
- .trellis/spec/backend/wp1-waveform-contract.md: default useful starts
  9856+m*10240, frame length 91776, runtime/validation and timing contracts.
- pyproject.toml: pytest, Ruff and mypy configuration; current source tree has no channel
  implementation or FFT/preprocessing receiver modules.

## Decisions and deferred details
Follow existing source-defined product scope; no unresolved user-owned decision found.
Use direct exponentials for independence, explicit random sources, rejection rather than
silent resampling, and actual-window CP/H validation. Statistical sample counts and block
chunk sizes are implementation details to choose and record before testing; no weakened
source acceptance or model changes permitted. Dataset persistence is WP3.
No web research needed: the execution specification explicitly supplies the physical model.
