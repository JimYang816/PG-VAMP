# WP8 documentation evidence — 2026-09-18

Owned changes: README.md, docs/CONFIGURATION.md, docs/ALGORITHMS.md,
docs/REPRODUCING.md. No product code or external WP7_EXAMPLE_REPORT.md/output/
was changed by this agent. No product experiments were run by this agent.

Read the native hook saved-output file, task PRD/design/implement context,
trellis-start and trellis-before-dev skills, backend/guides indexes, runtime and
numerical guidelines. Source §§2–17 and 20 were read in bounded ranges before
writing physical/config/math explanations. Source §§18, 21–23 were supplied in
curated source excerpts; full specification remains authority and unchanged.
FastCtx tools were unavailable; PowerShell/rg fallback was used.

Boundary: stale README current-WP6/future-WP7 wording replaced with current
delivery navigation; separate config, math and recipe guides avoid duplicating
actual experiment evidence owned by VALIDATION. Each command family has input,
output, prerequisite and scale explanations. Main/full sweep/sufficient
multi-seed/CUDA results explicitly remain 未执行. Historical WP6/WP7 reviews link
to archive paths without recycling their counts as new evidence.

Source checks: CLI parser flags; config loader merge/device/precision behavior;
all config YAMLs and bundled defaults; pyproject Python/dependency requirements;
WP6/WP7 acceptance harness scope; smoke/training/inference output paths; actual
module path inventory. Demo contract coordinated with wp8_code and checked
against demo.py: physical-only/full-size, fresh directory, strong affine channel,
fixed 10 dB, optional seed, full-frame artifacts, separate noiseless real-RF LFM
vs noisy ideal-IQ oracle receive windows, no training/detectors.

Static checks performed:
- Python pathlib/regex checked all 24 relative Markdown links across the four
  documents: zero missing targets.
- Markdown code-fence balance checked in all four documents: passed.
- `git diff --check`: exit 0, only Git's informational LF/CRLF conversion warning.
- Reviewed actual configs/cuda_example.yaml filename and actual module directories;
  corrected initial draft paths before delivery.

No metrics, convergence, BER advantage, sparse speedup or fresh-install success
is inferred. Commands are labeled recipes; bounded harness executions and final
whole-project checks are coordinated by root and independently checked later.
Main CUDA recipe requires a separate precision-consistent config/manifest/run;
inference precision conversion is distinguished from strict training resume.
