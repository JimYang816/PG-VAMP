# WP8 implementation evidence — 2026-09-18

User approved the latest planning and implementation. Activated child on
`codex/wp8-complete-delivery`, starting HEAD `3c7d68f`. The untracked
`docs/WP7_EXAMPLE_REPORT.md` predates this implementation turn and is outside
our edits/commit scope. Authoritative specification and independent reference
files are unchanged; see authority-reference-baseline.json.

## Fresh training and full-size smoke acceptance
Actual command: `.venv/Scripts/python.exe scripts/wp6_acceptance.py --output
runs/wp8-training-acceptance`, with `MKL_THREADING_LAYER=TBB` before launch.
All 11 subprocess commands exited 0; exact commands/stdout/stderr are copied
to training-cli-receipts.json. They cover both mathematical precisions, real
512/400/8192/CP2048/eight-block/T8/two-update system smoke, generation, waveform
audit, training, strict resume to four updates, materialization and inference.

Actual independent artifact command:
`.venv/Scripts/python.exe .trellis/tasks/09-18-wp8-complete-delivery/research/audit-system-artifacts.py`.
Exit 0. The NumPy audit rehashes inputs/checkpoints, recomputes bit/symbol/block/
frame counts and energy, verifies complete eight-block identities, no inference
labels, prediction roundtrip, exactly 16 trainable scalars/two smoke updates,
and independently FFTs saved physical waveforms and subtracts pilots.
Maximum relative waveform/H error: **1.1310709079623577e-11** (<1e-9).
Results/hashes/environment: system-artifact-audit.json.

These are small acceptance workloads, not convergence or comparative performance
evidence. Final demo/evaluation/report/full-regression/independent review evidence
will be added separately; no pending step is declared passed here.

## Fresh evaluation, timing and reports
Actual commands: `.venv/Scripts/python.exe scripts/wp7_acceptance.py --output
runs/wp8-evaluation-acceptance`, followed by the same command with `--reports-only`.
All 15 non-report and two report subprocesses exited 0. Exact receipts are
evaluation-receipts.json and evaluation-report-receipts.json.

Independent audit command: `python .trellis/tasks/09-18-wp8-complete-delivery/research/audit-evaluation-artifacts.py
runs/wp8-evaluation-acceptance/evaluation1 runs/wp8-evaluation-acceptance/evaluation2
--output .trellis/tasks/09-18-wp8-complete-delivery/research/evaluation-artifact-audit.json`.
Both bundles pass SHA-256 validation of 13 artifacts, lineage/shared-input checks,
and independent count/energy/rate recomputation over all six aggregate cells.
No failed blocks were dropped. The coordinator inspected all ten saved single-run
figures using QA contact sheets; report-visual-review.json records hashes and checks.

Two training seeds each received only two updates. Evaluation uses one physical
frame at two SNRs (0/10 dB), moderate Doppler, and three timing repeats. Repeated
SNR copies are not independent physical frames. This is CLI/report acceptance,
not a powered multi-seed comparison. Demo and final independent full regression
are separate checker responsibilities.
