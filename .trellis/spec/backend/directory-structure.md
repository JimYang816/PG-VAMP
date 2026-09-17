# Structure and interfaces

Source: [execution specification](../../../docs/CODEX_ENGINEERING_SPEC.md)
§§0.1–0.3, 11, 19–21. These rules are subordinate to that document.

- Use Python + PyTorch with pyproject.toml and src/pgvamp_ofdm/.
  Support both pgvamp-ofdm and python -m pgvamp_ofdm.
- Keep configs/ for base/profile YAML, tests/ for pytest, scripts/ for launch
  helpers, docs/ for explanations. Generated data/, runs/, results/ are ignored
  by Git. The exact module/file map is in §19; create it only as WP work proceeds.
- Under src/pgvamp_ofdm/, modulation/waveform/channel/receiver own the physical
  chain; data owns records and replay; algorithms owns detection; training owns
  optimization/checkpoints; evaluation owns metrics/timing/reporting; utils owns
  device/random/logging/validation support.
- Keep reference/dense_pg_vamp.py and reference/dense_vamp_cholesky.py independent
  of production algorithm implementations. Their initial correctness versions
  belong to WP0; production baselines and PG-VAMP belong to WP4/WP5.
- Follow §11 Detector/DetectionResult shapes and fields. Detect receives only
  H, y, sigma2 and diagnostic options, never targets, bits, SNR labels or path
  truth. Inference may lack labels; training/error-rate evaluation may not.
- Public functions require types, shapes, dtype/device requirements and input
  checks. Allow layer/path/reference-block loops; use batched linear algebra
  rather than ordinary per-sample Python loops.
- Core dependencies: PyTorch, NumPy, PyYAML, Matplotlib; pytest for tests;
  psutil optional for memory measurement. SciPy is not required for simulation.
  Prefer compatible installed PyTorch; do not upgrade unrelated dependencies
  or force CUDA installation. Python 3.11/3.12 is a recommendation, not a claim
  of tested compatibility.
- No runtime dependency on external oracles, older projects, MATLAB, IDEs,
  proprietary simulators, cloud services or downloaded real data. Reuse existing
  workspace material only when the user requests it (§0.1).

Contract examples already present in the source: §11 detector protocol, §12.1
Cholesky solve and §19 directory tree. They are design evidence, not executed code.
