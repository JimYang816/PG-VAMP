"""Reproduce WP4 implementation receipts; no main training or evaluation."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import shutil
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["MKL_THREADING_LAYER"] = "TBB"
ruff = shutil.which("ruff")
if ruff is None:
    raise RuntimeError("Install Ruff before running validation")
commands = [
    [sys.executable, "-m", "pytest", "-q", "--basetemp=runs/pytest-wp4-receipt"],
    [ruff, "check", "src", "tests"],
    [ruff, "format", "--check", "src", "tests"],
    [sys.executable, "-m", "mypy", "src"],
    [sys.executable, "-m", "pgvamp_ofdm", "inspect-config", "--config", "configs/cpu_dev.yaml"],
    [sys.executable, "-m", "pgvamp_ofdm", "inspect-config", "--config", "configs/vamp_reference_32.yaml"],
]
lint_only = "--lint-only" in sys.argv
if lint_only:
    commands = commands[1:3]
receipt = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
           "environment": {"MKL_THREADING_LAYER": "TBB"}, "commands": []}
for command in commands:
    started = time.time()
    proc = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
    receipt["commands"].append({"argv": command, "started_unix": started,
        "elapsed_seconds": time.time() - started, "exit_code": proc.returncode,
        "stdout": proc.stdout, "stderr": proc.stderr})
    receipt_path = OUT / ("implementation-lint.json" if lint_only else "implementation-validation.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(command[2:5], proc.returncode, proc.stdout[-180:], flush=True)

if lint_only:
    sys.exit(int(any(c["exit_code"] for c in receipt["commands"])))

import torch
from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector
from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.records import tensor_hash

torch.set_num_threads(4)
manifest = ROOT / "runs/pytest-wp4-receipt/data0/compact/manifest.json"
dataset = EffectiveDataset(manifest, "test")
sample = dataset[16]
payload = detection_inputs(sample)
physical = {"sample_id": sample["sample_id"], "input_sha256": tensor_hash(payload),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "path_epsilon": dataset.records[2]["path_epsilon"].tolist(),
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "device": "cpu", "dtype": str(payload["H"].dtype),
            "shape": list(payload["H"].shape), "detectors": {}}
for model in (MMSEDetector(), VAMPDetector()):
    output = model.detect(**{key: value[None] for key, value in payload.items()})
    physical["detectors"][model.name] = {
        "input_sha256_after": tensor_hash(payload),
        "output_sha256": tensor_hash({"x_soft": output.x_soft, "bits_hat": output.bits_hat}),
        "finite": bool(torch.isfinite(output.x_soft).all()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "counts": {k: v.tolist() for k, v in output.diagnostics.items() if isinstance(v, torch.Tensor)},
    }
(OUT / "implementation-physical.json").write_text(json.dumps(physical, indent=2), encoding="utf-8")
paths = [*ROOT.glob("src/pgvamp_ofdm/**/*.py"), *ROOT.glob("tests/test_*.py"),
         ROOT / "docs/CODEX_ENGINEERING_SPEC.md", ROOT / "configs/vamp_reference_32.yaml"]
hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT / "implementation-source-hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
if any(c["exit_code"] for c in receipt["commands"]):
    sys.exit(1)
