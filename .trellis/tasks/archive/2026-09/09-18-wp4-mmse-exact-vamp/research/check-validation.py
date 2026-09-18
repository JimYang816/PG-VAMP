"""Independent final review receipt; fresh pytest directory each execution."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["MKL_THREADING_LAYER"] = "TBB"
import torch

commands = [
    [sys.executable, "-m", "pytest", "-q", "-rs", f"--basetemp=runs/pytest-wp4-check-{time.time_ns()}"],
    ["C:/Software/Anaconda3/Scripts/ruff.exe", "check", "src", "tests"],
    ["C:/Software/Anaconda3/Scripts/ruff.exe", "format", "--check", "src", "tests"],
    [sys.executable, "-m", "mypy", "src"],
    [sys.executable, "-m", "pgvamp_ofdm", "inspect-config", "--config", "configs/cpu_dev.yaml"],
    [sys.executable, "-m", "pgvamp_ofdm", "inspect-config", "--config", "configs/vamp_reference_32.yaml"],
    [sys.executable, str(OUT / "check-numerics.py")],
]
receipt = dict(platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),
    python=sys.version, executable=sys.executable,torch=torch.__version__,cuda_available=torch.cuda.is_available(),
    cuda_device_count=torch.cuda.device_count(),pytest_threads=4,MKL_THREADING_LAYER="TBB",commands=[])
for command in commands:
    start = time.time()
    proc = subprocess.run(command,text=True,capture_output=True,encoding="utf-8",errors="replace")
    receipt["commands"].append(dict(argv=command,started_unix=start,elapsed_seconds=time.time()-start,
        exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    (OUT/"check-validation.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(command[1:5],proc.returncode,proc.stdout[-350:],flush=True)
paths = [*ROOT.glob("src/pgvamp_ofdm/**/*.py"),*ROOT.glob("tests/*.py"),
         ROOT/"docs/CODEX_ENGINEERING_SPEC.md",ROOT/"configs/vamp_reference_32.yaml"]
hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/"check-source-hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True),encoding="utf-8")
sys.exit(int(any(c["exit_code"] for c in receipt["commands"])))
