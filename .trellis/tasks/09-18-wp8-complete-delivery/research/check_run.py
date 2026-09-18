"""Independent WP8 exact-command receipts; no production imports."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESEARCH = Path(__file__).parent
OUTPUT = ROOT / "runs/wp8-check-commands"
OUTPUT.mkdir(exist_ok=False)
os.environ["MKL_THREADING_LAYER"] = "TBB"
python = str(ROOT / ".venv/Scripts/python.exe")
commands = [
    ("pytest", [python, "-m", "pytest", "-q", "--basetemp=runs/wp8-check-full", "-ra"]),
    ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
    ("format", [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]),
    ("mypy", [python, "-m", "mypy", "src"]),
    ("diff", ["git", "diff", "--check"]),
    *[
        (
            dtype,
            [
                python,
                "-m",
                "pgvamp_ofdm",
                "demo-frame",
                "--config",
                "configs/cpu_dev.yaml",
                "--device",
                "cpu",
                "--dtype",
                dtype,
                "--output",
                f"runs/wp8-check-demo-{dtype}",
            ],
        )
        for dtype in ("complex128", "complex64")
    ],
]
receipts = []
for name, argv in commands:
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, env=os.environ)
    stdout, stderr = OUTPUT / f"{name}.stdout.txt", OUTPUT / f"{name}.stderr.txt"
    stdout.write_bytes(result.stdout)
    stderr.write_bytes(result.stderr)
    receipts.append(
        {
            "name": name,
            "argv": argv,
            "cwd": str(ROOT),
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.perf_counter() - before,
            "exit_code": result.returncode,
            "MKL_THREADING_LAYER": os.environ["MKL_THREADING_LAYER"],
            "stdout": str(stdout.relative_to(ROOT)),
            "stderr": str(stderr.relative_to(ROOT)),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "stdout_text": result.stdout.decode("utf-8", errors="replace"),
            "stderr_text": result.stderr.decode("utf-8", errors="replace"),
        }
    )
    (RESEARCH / "check-command-receipts.json").write_text(
        json.dumps(receipts, indent=2), encoding="utf-8"
    )
    print(name, result.returncode, round(time.perf_counter() - before, 2), flush=True)
