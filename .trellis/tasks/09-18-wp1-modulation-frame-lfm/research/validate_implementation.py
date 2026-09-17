"""Reproduce the implementer's final command receipts without deleting prior evidence."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[4]
os.chdir(root)
python = str(root / ".venv/Scripts/python.exe")
ruff_python = "C:/Software/Anaconda3/python.exe"
env = dict(os.environ, MKL_THREADING_LAYER="TBB")
receipt = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "scope": "WP1 implementation verification; not check-agent approval or WP2",
    "environment": {"MKL_THREADING_LAYER": "TBB", "python_executable": python,
                    "ruff_python": ruff_python, "runner_python": sys.version},
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    "commands": [],
    "preliminary": [
        {"command": "$env:MKL_THREADING_LAYER='TBB'; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_qpsk.py tests/test_allocation.py tests/test_waveform.py tests/test_lfm.py --basetemp=runs/wp1-targeted-01 -ra",
         "exit_code": 0, "result": "80 passed, 1 skipped in 1.85s (CUDA hardware unavailable)"},
        {"command": "python -m ruff check src tests scripts/run_wp1_audit.py",
         "exit_code": 1, "result": "52 line-length issues on first draft; formatting fixed 50, two string literals manually wrapped; no numerical failure"},
        {"command": ".\\.venv\\Scripts\\python.exe -m mypy src scripts/run_wp1_audit.py",
         "exit_code": 0, "result": "Success: no issues found in 21 source files"},
        {"command": "$env:MKL_THREADING_LAYER='TBB'; .\\.venv\\Scripts\\python.exe scripts/run_wp1_audit.py --config configs/cpu_dev.yaml --output runs/wp1-audit",
         "exit_code": 0, "result": "passed; original audit retained, subsequent figure text clarified LFM Tukey versus OFDM window"}
    ],
}
commands = [
    [python, "-m", "pytest", "-q", "tests/test_qpsk.py", "tests/test_allocation.py", "tests/test_waveform.py", "tests/test_lfm.py", "--basetemp=runs/wp1-targeted-02", "-ra"],
    [python, "-m", "pytest", "-q", "--basetemp=runs/wp1-full-01", "-ra"],
    [python, "scripts/run_wp1_audit.py", "--config", "configs/cpu_dev.yaml", "--output", "runs/wp1-audit-final"],
    [python, "scripts/run_wp1_audit.py", "--config", "configs/cpu_dev.yaml", "--dtype", "complex64", "--output", "runs/wp1-audit-complex64"],
    [ruff_python, "-m", "ruff", "check", "src", "tests", "scripts/run_wp1_audit.py"],
    [ruff_python, "-m", "ruff", "format", "--check", "src", "tests", "scripts/run_wp1_audit.py"],
    [python, "-m", "mypy", "src", "scripts/run_wp1_audit.py"],
    [python, "-m", "pgvamp_ofdm", "inspect-config", "--config", "configs/cpu_dev.yaml"],
    [str(root / ".venv/Scripts/pgvamp-ofdm.exe"), "inspect-config", "--config", "configs/cpu_dev.yaml"],
    ["git", "diff", "--check"],
]
destination = Path(__file__).with_name("implementation-validation.json")
for command in commands:
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    receipt["commands"].append({"command": command, "exit_code": result.returncode,
                                "stdout": result.stdout, "stderr": result.stderr})
    destination.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(f"{result.returncode}: {' '.join(command)}", flush=True)
    print(result.stdout[:600], flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
paths = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
paths += [root / "scripts/run_wp1_audit.py", root / "docs/CODEX_ENGINEERING_SPEC.md"]
receipt["source_sha256"] = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
receipt["audits"] = {name: json.loads((root / "runs" / name / "summary.json").read_text())
                     for name in ("wp1-audit-final", "wp1-audit-complex64")
                     if (root / "runs" / name / "summary.json").exists()}
receipt["status"] = "passed" if all(c["exit_code"] == 0 for c in receipt["commands"]) else "failed"
receipt["figure_review"] = "Initial real PSD/correlation PNGs opened and visually inspected; final images need main/check independent review. Labels show spectral intervals, units, normalization, threshold and injected template start."
receipt["remaining_limits"] = ["No CUDA hardware; conditional hardware tests skipped", "WP2 channel/CP path validity/effective H and full-system smoke not executed", "Synchronization maximum is not earliest physical path; exact computed-score ties choose first, numerically split mathematical ties can choose either", "Extreme dynamic range losing positive prefix-window energy explicitly errors"]
destination.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
raise SystemExit(0 if receipt["status"] == "passed" else 1)
