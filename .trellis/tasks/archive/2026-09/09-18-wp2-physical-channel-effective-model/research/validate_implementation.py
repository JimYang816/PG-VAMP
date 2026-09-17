"""Capture actual WP2 implementation commands and final source fingerprints."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[4]
research = Path(__file__).resolve().parent
python = str(root / ".venv/Scripts/python.exe")
env = dict(os.environ, MKL_THREADING_LAYER="TBB")
commands = [
    [python, "-m", "pytest", "-q", "--basetemp=runs/wp2-final-tests-01", "-ra"],
    ["C:/Software/Anaconda3/Scripts/ruff.exe", "check", "src", "tests",
     "scripts/run_wp1_audit.py", "scripts/run_wp2_audit.py"],
    ["C:/Software/Anaconda3/Scripts/ruff.exe", "format", "--check", "src", "tests",
     "scripts/run_wp1_audit.py", "scripts/run_wp2_audit.py"],
    [python, "-m", "mypy", "src"],
    [python, "scripts/run_wp2_audit.py", "--config", "configs/cpu_dev.yaml",
     "--output", "runs/wp2-audit-final-01"],
    ["git", "diff", "--check"],
]
results = []
for command in commands:
    run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    results.append({"command": command, "exit_code": run.returncode,
                    "stdout": run.stdout, "stderr": run.stderr})
    print(json.dumps({"command": command, "exit_code": run.returncode,
                      "stdout_summary": run.stdout[-500:]}, ensure_ascii=True), flush=True)
paths = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
paths += [root / "scripts/run_wp2_audit.py", root / "docs/CODEX_ENGINEERING_SPEC.md",
          root / "pyproject.toml", root / "configs/cpu_dev.yaml"]
hashes = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(paths)}
receipt = {"environment_override": {"MKL_THREADING_LAYER": "TBB"}, "results": results,
           "source_hashes": hashes, "passed": all(r["exit_code"] == 0 for r in results),
           "earlier_attempts": [
               {"command": ".venv/Scripts/python.exe -m ruff check ...", "outcome": "launcher unavailable",
                "error": "FileNotFoundError: D:/SCI/code/PG-VAMP/.venv/Scripts/ruff.exe"},
               {"command": "ruff check .", "outcome": "192 existing errors under .codex/.trellis; product scope clean"},
               {"command": ".venv/Scripts/python.exe -m pytest -q", "outcome": "158 passed, 3 skipped, 31 setup errors",
                "error": "PermissionError [WinError 5]: C:/Users/jimya/AppData/Local/Temp/pytest-of-jimya",
                "resolution": "fresh workspace --basetemp; no test changed to hide setup errors"},
               {"command": ".venv/Scripts/python.exe scripts/run_wp2_audit.py --config configs/cpu_dev.yaml --output runs/wp2-audit",
                "outcome": "passed; earlier audit retained"}]}
(research / "implementation-validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
audit = json.loads((root / "runs/wp2-audit-final-01/audit.json").read_text(encoding="utf-8"))
(research / "physical-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
if not receipt["passed"]:
    raise SystemExit(1)
