"""Independent final check commands, receipts and unchanged-authority audit."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

research = Path(__file__).parent
root = Path("runs/wp7-cli-review-v2")
python = str(Path(sys.executable).resolve())
ruff = "C:/Software/Anaconda3/Scripts/ruff.exe"
commands = [
    [ruff, "check", "src", "tests"],
    [ruff, "format", "--check", "src", "tests"],
    [python, "-m", "mypy", "src"],
    [python, "-m", "pytest", "-q", "--basetemp", "runs/wp7-review-full-v2"],
    [python, str(research / "coordinator-audit.py"), str(root / "evaluation1"),
     str(root / "evaluation2"), "--output", str(research / "check-artifact-audit.json")],
]
receipts = []
for argv in commands:
    start = time.perf_counter()
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ,
                            "MKL_THREADING_LAYER": "TBB", "PYTHONIOENCODING": "utf-8"})
    receipts.append(dict(argv=argv, cwd=str(Path.cwd()), exit_code=result.returncode,
                         elapsed_seconds=time.perf_counter()-start,
                         stdout=result.stdout, stderr=result.stderr))
    (research / "check-receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    print(argv, result.returncode, result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

baseline = json.loads((research / "implementation-baseline.json").read_text())
authority = {name: digest(Path(name)) for name in baseline["authority_and_oracles"]}
assert authority == baseline["authority_and_oracles"], "authority/oracle changed"
fingerprints = {
    "authority_and_oracles": authority,
    "source_and_tests": {str(p): digest(p) for folder in ("src", "tests")
                         for p in sorted(Path(folder).rglob("*.py"))},
    "cli_artifacts": {str(p): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
}
(research / "check-fingerprints.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")
print("unchanged authority/oracles and artifact fingerprints recorded", flush=True)
