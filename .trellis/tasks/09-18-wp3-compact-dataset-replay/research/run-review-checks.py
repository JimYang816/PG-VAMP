"""Reproducible final WP3 review commands with source fingerprints and receipts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
evidence = root / '.trellis/tasks/09-18-wp3-compact-dataset-replay/research'
os.environ['MKL_THREADING_LAYER'] = 'TBB'
py = str(root / '.venv/Scripts/python.exe')
ruff = 'C:/Software/Anaconda3/Scripts/ruff.exe'
source_hashes = {
    p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ('src', 'tests', 'scripts', 'configs')
    for p in sorted(Path(folder).rglob('*'))
    if p.is_file() and p.suffix in ('.py', '.yaml')
}
commands = [
    [ruff, 'check', 'src', 'tests', 'scripts'],
    [ruff, 'format', '--check', 'src', 'tests', 'scripts'],
    [py, '-m', 'mypy', 'src'],
    [py, '-m', 'pytest', '-q', '--basetemp=runs/pytest-wp3-review-final'],
    [py, '-m', 'pgvamp_ofdm', 'simulate', '--config', 'configs/wp3_smoke.yaml', '--output', 'runs/wp3-review-data'],
    [py, '-m', 'pgvamp_ofdm', 'audit-data', '--manifest', 'runs/wp3-review-data/manifest.json', '--waveform-frames', '2', '--output', 'runs/wp3-review-audit'],
    [py, '-m', 'pgvamp_ofdm', 'materialize', '--manifest', 'runs/wp3-review-data/manifest.json', '--split', 'test', '--output', 'runs/wp3-review-test.pt', '--max-output-bytes', '1073741824'],
    [py, 'scripts/verify_wp3_artifacts.py', '--manifest', 'runs/wp3-review-data/manifest.json', '--audit', 'runs/wp3-review-audit', '--materialized', 'runs/wp3-review-test.pt', '--output', 'runs/wp3-review-verification.json'],
    [py, '-m', 'pgvamp_ofdm', 'inspect-config', '--config', 'configs/cpu_dev.yaml', '--output', 'runs/wp3-review-inspect'],
]
receipts = {'MKL_THREADING_LAYER': 'TBB', 'source_hashes': source_hashes, 'commands': []}
receipt_name = 'check-command-receipts.json'
if '--final' in sys.argv:
    commands = [[part.replace('wp3-review', 'wp3-review-final2') for part in command] for command in commands]
    receipt_name = 'check-final-command-receipts.json'
if '--followup' in sys.argv:
    commands = commands[:3] + commands[7:] + [[py, str(evidence / 'inspect-review-manifest.py')]]
    receipt_name = 'check-followup-command-receipts.json'
for command in commands:
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process = subprocess.run(command, text=True, capture_output=True)
    receipts['commands'].append({
        'argv': command, 'started_utc': start,
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr,
    })
    (evidence / receipt_name).write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'argv': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}), flush=True)
    if process.returncode:
        sys.exit(process.returncode)
