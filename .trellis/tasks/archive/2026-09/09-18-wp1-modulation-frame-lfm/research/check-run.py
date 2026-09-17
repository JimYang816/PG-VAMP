"""Independent reviewer command receipts; preserve every command and output."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[4]
os.chdir(root)
python = str(root / '.venv/Scripts/python.exe')
env = dict(os.environ, MKL_THREADING_LAYER='TBB')
commands = [
    [python, '-m', 'pytest', '-q', 'tests/test_qpsk.py', 'tests/test_allocation.py', 'tests/test_waveform.py', 'tests/test_lfm.py', '--basetemp=runs/wp1-check-targeted-01', '-ra'],
    [python, '-m', 'pytest', '-q', '--basetemp=runs/wp1-check-full-01', '-ra'],
    [python, 'scripts/run_wp1_audit.py', '--config', 'configs/cpu_dev.yaml', '--output', 'runs/wp1-check-audit'],
    [python, 'scripts/run_wp1_audit.py', '--config', 'configs/cpu_dev.yaml', '--dtype', 'complex64', '--output', 'runs/wp1-check-audit-complex64'],
    ['C:/Software/Anaconda3/python.exe', '-m', 'ruff', 'check', 'src', 'tests', 'scripts/run_wp1_audit.py'],
    ['C:/Software/Anaconda3/python.exe', '-m', 'ruff', 'format', '--check', 'src', 'tests', 'scripts/run_wp1_audit.py'],
    [python, '-m', 'mypy', 'src', 'scripts/run_wp1_audit.py'],
    [python, '-m', 'pgvamp_ofdm', 'inspect-config', '--config', 'configs/cpu_dev.yaml'],
    [str(root / '.venv/Scripts/pgvamp-ofdm.exe'), 'inspect-config', '--config', 'configs/cpu_dev.yaml'],
    ['git', 'diff', '--check'],
]
receipt = {'timestamp': datetime.now(timezone.utc).isoformat(), 'environment': {'MKL_THREADING_LAYER':'TBB'}, 'commands': [],
           'pre_fix_observation': 'Matching constant recording/template float64=1e-100 and float32=1e-15 returned all-zero scores despite nonzero per-sample energies; energy product and numerator underflowed before additive epsilon. Fixed by explicit rejection. Allocation.validate skipped non-tensor fields and permitted non-int scalar metadata; fixed by validation.',
           'fix_regression': {'command': '.venv/Scripts/python.exe -m pytest -q tests/test_lfm.py tests/test_allocation.py --basetemp=runs/wp1-check-fixes-01 -ra', 'exit_code':0, 'stdout':'53 passed in 1.12s'}}
path = Path(__file__).with_name('check-validation.json')
for command in commands:
    p = subprocess.run(command, text=True, capture_output=True, env=env)
    receipt['commands'].append({'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(p.returncode, ' '.join(command), p.stdout[:300], flush=True)
paths = sorted(list((root/'src').rglob('*.py'))+list((root/'tests').glob('*.py'))+[root/'scripts/run_wp1_audit.py',root/'docs/CODEX_ENGINEERING_SPEC.md'])
receipt['source_sha256'] = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
receipt['status'] = 'passed' if all(x['exit_code']==0 for x in receipt['commands']) else 'failed'
path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
raise SystemExit(0 if receipt['status']=='passed' else 1)
