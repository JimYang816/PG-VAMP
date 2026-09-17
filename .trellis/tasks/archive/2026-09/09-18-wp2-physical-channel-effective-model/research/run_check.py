"""Independent WP2 review receipt and safe artifact cross-check."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path.cwd()
RESEARCH = ROOT / '.trellis/tasks/09-18-wp2-physical-channel-effective-model/research'
AUDIT = ROOT / 'runs/wp2-check-audit-final-01'
PYTHON = '.venv/Scripts/python.exe'
RUFF = 'C:/Software/Anaconda3/Scripts/ruff.exe'
os.environ['MKL_THREADING_LAYER'] = 'TBB'
commands = [
    [PYTHON, '-m', 'pytest', '-q', 'tests/test_wp2.py', '--basetemp', 'runs/wp2-check-target-final-01'],
    [PYTHON, '-m', 'pytest', '-q', '--basetemp', 'runs/wp2-check-full-final-01'],
    [RUFF, 'check', 'src', 'tests', 'scripts/run_wp1_audit.py', 'scripts/run_wp2_audit.py'],
    [RUFF, 'format', '--check', 'src', 'tests', 'scripts/run_wp1_audit.py', 'scripts/run_wp2_audit.py'],
    [PYTHON, '-m', 'mypy', 'src'],
    [PYTHON, 'scripts/run_wp2_audit.py', '--config', 'configs/cpu_dev.yaml', '--output', str(AUDIT)],
]
receipt = {'environment': {'MKL_THREADING_LAYER': 'TBB'}, 'commands': []}
for command in commands:
    started = time.time()
    p = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    receipt['commands'].append(dict(command=command, started=started, status=p.returncode,
                                    stdout=p.stdout, stderr=p.stderr))
    (RESEARCH / 'check-final-validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(command, p.returncode, p.stdout[:200], flush=True)
    if p.returncode:
        raise RuntimeError('verification failed')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = json.loads((AUDIT / 'audit.json').read_text())
assert report['resolved_config_sha256'] == sha(AUDIT / 'resolved_config.yaml')
q = np.arange(-256, 256)
# Source section 3: midpoint spacing over each positive/negative active half.
positive_pilots = np.array([1 + math.floor((j + 0.5) * 232 / 32) for j in range(32)])
pilot_q = np.concatenate((-positive_pilots[::-1], positive_pilots))
pilot_idx = pilot_q + 256
active_q = np.concatenate((np.arange(-232, 0), np.arange(1, 233)))
data_q = np.setdiff1d(active_q, pilot_q)
data_idx = data_q + 256
crosschecks = []
for fr in report['frames']:
    artifact = AUDIT / fr['tensor_artifact']
    assert sha(artifact) == fr['tensor_sha256']
    record = torch.load(artifact, map_location='cpu', weights_only=True)
    grid = record['grid'].numpy()[0]
    assert len(data_idx) == 400 and len(pilot_idx) == 64
    assert np.all(np.abs(grid[:, np.setdiff1d(np.arange(512), active_q + 256)]) == 0)
    for window, saved in zip(record['windows'], fr['windows'], strict=True):
        h = window['h_grid'].numpy()
        wave = window['iq_waveform'].numpy()[0]
        observed = np.fft.fft(wave, norm='ortho')[q % 8192]
        block, offset = window['block'], window['offset_s']
        prediction = h @ grid[block]
        relative = np.linalg.norm(observed - prediction) / np.linalg.norm(observed)
        assert relative < 1e-9
        np.testing.assert_allclose(observed, window['observed_grid'].numpy(), atol=1e-12, rtol=1e-12)
        hdd = h[np.ix_(data_idx, data_idx)]
        leakage = h[np.ix_(data_idx, pilot_idx)] @ grid[block, pilot_idx]
        clean = observed[data_idx] - leakage
        np.testing.assert_allclose(clean, window['y_clean'].numpy(), atol=1e-12, rtol=1e-12)
        noise = np.fft.fft(window['noise'].numpy(), norm='ortho')[q % 8192][data_idx]
        np.testing.assert_allclose(window['y_noisy'].numpy(), hdd @ grid[block, data_idx] + noise,
                                   atol=1e-9, rtol=1e-9)
        snr = 10 * math.log10(np.sum(np.abs(hdd)**2) / (400 * saved['sigma2']))
        assert abs(snr - saved['snr_rx_db']) < 1e-12
        assert abs(np.sum(np.abs(leakage)**2) - saved['pilot_leakage_energy']) < 1e-12
        indices = np.array([0, 1, 4095, 8191])
        start = (9856 + block * 10240) / 96000
        receive_times = start + offset + indices / 96000
        direct = np.zeros(4, dtype=np.complex128)
        for gain, delay, epsilon in zip(record['gain'].numpy(), record['delay_s'].numpy(), record['epsilon'].numpy()):
            transmit_times = (1 + epsilon) * receive_times - delay
            local = transmit_times - start
            assert np.all(local >= -2048 / 96000) and np.all(local < 8192 / 96000)
            direct += gain * (grid[block] @ np.exp(2j*np.pi*q[:, None]*(96000/8192)*local) / np.sqrt(8192)) * np.exp(2j*np.pi*24000*transmit_times)
        direct *= np.exp(-2j*np.pi*24000*receive_times)
        np.testing.assert_allclose(direct, wave[indices], atol=1e-10, rtol=1e-9)
        actual64 = torch.fft.fft(window['iq_waveform'].to(torch.complex64), norm='ortho')[0, torch.tensor(q % 8192)]
        pred64 = window['h_grid'].to(torch.complex64) @ record['grid'][0, block].to(torch.complex64)
        relative64 = float(torch.linalg.vector_norm(actual64 - pred64) / torch.linalg.vector_norm(actual64))
        assert relative64 < 2e-6
        crosschecks.append(dict(frame=fr['frame_index'], block=block, offset=offset,
                                error=relative, error64=relative64, direct_max_abs=float(np.max(np.abs(direct-wave[indices])))))
paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'tests').rglob('*.py')) + [ROOT/'scripts/run_wp2_audit.py', ROOT/'docs/CODEX_ENGINEERING_SPEC.md']
result = dict(status='passed', safe_load='weights_only=True, map_location=cpu', windows=crosschecks,
              hashes={str(p.relative_to(ROOT)): sha(p) for p in paths}, audit_sha256=sha(AUDIT/'audit.json'))
(RESEARCH/'check-artifact-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('Independent artifact checks passed:', len(crosschecks), flush=True)
