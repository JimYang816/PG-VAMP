"""Read persisted artifacts; independently check numerical values and deterministic replay."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from pgvamp_ofdm.config import load_config
from pgvamp_ofdm.utils.device import resolve_runtime
from pgvamp_ofdm.waveform.frame import build_frame

root = Path(__file__).resolve().parents[4]
results = {}
for name in ('wp1-check-audit', 'wp1-check-audit-complex64'):
    directory = root / 'runs' / name
    environment = json.loads((directory/'environment.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    hashes = json.loads((directory/'artifact_hashes.json').read_text())
    for relative, expected in hashes.items():
        assert hashlib.sha256((directory/relative).read_bytes()).hexdigest() == expected
    for relative, expected in environment['package_source_hashes'].items():
        assert hashlib.sha256((root/'src/pgvamp_ofdm'/relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root/'scripts/run_wp1_audit.py').read_bytes()).hexdigest() == environment['audit_script_sha256']
    assert hashlib.sha256((root/'docs/CODEX_ENGINEERING_SPEC.md').read_bytes()).hexdigest() == environment['engineering_spec_sha256']
    tensors = torch.load(directory/'waveforms.pt', weights_only=True, map_location='cpu')
    config = load_config(directory/'resolved_config.yaml')
    runtime = resolve_runtime(dtype=config.values['runtime']['dtype'])
    seeds = environment['seeds']
    for domain, seed in seeds.items():
        digest = hashlib.sha256(f"wp1-audit-v1:{environment['master_seed']}:{domain}".encode()).digest()
        assert seed == int.from_bytes(digest[:8], 'big') % (2**63)
    bits = torch.randint(2, (1,8,400,2), dtype=torch.uint8, generator=torch.Generator().manual_seed(seeds['bits']))
    assert torch.equal(bits, tensors['data_bits'])
    pilot_classes = torch.randint(4, (1,8,64), generator=torch.Generator().manual_seed(seeds['pilots']))
    pilot_real = 1 - 2 * (pilot_classes // 2).to(runtime.real_dtype)
    pilot_imag = 1 - 2 * (pilot_classes % 2).to(runtime.real_dtype)
    pilots = torch.complex(pilot_real, pilot_imag) / math.sqrt(2)
    assert torch.equal(pilots, tensors['pilot_symbols'])
    # Consuming arbitrary data samples cannot change a separately seeded pilot stream.
    torch.randint(2, (10000,), generator=torch.Generator().manual_seed(seeds['bits']+1))
    replay_classes = torch.randint(4, (1,8,64), generator=torch.Generator().manual_seed(seeds['pilots']))
    assert torch.equal(replay_classes, pilot_classes)
    before = torch.random.get_rng_state()
    frame = build_frame(bits, pilots, config, runtime)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(frame.real, tensors['real']) and torch.equal(frame.analytic, tensors['analytic'])
    offset = int(torch.randint(1921, (), generator=torch.Generator().manual_seed(seeds['arrival_offset'])))
    assert offset == tensors['arrival_offset_samples'] == summary['arrival_offset_samples']
    assert tensors['recording_real'].shape[-1] == 91776 + offset
    assert torch.equal(tensors['recording_real'][:,offset:],tensors['real'])
    assert torch.count_nonzero(tensors['recording_real'][:,:offset]) == 0
    real = tensors['real'].numpy()[0].astype(np.float64)
    grid = tensors['grid'].numpy()[0]
    errors = []
    bit_errors = 0
    for m in range(8):
        start = 9856 + m*10240
        recovered = np.fft.fft(real[start:start+8192], norm='ortho')[np.arange(1792,2304)]*math.sqrt(2)
        recovered *= np.exp(-2j*np.pi*24000*start/96000)
        atol = 1e-9 if runtime.complex_dtype == torch.complex128 else 2e-5
        rtol = 1e-8 if runtime.complex_dtype == torch.complex128 else 2e-4
        np.testing.assert_allclose(recovered, grid[m], atol=atol, rtol=rtol)
        errors.append(float(np.max(np.abs(recovered-grid[m]))))
        data = recovered[tensors['allocation']['data_grid_index'].numpy()]
        signs = np.stack((data.real<0,data.imag<0),axis=-1)
        bit_errors += int(np.count_nonzero(signs != bits.numpy()[0,m]))
    assert bit_errors == 0
    psd = np.load(directory/'psd.npz')
    segments = {'frame':real,'ofdm_useful_block0':real[9856:18048], 'lfm':tensors['lfm_real'].numpy().astype(np.float64)}
    psd_errors = {}
    for segment, samples in segments.items():
        density = psd[segment+'_density']
        frequency = psd[segment+'_frequency_hz']
        expected = np.abs(np.fft.fft(samples,n=len(density)))**2/(96000*len(samples))
        np.testing.assert_array_equal(density, expected)
        np.testing.assert_array_equal(frequency,np.fft.fftfreq(len(density),1/96000))
        error = abs(density.sum()*96000/len(density)-np.mean(samples**2))
        assert error < 1e-14
        outside = (np.abs(frequency)<21000)|(np.abs(frequency)>27000)
        oob = density[outside].sum()/density.sum()
        assert abs(oob-summary['psd'][segment]['outside_design_band_energy_ratio']) < 1e-15
        psd_errors[segment] = {'integration_absolute_error':error,'outside_band_fraction':float(oob)}
    correlations = np.load(directory/'correlation.npz')
    direct_errors = {}
    for kind in ('real','analytic'):
        recording = tensors['recording_'+kind].numpy()[0].astype(np.float64 if kind=='real' else np.complex128)
        template = tensors['lfm_'+kind].numpy().astype(np.float64 if kind=='real' else np.complex128)
        scores = correlations[kind+'_scores'][0]
        assert int(np.argmax(scores)) == offset+1920 == summary['sync'][('peak_start' if kind=='real' else 'analytic_peak_start')]
        assert np.all(np.isfinite(scores))
        lags = [0, offset+1919, offset+1920, offset+1921,5000,8000,len(scores)-1]
        errors_at_lags = []
        for lag in lags:
            window = recording[lag:lag+len(template)]
            expected = abs(np.vdot(template,window))**2/(np.vdot(window,window).real*np.vdot(template,template).real+summary['sync']['epsilon_sync'])
            errors_at_lags.append(float(abs(expected-scores[lag])))
        assert max(errors_at_lags) < (1e-9 if runtime.complex_dtype==torch.complex128 else 2e-5)
        direct_errors[kind] = max(errors_at_lags)
    results[name] = {'artifact_hashes_checked':len(hashes),'package_source_hashes_checked':len(environment['package_source_hashes']), 'replay_exact':True,'bit_errors':bit_errors,'numpy_recovery_max_abs':max(errors),'psd':psd_errors,'direct_correlation_max_abs':direct_errors,'noise_real_max':max(summary['noise_fixture']['real_peaks']),'noise_complex_max':max(summary['noise_fixture']['complex_peaks'])}
manifest = json.loads((root/'.trellis/tasks/archive/2026-09/09-17-wp0-foundation-references/research/final-source-manifest.json').read_text())
for relative, details in manifest['files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == details['sha256']
results['wp0_unchanged_files'] = len(manifest['files'])
results['status'] = 'passed'
Path(__file__).with_name('check-artifacts.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
