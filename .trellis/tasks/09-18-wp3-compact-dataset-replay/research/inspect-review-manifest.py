"""Review saved compact metadata independently of the product manifest loader."""
import hashlib
import json
import sys
from pathlib import Path
import torch

prefix = sys.argv[1] if len(sys.argv) > 1 else 'runs/wp3-review'
path = Path(prefix + '-data/manifest.json')
m = json.loads(path.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert hashlib.sha256(json.dumps(m['resolved_config'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == m['config_sha256']
records = []
for s in m['shards']:
    p = path.parent / s['path']
    assert digest(p) == s['sha256']
    batch = torch.load(p, weights_only=True, map_location='cpu')
    assert len(batch) == s['frame_copies']
    assert all(r['split'] == s['split'] for r in batch)
    records.extend(batch)
counts = {}
frames, channels = {}, {}
for split in ('train', 'val', 'test'):
    subset = [r for r in records if r['split'] == split]
    frames[split] = {r['frame_id'] for r in subset}
    channels[split] = {r['channel_id'] for r in subset}
    counts[split] = {'independent_frames': len(frames[split]), 'independent_channels': len(channels[split]), 'frame_copies': len(subset), 'samples': 8 * len(subset)}
    assert counts[split] == m['counts'][split]
for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
    assert frames[a].isdisjoint(frames[b]) and channels[a].isdisjoint(channels[b])
test = [r for r in records if r['split'] == 'test']
for frame in frames['test']:
    pair = [r for r in test if r['frame_id'] == frame]
    assert {float(r['esn0_db'][0]) for r in pair} == {0, 10}
    for r in pair:
        assert bool((r['esn0_db'] == r['esn0_db'][0]).all())
        for key in ('data_bits', 'pilot_symbols', 'path_gain_complex', 'path_delay_s', 'path_epsilon'):
            assert torch.equal(pair[0][key], r[key])
    assert not torch.equal(pair[0]['noise_seed_real'], pair[1]['noise_seed_real'])
a = m['allocation']
assert len(a['data_grid_index']) == 400 and len(a['pilot_grid_index']) == 64
assert set(a['data_grid_index']).isdisjoint(a['pilot_grid_index'])
for relative, expected in m['environment']['package_source_hashes'].items():
    p = Path('src/pgvamp_ofdm') / relative
    assert digest(p) == expected, str(p)
result = {'status': 'passed', 'counts': counts, 'manifest_sha256': digest(path), 'package_source_sha256': m['environment']['package_source_sha256'], 'engineering_spec_sha256': m['environment']['engineering_spec_sha256'], 'audit_json_sha256': digest(Path(prefix + '-audit/audit.json')), 'dense_sha256': digest(Path(prefix + '-test.pt'))}
out = Path('.trellis/tasks/09-18-wp3-compact-dataset-replay/research/check-independent-manifest.json')
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
