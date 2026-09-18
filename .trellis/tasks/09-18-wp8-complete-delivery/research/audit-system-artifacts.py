"""WP8 independent NumPy audit, reused from the archived WP6 verification."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("runs/wp8-training-acceptance")
OUT = Path(".trellis/tasks/09-18-wp8-complete-delivery/research")
torch.set_num_threads(4)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pt(path):
    return torch.load(path, map_location="cpu", weights_only=True)


system = ROOT / "system"
report = json.loads((system / "smoke.json").read_text())
assert [report[k] for k in ("n_grid", "n_data", "n_fft_wave", "cp_samples",
                            "blocks_per_frame", "depth", "updates")] == [512, 400, 8192, 2048, 8, 8, 2]
saved = pt(system / "shared_predictions.pt")
digest = hashlib.sha256()
for key in ("H", "sigma2", "y"):
    tensor = saved[key]
    for text in (key, str(tensor.dtype), str(tuple(tensor.shape))):
        digest.update(text.encode())
    digest.update(tensor.numpy().tobytes())
shared_hash = digest.hexdigest()
assert shared_hash == saved["shared_input_hash"]
assert len(saved["sample_ids"]) == len(set(saved["sample_ids"])) == 8
assert len(set(saved["frame_ids"])) == 1
assert sorted(int(s.split(":")[-1]) for s in saved["sample_ids"]) == list(range(8))
truth = saved["target_bits"].numpy()
recounts = {}
for name in ("mmse", "vamp", "pg_vamp"):
    assert report["counts"][name]["input_hash"] == shared_hash
    errors = saved[name]["bits_hat"].numpy() != truth
    symbols = errors.any(axis=-1)
    blocks = symbols.any(axis=-1)
    current = {
        "bit_errors": int(errors.sum()), "bits": int(errors.size),
        "symbol_errors": int(symbols.sum()), "symbols": int(symbols.size),
        "block_errors": int(blocks.sum()), "blocks": 8,
        "frame_errors": int(blocks.any()), "frames": 1, "incomplete_frames": 0,
        "squared_error_energy": float(np.sum(np.abs(saved[name]["x_soft"].numpy() - saved["x"].numpy()) ** 2)),
        "target_energy": float(np.sum(np.abs(saved["x"].numpy()) ** 2)),
    }
    for key, value in current.items():
        np.testing.assert_allclose(value, report["counts"][name][key], rtol=1e-14, atol=1e-12)
    recounts[name] = current
unlabeled = pt(system / "unlabeled.pt")
inference = pt(system / "inference.pt")
assert "x" not in unlabeled and "bits" not in unlabeled
assert inference["metadata"]["sample_ids"] == saved["sample_ids"]
np.testing.assert_allclose(inference["x_soft"].numpy(), saved["pg_vamp"]["x_soft"].numpy(), atol=1e-9, rtol=1e-8)
assert inference["input_sha256"] == sha(system / "unlabeled.pt")
assert inference["checkpoint_sha256"] == sha(system / "training/last.pt") == report["checkpoint_sha256"]
checkpoint = pt(system / "training/last.pt")
assert checkpoint["step"] == 2
assert sum(t.numel() for t in checkpoint["model_state_dict"].values()) == 16
events = [json.loads(s) for s in (system / "training/training.jsonl").read_text().splitlines()]
assert [e["step"] for e in events if e["event"] == "update"] == [1, 2]
audit = json.loads((system / "audit/audit.json").read_text())
assert audit["source_manifest_sha256"] == sha(system / "data/manifest.json")
maximum = 0.0
for entry in audit["frames"]:
    path = system / "audit" / entry["artifact"]
    assert sha(path) == entry["sha256"]
    frame = pt(path)
    epsilon = frame["record"]["path_epsilon"].numpy()
    assert len(np.unique(epsilon[epsilon != 0])) >= 2
    assert len(frame["windows"]) == 8
    allocation = audit["allocation"]
    di, pi = allocation["data_grid_index"], allocation["pilot_grid_index"]
    for window in frame["windows"]:
        start, end = window["recording_span"]
        wave = frame["received_iq"][start:end].numpy()
        assert wave.shape == (8192,)
        observed = np.fft.fft(wave, norm="ortho")[allocation["baseband_fft_index"]]
        h = window["h_grid"].numpy()
        grid = frame["grid"][window["block"]].numpy()
        relative = float(np.linalg.norm(observed - h @ grid) / np.linalg.norm(observed))
        maximum = max(maximum, relative)
        assert relative < 1e-9
        noisy = np.fft.fft(wave + window["noise"].numpy(), norm="ortho")[allocation["baseband_fft_index"]]
        cancelled = noisy[di] - h[np.ix_(di, pi)] @ grid[pi]
        np.testing.assert_allclose(cancelled, window["y"].numpy(), atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(h[np.ix_(di, di)], window["H"].numpy())
receipts = json.loads((ROOT / "receipts.json").read_text())
assert len(receipts) == 11 and all(r["exit_code"] == 0 for r in receipts)
result = {
    "status": "passed", "receipt_count": 11, "dimension_contract": [512, 400, 8192, 2048, 8, 8, 2],
    "shared_input_sha256": shared_hash, "recomputed_counts": recounts,
    "numpy_fft_max_relative_error": maximum,
    "artifacts_sha256": {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob("*") if p.is_file()},
    "source_sha256": {str(p): sha(p) for p in [*Path("src").rglob("*.py"), Path("docs/CODEX_ENGINEERING_SPEC.md")]},
    "environment": checkpoint["environment"],
}
(OUT / "system-artifact-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("artifacts_sha256", "source_sha256", "environment")}, indent=2))
