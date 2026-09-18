import copy
import json
import shutil

import pytest
import torch

from pgvamp_ofdm.data.manifest import file_hash, load_manifest, validate_lineage
from pgvamp_ofdm.data.records import validate_record


def test_roundtrip_counts_and_pairing(data_manifest):
    manifest, config, records = load_manifest(data_manifest)
    assert manifest["counts"]["test"] == {
        "independent_frames": 2,
        "independent_channels": 2,
        "frame_copies": 4,
        "samples": 32,
    }
    assert config.values["waveform"]["n_fft_wave"] == 8192
    assert manifest["environment"]["engineering_spec_sha256"]
    assert manifest["distribution"]["label"] == "development_only"
    frames = [
        {r["frame_id"] for r in records if r["split"] == split}
        for split in ("train", "val", "test")
    ]
    assert not (frames[0] & frames[1] or frames[0] & frames[2] or frames[1] & frames[2])
    test = [r for r in records if r["split"] == "test"]
    assert torch.equal(test[0]["data_bits"], test[1]["data_bits"])
    assert not torch.equal(test[0]["noise_seed_real"], test[1]["noise_seed_real"])
    for shard in manifest["shards"]:
        assert file_hash(data_manifest.parent / shard["path"]) == shard["sha256"]
        saved = torch.load(data_manifest.parent / shard["path"], weights_only=True)
        assert not any("H" in r for r in saved)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 3),
        ("schema_version", True),
        ("waveform_config_hash", "wrong"),
        ("cp_valid", False),
        ("path_count", 99),
        ("arrival_offset_samples", -1),
        ("data_bits", torch.full((8, 400, 2), 2, dtype=torch.uint8)),
        ("pilot_symbols", torch.ones((8, 64), dtype=torch.complex128)),
        ("esn0_db", torch.full((8,), float("nan"), dtype=torch.float64)),
        ("noise_seed_real", torch.zeros(8, dtype=torch.int64)),
        ("snr_copy", "invalid"),
        ("frame_id", "z" * 64),
        ("channel_id", "a" * 10000),
    ],
)
def test_record_rejects_bad_fields(data_manifest, key, value):
    _, config, records = load_manifest(data_manifest)
    record = copy.deepcopy(records[0])
    record[key] = value
    with pytest.raises(ValueError):
        validate_record(record, config)


def test_cp_and_leakage_rejected(data_manifest):
    _, config, records = load_manifest(data_manifest)
    broken = copy.deepcopy(records[0])
    broken["path_delay_s"].fill_(1)
    with pytest.raises(ValueError, match="CP support"):
        validate_record(broken, config)
    records[1]["channel_id"] = records[0]["channel_id"]
    with pytest.raises(ValueError, match="leakage"):
        validate_lineage(records, config)


@pytest.mark.parametrize(
    "mutation", ["checksum", "schema", "hash", "index", "counts", "path", "duplicate"]
)
def test_manifest_tampering(data_manifest, tmp_path, mutation):
    target = tmp_path / "copy"
    shutil.copytree(data_manifest.parent, target)
    path = target / "manifest.json"
    value = json.loads(path.read_text())
    if mutation == "checksum":
        (target / value["shards"][0]["path"]).write_bytes(b"corrupt")
    elif mutation == "schema":
        value["schema_version"] = 99
    elif mutation == "hash":
        value["resolved_config"]["seed"] += 1
    elif mutation == "index":
        value["allocation"]["data_grid_index"][0] += 1
    elif mutation == "counts":
        value["counts"]["train"]["samples"] += 1
    elif mutation == "path":
        value["shards"][0]["path"] = "../outside.pt"
    else:
        value["shards"].append(value["shards"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_manifest(path)
