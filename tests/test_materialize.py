import copy
from pathlib import Path

import pytest
import torch

from pgvamp_ofdm.data.dataset import EffectiveDataset
from pgvamp_ofdm.data.generate import generate_dataset
from pgvamp_ofdm.data.materialize import (
    estimate_size,
    load_materialized,
    materialize,
    validate_materialized,
)


@pytest.mark.parametrize("dtype,c,r", [("complex128", 16, 8), ("complex64", 8, 4)])
def test_capacity(dtype, c, r):
    result = estimate_size(8192, 400, dtype)
    assert result["matrix_payload_bytes"] == 8192 * 400 * 400 * c
    assert result["other_tensor_payload_bytes"] == 8192 * (800 * c + r + 800)
    if c == 16:
        assert result["matrix_payload_bytes"] == 20_971_520_000
    assert (
        estimate_size(1, 400, dtype, labeled=False)["tensor_payload_bytes"]
        == (400 * 400 + 400) * c + r
    )


def test_guard_precedes_reconstruction(data_manifest, tmp_path, monkeypatch, data_config):
    def unexpected(*args):
        raise AssertionError("dense reconstruction before budget guard")

    monkeypatch.setattr(EffectiveDataset, "__getitem__", unexpected)
    with pytest.raises(ValueError, match="allow-large-output"):
        materialize(data_manifest, "test", tmp_path / "dense.pt", max_output_bytes=1)
    assert not (tmp_path / "dense.pt").exists()
    data_config.values["data"]["storage"] = "materialized"
    data_config.values["data"]["max_output_bytes"] = 1
    with pytest.raises(ValueError, match="allow-large-output"):
        generate_dataset(data_config, tmp_path / "large")
    assert not (tmp_path / "large").exists()


def test_multifile_budget_includes_each_file_reserve(data_config, tmp_path, monkeypatch):
    import pgvamp_ofdm.data.generate as module

    # The old combined tensor estimate reserved overhead for only one of three files.
    data_config.values["data"]["storage"] = "materialized"
    data_config.values["data"]["max_output_bytes"] = estimate_size(48, 400, "complex128")[
        "estimated_output_bytes"
    ]

    def unexpected(*args):
        raise AssertionError("record generation before aggregate budget guard")

    monkeypatch.setattr(module, "generate_records", unexpected)
    with pytest.raises(ValueError, match="allow-large-output"):
        generate_dataset(data_config, tmp_path / "three-files")
    assert not (tmp_path / "three-files").exists()


def test_export_reserves_full_split_metadata(data_manifest, tmp_path):
    result = materialize(data_manifest, "val", tmp_path / "measured.pt")
    assert result["actual_file_bytes"] < result["estimated_output_bytes"]
    assert (
        result["serialization_metadata_reserve_bytes"]
        > estimate_size(8, 400, "complex128")["serialization_metadata_reserve_bytes"]
    )


@pytest.fixture
def dense(data_manifest, tmp_path):
    output = tmp_path / "dense.pt"
    materialize(data_manifest, "val", output, max_output_bytes=1, allow_large_output=True)
    return load_materialized(output, purpose="evaluation")


def test_roundtrip_and_unlabeled(dense, data_manifest, tmp_path):
    dataset = EffectiveDataset(data_manifest, "val")
    for index in range(len(dataset)):
        sample = dataset[index]
        assert dense["metadata"]["sample_ids"][index] == sample["sample_id"]
        for key in ("H", "y", "sigma2", "x", "bits"):
            assert torch.equal(dense[key][index], sample[key])
    path = tmp_path / "inference.pt"
    materialize(data_manifest, "val", path, labeled=False)
    inference = load_materialized(path)
    assert not {"x", "bits"}.intersection(inference)
    for purpose in ("train", "evaluation"):
        with pytest.raises(ValueError, match="requires x and bits"):
            load_materialized(path, purpose=purpose)


@pytest.mark.parametrize(
    "case",
    [
        "square",
        "dtype",
        "finite",
        "variance",
        "qpsk",
        "bits",
        "label_consistency",
        "count",
        "mapping",
        "config",
        "split",
        "schema",
        "source_hash",
        "sample_block",
        "sample_snr",
    ],
)
def test_bad_dense(dense, case):
    value = copy.deepcopy(dense)
    if case == "square":
        value["H"] = value["H"][:, :, :-1]
    elif case == "dtype":
        value["sigma2"] = value["sigma2"].float()
    elif case == "finite":
        value["y"][0, 0] = float("inf")
    elif case == "variance":
        value["sigma2"][0] = 0
    elif case == "qpsk":
        value["x"][0, 0] = 1
    elif case == "bits":
        value["bits"][0, 0, 0] = 2
    elif case == "label_consistency":
        value["bits"][0, 0, 0] ^= 1
    elif case == "count":
        value["metadata"]["sample_ids"].pop()
    elif case == "mapping":
        value["metadata"]["allocation"]["data_grid_index"][0] += 1
    elif case == "config":
        value["metadata"]["config_sha256"] = "wrong"
    elif case == "split":
        value["metadata"]["split_table"]["train"].extend(value["metadata"]["split_table"]["val"])
    elif case == "source_hash":
        value["metadata"]["source_manifest_sha256"] = "z" * 64
    elif case == "sample_block":
        value["metadata"]["sample_ids"][0] = value["metadata"]["frame_ids"][0] + ":continuous:8"
    elif case == "sample_snr":
        value["metadata"]["sample_ids"][0] = value["metadata"]["frame_ids"][0] + ":invalid:0"
    else:
        value["schema_version"] = True
    with pytest.raises(ValueError):
        validate_materialized(value)


def test_unsafe_pickle_is_rejected(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"object": Path("not-a-basic-type")}, path)
    with pytest.raises(ValueError, match="unsupported tensor artifact"):
        load_materialized(path)


def test_materialized_generation_failure_has_no_final_manifest(data_config, tmp_path, monkeypatch):
    import pgvamp_ofdm.data.materialize as module

    data_config.values["data"]["storage"] = "materialized"

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(module, "materialize", fail)
    with pytest.raises(OSError, match="disk failure"):
        generate_dataset(data_config, tmp_path / "failed")
    assert not (tmp_path / "failed" / "manifest.json").exists()
    assert (tmp_path / "failed" / "manifest.pending.json").exists()
