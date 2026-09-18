import json
import subprocess
import sys

import pytest
import torch

from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.generate import generate_dataset
from pgvamp_ofdm.data.records import tensor_hash


def test_cache_order_block_identity_and_consumers(data_manifest):
    reference = EffectiveDataset(data_manifest, "test", cache_entries=0)
    cached = EffectiveDataset(data_manifest, "test", cache_entries=1)
    # First test frame is static; second scenario starts at sample 16.
    expected = reference[16]
    assert not torch.equal(expected["H"], reference[17]["H"])
    hashes = [tensor_hash(EffectiveDataset(data_manifest, "test")[16]) for _ in range(3)]
    assert hashes == [tensor_hash(expected)] * 3
    cached[16]["H"].zero_()
    cached[16]["y"].zero_()
    cached[17]
    assert tensor_hash(cached[16]) == tensor_hash(expected)
    assert not any(expected[k].requires_grad for k in ("H", "y", "sigma2", "x"))
    projected = detection_inputs(expected)
    expected["x"].zero_()
    expected["bits"].zero_()
    assert set(projected) == {"H", "y", "sigma2"}
    assert all(torch.equal(projected[k], detection_inputs(expected)[k]) for k in projected)


def test_cross_process_replay(data_manifest):
    code = (
        "import json,sys; from pgvamp_ofdm.data.dataset import EffectiveDataset; "
        "from pgvamp_ofdm.data.records import tensor_hash; "
        "print(json.dumps(tensor_hash(EffectiveDataset(sys.argv[1], 'test')[16])))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(data_manifest)], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == tensor_hash(EffectiveDataset(data_manifest, "test")[16])


def test_waveform_backend_and_complex64(data_config, data_manifest, tmp_path):
    fast = EffectiveDataset(data_manifest, "test")[16]
    data_config.values["data"]["backend"] = "waveform_reference"
    wave_manifest = generate_dataset(data_config, tmp_path / "wave")
    wave = EffectiveDataset(wave_manifest, "test")[16]
    assert fast["sample_id"] == wave["sample_id"]
    assert torch.equal(fast["H"], wave["H"])
    torch.testing.assert_close(fast["y"], wave["y"], atol=1e-9, rtol=1e-8)
    data_config.values["data"]["backend"] = "effective_fast"
    data_config.values["runtime"]["dtype"] = "complex64"
    small = EffectiveDataset(generate_dataset(data_config, tmp_path / "single"), "test")[16]
    assert small["H"].dtype == torch.complex64 and small["sigma2"].dtype == torch.float32
    torch.testing.assert_close(small["H"], fast["H"].to(torch.complex64), atol=0, rtol=0)
    assert bool(torch.isfinite(small["y"]).all())
    data_config.values["data"]["backend"] = "waveform_reference"
    single_wave = EffectiveDataset(generate_dataset(data_config, tmp_path / "single-wave"), "test")[
        16
    ]
    # Same single-precision quadrature RNG; only physical reconstruction differs.
    torch.testing.assert_close(single_wave["y"], small["y"], atol=2e-6, rtol=2e-5)


def test_invalid_index_and_lfm(data_config, data_manifest):
    dataset = EffectiveDataset(data_manifest, "train")
    with pytest.raises(IndexError):
        dataset[-1]
    data_config.values["receiver"]["sync_mode"] = "lfm_detect"
    from pgvamp_ofdm.data.generate import generate_records

    with pytest.raises(ValueError, match="oracle_timing"):
        generate_records(data_config)
