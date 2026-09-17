import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pgvamp_ofdm.config import load_config


@pytest.mark.parametrize("console", [False, True])
def test_both_entries(console):
    command = (
        [
            str(
                Path(sys.executable).with_name(
                    "pgvamp-ofdm.exe" if os.name == "nt" else "pgvamp-ofdm"
                )
            )
        ]
        if console
        else [sys.executable, "-m", "pgvamp_ofdm"]
    )
    result = subprocess.run(
        command + ["inspect-config", "--config", "configs/cpu_dev.yaml"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["frame_samples"] == 91776


def test_output_and_error(tmp_path):
    out = tmp_path / "run"
    command = [sys.executable, "-m", "pgvamp_ofdm", "inspect-config"]
    result = subprocess.run(command + ["--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert set(p.name for p in out.iterdir()) == {"resolved_config.yaml", "environment.json"}
    load_config(out / "resolved_config.yaml")
    evidence = json.loads((out / "environment.json").read_text())
    assert evidence["engineering_spec_sha256"].upper() == (
        "A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B"
    )
    assert evidence["package_source_hashes"]
    result = subprocess.run(command + ["--dtype", "float16"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "configuration error" in result.stderr
