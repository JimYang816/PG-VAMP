import json

import pytest

from pgvamp_ofdm.cli import main


def test_generate_alias_and_capacity_cli(tmp_path, capsys):
    output = tmp_path / "generated"
    assert main(["generate", "--config", "configs/wp3_smoke.yaml", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["manifest"]
    with pytest.raises(SystemExit) as result:
        main(
            [
                "materialize",
                "--manifest",
                str(output / "manifest.json"),
                "--split",
                "test",
                "--output",
                str(tmp_path / "dense.pt"),
                "--max-output-bytes",
                "1",
            ]
        )
    assert result.value.code == 2
    assert "allow-large-output" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["simulate", "--config", "configs/wp3_smoke.yaml", "--output", str(output)])
