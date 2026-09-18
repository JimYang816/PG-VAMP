"""Run the bounded WP6 CLI acceptance sequence and retain exact process receipts."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/wp6-final-cli"))
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=False)
    (root / "two.yaml").write_text(
        "data:\n  train_frames: 2\n  val_frames: 1\n"
        "training:\n  max_steps: 2\n  validation_every_steps: 1\n"
        "  validation_max_blocks: 2\nevaluation:\n"
        "  frames_per_cell: 1\n  label: smoke_system\n",
        encoding="utf-8",
    )
    (root / "four.yaml").write_text(
        (root / "two.yaml").read_text().replace("max_steps: 2", "max_steps: 4"), encoding="utf-8"
    )
    commands = [
        ["inspect-config", "--config", "configs/cpu_dev.yaml"],
        ["inspect-config", "--config", "configs/smoke_system.yaml"],
        [
            "smoke",
            "--config",
            "configs/smoke_math.yaml",
            "--device",
            "cpu",
            "--output",
            str(root / "math"),
        ],
        [
            "smoke",
            "--config",
            "configs/smoke_math.yaml",
            "--device",
            "cpu",
            "--dtype",
            "complex64",
            "--output",
            str(root / "math-single"),
        ],
        [
            "smoke",
            "--config",
            "configs/smoke_system.yaml",
            "--device",
            "cpu",
            "--output",
            str(root / "system"),
        ],
        ["simulate", "--config", str(root / "two.yaml"), "--output", str(root / "data")],
        [
            "audit-data",
            "--manifest",
            str(root / "data/manifest.json"),
            "--waveform-frames",
            "1",
            "--output",
            str(root / "audit"),
        ],
        [
            "train",
            "--config",
            str(root / "two.yaml"),
            "--manifest",
            str(root / "data/manifest.json"),
            "--device",
            "cpu",
            "--output",
            str(root / "training"),
        ],
        [
            "train",
            "--config",
            str(root / "four.yaml"),
            "--manifest",
            str(root / "data/manifest.json"),
            "--device",
            "cpu",
            "--resume",
            str(root / "training/last.pt"),
            "--output",
            str(root / "training"),
        ],
        [
            "materialize",
            "--manifest",
            str(root / "data/manifest.json"),
            "--split",
            "val",
            "--without-labels",
            "--output",
            str(root / "unlabeled.pt"),
        ],
        [
            "infer",
            "--checkpoint",
            str(root / "training/best.pt"),
            "--input",
            str(root / "unlabeled.pt"),
            "--device",
            "cpu",
            "--output",
            str(root / "inference.pt"),
        ],
    ]
    receipts = []
    for number, args in enumerate(commands):
        command = [sys.executable, "-m", "pgvamp_ofdm", *args]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={**os.environ, "MKL_THREADING_LAYER": "TBB"},
        )
        receipt = {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        receipts.append(receipt)
        (root / "receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        print(number, args[0], result.returncode, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
