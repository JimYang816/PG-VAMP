"""Bounded full-dimensional WP7 CLI acceptance with exact durable receipts."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/wp7-cli-acceptance"))
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    if not args.reports_only:
        root.mkdir(parents=True, exist_ok=False)
        (root / "small.yaml").write_text(
            "data:\n  train_frames: 2\n  val_frames: 1\n"
            "training:\n  max_steps: 2\n  validation_every_steps: 1\n"
            "  validation_max_blocks: 2\nevaluation:\n"
            "  scenarios: [affine_doppler_moderate]\n  esn0_db: [0, 10]\n"
            "  frames_per_cell: 1\n  label: smoke_system\n  batch_size: 2\n"
            "  timing_warmup: 1\n  timing_repeats: 3\n",
            encoding="utf-8",
        )
    config, manifest = str(root / "small.yaml"), str(root / "data/manifest.json")
    commands = [
        ["inspect-config", "--config", "configs/cpu_dev.yaml"],
        ["inspect-config", "--config", "configs/main.yaml"],
        ["evaluate", "--help"],
        ["benchmark", "--help"],
        ["report", "--help"],
        ["generate", "--config", config, "--output", str(root / "data")],
        [
            "audit-data",
            "--manifest",
            manifest,
            "--waveform-frames",
            "1",
            "--output",
            str(root / "audit"),
        ],
        ["train", "--config", config, "--manifest", manifest, "--output", str(root / "train1")],
        [
            "train",
            "--config",
            config,
            "--manifest",
            manifest,
            "--seed",
            "20260918",
            "--output",
            str(root / "train2"),
        ],
        [
            "evaluate",
            "--config",
            config,
            "--manifest",
            manifest,
            "--checkpoint",
            str(root / "train1/best.pt"),
            "--output",
            str(root / "evaluation1"),
        ],
        [
            "evaluate",
            "--config",
            config,
            "--manifest",
            manifest,
            "--checkpoint",
            str(root / "train2/best.pt"),
            "--output",
            str(root / "evaluation2"),
        ],
        [
            "benchmark",
            "--config",
            config,
            "--manifest",
            manifest,
            "--checkpoint",
            str(root / "train1/best.pt"),
            "--timing-mode",
            "per_observation_cold_H",
            "--output",
            str(root / "cold"),
        ],
        [
            "benchmark",
            "--config",
            config,
            "--manifest",
            manifest,
            "--checkpoint",
            str(root / "train1/best.pt"),
            "--timing-mode",
            "same_H_amortized",
            "--output",
            str(root / "amortized"),
        ],
        [
            "materialize",
            "--manifest",
            manifest,
            "--split",
            "val",
            "--without-labels",
            "--output",
            str(root / "unlabeled.pt"),
        ],
        [
            "infer",
            "--checkpoint",
            str(root / "train1/best.pt"),
            "--input",
            str(root / "unlabeled.pt"),
            "--output",
            str(root / "predictions.pt"),
        ],
    ]
    reports = [
        ["report", "--results", str(root / "evaluation1")],
        [
            "report",
            "--results",
            str(root / "evaluation1"),
            str(root / "evaluation2"),
            "--output",
            str(root / "combined-report"),
        ],
    ]
    commands = reports if args.reports_only else commands
    receipts_path = root / ("report-receipts.json" if args.reports_only else "receipts.json")
    receipts = []
    for i, command in enumerate(commands):
        argv = [sys.executable, "-m", "pgvamp_ofdm", *command]
        start = time.perf_counter()
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "MKL_THREADING_LAYER": "TBB", "PYTHONIOENCODING": "utf-8"},
        )
        receipts.append(
            {
                "argv": argv,
                "cwd": str(Path.cwd()),
                "exit_code": result.returncode,
                "elapsed_seconds": time.perf_counter() - start,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        receipts_path.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        print(i, command[0], result.returncode, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
