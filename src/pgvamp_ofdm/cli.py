"""Configuration inspection and explicit WP3 data commands."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .config import load_config, summarize
from .utils.device import Runtime, resolve_runtime


def _provenance(runtime: Runtime) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if root.name == "src":
        root = root.parent
    # Package source hashes remain available when installed outside a checkout.
    package = Path(__file__).resolve().parent
    hashes = {
        str(p.relative_to(package)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.suffix in (".py", ".yaml")
    }
    spec = root / "docs" / "CODEX_ENGINEERING_SPEC.md"
    commit, dirty = None, None
    if (root / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False
        )
        commit = head.stdout.strip() if head.returncode == 0 else None
        dirty = bool(status.stdout) if status.returncode == 0 else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "device": str(runtime.device),
        "cpu_threads": runtime.cpu_threads,
        "complex_dtype": str(runtime.complex_dtype),
        "real_dtype": str(runtime.real_dtype),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER"),
        "git_commit": commit,
        "git_dirty": dirty,
        "package_source_hashes": hashes,
        "package_source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "engineering_spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest()
        if spec.exists()
        else None,
        "engineering_spec_status": "present"
        if spec.exists()
        else "not bundled; no runtime dependency",
    }


def main(argv: list[str] | None = None) -> int:
    """Inspect validated config; optional --output saves config and provenance, never samples."""
    parser = argparse.ArgumentParser(prog="pgvamp-ofdm")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-config", help="validate and print static configuration")
    inspect.add_argument("--config", type=Path)
    inspect.add_argument("--device")
    inspect.add_argument("--dtype")
    inspect.add_argument("--output", type=Path)
    simulate = sub.add_parser(
        "simulate", aliases=["generate"], help="generate compact physical data"
    )
    simulate.add_argument("--config", type=Path)
    simulate.add_argument("--device")
    simulate.add_argument("--dtype")
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--allow-large-output", action="store_true")
    audit = sub.add_parser("audit-data", help="audit saved records through independent waveforms")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--waveform-frames", type=int)
    audit.add_argument("--output", type=Path, required=True)
    dense = sub.add_parser("materialize", help="explicit budgeted dense export")
    dense.add_argument("--manifest", type=Path, required=True)
    dense.add_argument("--split", choices=("train", "val", "test"), required=True)
    dense.add_argument("--output", type=Path, required=True)
    dense.add_argument("--max-output-bytes", type=int)
    dense.add_argument("--allow-large-output", action="store_true")
    dense.add_argument("--without-labels", action="store_true")
    train_parser = sub.add_parser("train", help="train PG-VAMP or strictly resume")
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--dtype")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--output", type=Path, required=True)
    inference = sub.add_parser("infer", help="label-free inference from safe checkpoint")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--dtype")
    inference.add_argument("--output", type=Path, required=True)
    smoke_parser = sub.add_parser("smoke", help="two-update acceptance exercise")
    smoke_parser.add_argument("--config", type=Path, required=True)
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--dtype")
    smoke_parser.add_argument("--seed", type=int)
    smoke_parser.add_argument("--output", type=Path)
    for command in ("evaluate", "benchmark"):
        evaluation = sub.add_parser(
            command,
            help="paired evaluation"
            if command == "evaluate"
            else "declared detector timing protocols",
        )
        evaluation.add_argument("--config", type=Path)
        evaluation.add_argument("--manifest", type=Path, required=True)
        evaluation.add_argument("--checkpoint", type=Path)
        evaluation.add_argument("--algorithms", nargs="+", choices=("mmse", "vamp", "pg_vamp"))
        evaluation.add_argument("--device", default="cpu")
        evaluation.add_argument("--dtype")
        evaluation.add_argument(
            "--seed", type=int, help="test seed; must match pre-generated manifest"
        )
        evaluation.add_argument("--allow-untrained", action="store_true")
        evaluation.add_argument("--output", type=Path, required=True)
        if command == "evaluate":
            evaluation.add_argument("--bootstrap-seed", type=int)
        else:
            evaluation.add_argument(
                "--timing-mode",
                choices=("per_observation_cold_H", "same_H_amortized", "both"),
                default="both",
            )
    report_parser = sub.add_parser("report", help="read persisted results; never rerun detection")
    report_parser.add_argument("--results", nargs="+", type=Path, required=True)
    report_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            from .reporting import report

            print(json.dumps(report(args.results, args.output), allow_nan=False))
            return 0
        if args.command == "infer":
            from .inference import infer

            print(
                json.dumps(
                    infer(
                        args.checkpoint,
                        args.input,
                        args.output,
                        device=args.device,
                        dtype=args.dtype,
                    )
                )
            )
            return 0
        if args.command == "audit-data":
            from .data.audit import audit_dataset

            print(
                json.dumps(
                    audit_dataset(args.manifest, args.output, waveform_frames=args.waveform_frames)
                )
            )
            return 0
        if args.command == "materialize":
            from .data.materialize import materialize

            print(
                json.dumps(
                    materialize(
                        args.manifest,
                        args.split,
                        args.output,
                        max_output_bytes=args.max_output_bytes,
                        allow_large_output=args.allow_large_output,
                        labeled=not args.without_labels,
                    )
                )
            )
            return 0
        config = load_config(args.config, device=args.device, dtype=args.dtype)
        if args.command in ("evaluate", "benchmark"):
            from .evaluation.runner import benchmark, evaluate
            from .evaluation.timing import MODES

            if args.seed is not None:
                if args.seed < 0:
                    raise ValueError("seed must be nonnegative")
                config.values["seed"] = args.seed
            options = dict(
                checkpoint=args.checkpoint,
                algorithms=args.algorithms,
                allow_untrained=args.allow_untrained,
                argv=argv if argv is not None else __import__("sys").argv,
            )
            if args.command == "evaluate":
                result = evaluate(
                    config,
                    args.manifest,
                    args.output,
                    bootstrap_seed=args.bootstrap_seed,
                    **options,
                )
            else:
                modes = MODES if args.timing_mode == "both" else (args.timing_mode,)
                result = benchmark(config, args.manifest, args.output, modes=modes, **options)
            print(json.dumps(result, allow_nan=False))
            return 0
        if args.command in ("train", "smoke"):
            if args.seed is not None:
                if args.seed < 0:
                    raise ValueError("seed must be nonnegative")
                config.values["seed"] = args.seed
            if args.command == "train":
                from .training.trainer import train

                print(json.dumps(train(config, args.manifest, args.output, resume=args.resume)))
            else:
                import uuid

                from .smoke import smoke

                destination = args.output or Path("runs") / ("smoke-" + uuid.uuid4().hex[:12])
                print(json.dumps(smoke(config, destination), allow_nan=False))
            return 0
        if args.command in ("simulate", "generate"):
            from .data.generate import generate_dataset

            print(
                json.dumps(
                    {
                        "manifest": str(
                            generate_dataset(
                                config, args.output, allow_large_output=args.allow_large_output
                            )
                        )
                    }
                )
            )
            return 0
        rt = config.values["runtime"]
        runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
        rt["cpu_threads"] = runtime.cpu_threads
        rt["device"] = str(runtime.device)
        summary = summarize(config)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            config.save(args.output / "resolved_config.yaml")
            (args.output / "environment.json").write_text(
                json.dumps(_provenance(runtime), indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"configuration error: {exc}\n")
    return 0
