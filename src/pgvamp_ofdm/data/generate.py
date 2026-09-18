"""Compact generation: allocate physical identities before sampling and SNR expansion."""

from pathlib import Path
from typing import Any

import torch

from ..channel.parameters import sample_paths
from ..channel.validity import validate_cp_support
from ..config import Config, config_from_values
from ..modulation.qpsk import classes_to_symbols
from ..utils.device import Runtime, resolve_runtime
from ..utils.random import SEED_VERSION, derive_seed, generator, seed_runtime, stable_hash
from .manifest import (
    counts,
    distribution,
    file_hash,
    save_json,
    save_tensors,
    split_table,
    validate_lineage,
)
from .records import (
    GENERATOR_VERSION,
    MODEL_VERSION,
    SPLITS,
    allocation_metadata,
    record_frame,
    waveform_hash,
)


def generate_records(config: Config) -> list[dict[str, Any]]:
    """Generate compact CPU double-precision physical records with bounded CP retries."""
    config = config_from_values(config.values)
    c, data, evaluation = config.values, config.values["data"], config.values["evaluation"]
    if c["receiver"]["sync_mode"] != "oracle_timing":
        raise ValueError("compact datasets require oracle_timing; lfm_detect is a separate demo")
    master, blocks = c["seed"], c["frame"]["n_ofdm_symbols"]
    total = (
        data["train_frames"]
        + data["val_frames"]
        + len(evaluation["scenarios"]) * evaluation["frames_per_cell"]
    )
    physical_generation = {
        key: c[key] for key in ("seed", "waveform", "frame", "channel", "receiver")
    }
    physical_generation["distribution"] = {
        key: data[key]
        for key in ("train_frames", "val_frames", "train_esn0_range_db", "train_scenario_weights")
    }
    physical_generation["evaluation"] = {
        key: evaluation[key] for key in ("scenarios", "esn0_db", "frames_per_cell")
    }
    namespace = stable_hash([GENERATOR_VERSION, physical_generation])
    order = torch.randperm(total, generator=generator(derive_seed(master, "split", namespace)))
    identities: list[tuple[str, str | None]] = (
        [("train", None)] * data["train_frames"]
        + [("val", None)] * data["val_frames"]
        + [
            ("test", scenario)
            for scenario in evaluation["scenarios"]
            for _ in range(evaluation["frames_per_cell"])
        ]
    )
    runtime = Runtime(torch.device("cpu"), torch.complex128, torch.float64, torch.get_num_threads())
    records = []
    for index, (split, fixed_scenario) in zip(order.tolist(), identities):
        frame_id = stable_hash([namespace, "frame", index])
        channel_id = stable_hash([namespace, "channel", index])

        def gen(stream: str, *keys: Any) -> torch.Generator:
            return generator(derive_seed(master, stream, frame_id, *keys))

        scenarios = list(data["train_scenario_weights"])
        weights = torch.tensor(list(data["train_scenario_weights"].values()), dtype=torch.float64)
        scenario = (
            fixed_scenario
            or scenarios[int(torch.multinomial(weights, 1, generator=gen("channel", "scenario")))]
        )
        record: dict[str, Any] = {
            "schema_version": 1,
            "frame_id": frame_id,
            "channel_id": channel_id,
            "split": split,
            "scenario": scenario,
            "data_bits": torch.randint(
                2, (blocks, 400, 2), generator=gen("bits"), dtype=torch.uint8
            ),
            "pilot_symbols": classes_to_symbols(
                torch.randint(4, (blocks, 64), generator=gen("pilots"))
            ),
            "arrival_offset_samples": int(
                torch.randint(
                    c["frame"]["arrival_offset_max_samples"] + 1,
                    (),
                    generator=gen("arrival_offset"),
                )
            ),
            "waveform_config_hash": waveform_hash(config),
            "generator_version": GENERATOR_VERSION,
            "cp_valid": True,
        }
        frame = record_frame(record, config, runtime)
        last_error = ""
        for attempt in range(data["max_channel_attempts"]):
            paths = sample_paths(config, scenario, generator=gen("channel", "paths", attempt))
            try:
                validate_cp_support(paths, frame.layout, config)
            except ValueError as exc:
                last_error = str(exc)
                continue
            break
        else:
            raise ValueError(f"channel retries exhausted for frame {frame_id}: {last_error}")
        record.update(
            path_count=paths.gain.numel(),
            path_gain_complex=paths.gain,
            path_delay_s=paths.delay_s,
            path_epsilon=paths.epsilon,
            generation_rejections=attempt,
        )
        low, high = data["train_esn0_range_db"]
        snr_copies = evaluation["esn0_db"] if split == "test" else [None]
        for snr in snr_copies:
            copy_id = "continuous" if snr is None else float(snr).hex()
            esn0 = (
                low + (high - low) * torch.rand(blocks, generator=gen("snr"), dtype=torch.float64)
                if snr is None
                else torch.full((blocks,), float(snr), dtype=torch.float64)
            )
            records.append(
                {
                    **record,
                    "snr_copy": copy_id,
                    "esn0_db": esn0,
                    **{
                        f"noise_seed_{part}": torch.tensor(
                            [
                                derive_seed(master, "noise", frame_id, block, copy_id, part)
                                for block in range(blocks)
                            ],
                            dtype=torch.int64,
                        )
                        for part in ("real", "imag")
                    },
                }
            )
    validate_lineage(records, config)
    return records


def generate_dataset(
    config: Config, output: str | Path, *, allow_large_output: bool = False
) -> Path:
    """Publish shards first and manifest last. Never overwrite an existing dataset."""
    from ..cli import _provenance
    from .materialize import _metadata_reserve, estimate_size, materialize

    config = config_from_values(config.values)
    c = config.values
    rt = c["runtime"]
    runtime = resolve_runtime(rt["device"], rt["dtype"], rt["cpu_threads"], rt["deterministic"])
    seed_runtime(c["seed"], runtime.device)
    if c["data"]["storage"] == "materialized":
        test_frames = len(c["evaluation"]["scenarios"]) * c["evaluation"]["frames_per_cell"]
        frame_counts = (
            c["data"]["train_frames"],
            c["data"]["val_frames"],
            test_frames * len(c["evaluation"]["esn0_db"]),
        )
        physical_frames = c["data"]["train_frames"] + c["data"]["val_frames"] + test_frames
        estimate = sum(
            estimate_size(frames * c["frame"]["n_ofdm_symbols"], 400, rt["dtype"])[
                "estimated_output_bytes"
            ]
            + _metadata_reserve(c, physical_frames)
            for frames in frame_counts
        )
        if estimate > c["data"]["max_output_bytes"] and not allow_large_output:
            raise ValueError(
                f"materialized output estimate {estimate} exceeds budget; use --allow-large-output"
            )
    output = Path(output)
    if output.exists():
        raise ValueError("dataset output already exists; choose a fresh path")
    records = generate_records(config)
    output.mkdir(parents=True)
    shards = []
    for split in SPLITS:
        directory = output / split
        directory.mkdir()
        selected = [r for r in records if r["split"] == split]
        size = c["data"]["shard_frames"]
        for number, start in enumerate(range(0, len(selected), size)):
            path = directory / f"frame_records_{number:03d}.pt"
            batch = selected[start : start + size]
            save_tensors(path, batch)
            shards.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "split": split,
                    "sha256": file_hash(path),
                    "frame_copies": len(batch),
                }
            )
    manifest = {
        "schema_version": 1,
        "resolved_config": c,
        "config_sha256": stable_hash(c),
        "waveform_config_hash": waveform_hash(config),
        "generator_version": GENERATOR_VERSION,
        "model_version": MODEL_VERSION,
        "allocation": allocation_metadata(config),
        "qpsk_mapping": "class=2*b_real+b_imag; x=((1-2*b_real)+j*(1-2*b_imag))/sqrt(2)",
        "snr_definition": "Es/N0 (dB), unit-energy data QPSK; sigma2=E|w|^2",
        "master_seed": c["seed"],
        "seed_derivation": SEED_VERSION,
        "distribution": distribution(config),
        "environment": _provenance(runtime),
        "counts": counts(records, c["frame"]["n_ofdm_symbols"]),
        "split_table": split_table(records),
        "shards": shards,
        "audit_policy": (
            "explicit audit-data command; default frame count from data.audit_waveform_frames"
        ),
    }
    path = output / "manifest.json"
    if c["data"]["storage"] == "materialized":
        pending = output / "manifest.pending.json"
        save_json(pending, manifest)
        for split in SPLITS:
            materialize(
                pending, split, output / f"{split}.pt", allow_large_output=allow_large_output
            )
        pending.replace(path)
    else:
        save_json(path, manifest)
    return path
