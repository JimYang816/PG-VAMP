"""Label-free inference from restricted CPU-loaded checkpoint and WP3 input."""

from pathlib import Path
from typing import Any

import torch

from .config import config_from_values
from .data.manifest import file_hash, save_tensors
from .data.materialize import load_materialized
from .training.checkpoint import load_checkpoint, model_for, physical_contract
from .utils.device import resolve_runtime


def infer(
    checkpoint: Path,
    input_path: Path,
    output: Path,
    *,
    device: str = "cpu",
    dtype: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("inference output already exists")
    state = load_checkpoint(checkpoint)
    if state["execution_kind"] != "physical":
        raise ValueError("algebra fixture checkpoint cannot infer physical data")
    config = config_from_values(state["resolved_config"])
    data = load_materialized(input_path, purpose="inference")
    input_config = config_from_values(data["metadata"]["resolved_config"], allow_legacy_data=True)
    if physical_contract(input_config) != physical_contract(config):
        raise ValueError("inference physics/mapping differs from checkpoint")
    runtime = resolve_runtime(device, dtype or state["dtype"])
    model = model_for(config, runtime)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    outputs: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("x_soft", "class_hat", "bits_hat", "probabilities")
    }
    with torch.inference_mode():
        for i in range(data["H"].shape[0]):
            h = data["H"][i : i + 1].to(runtime.device, runtime.complex_dtype)
            context = (
                f"sample={data['metadata']['sample_ids'][i]}, "
                f"dtype={h.dtype}, device={h.device}, "
                f"H_norm={torch.linalg.vector_norm(h).item()}, "
                f"H_max_abs={h.abs().amax().item()}"
            )
            try:
                result = model(
                    h,
                    data["y"][i : i + 1].to(runtime.device, runtime.complex_dtype),
                    data["sigma2"][i : i + 1].to(runtime.device, runtime.real_dtype),
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                raise type(exc)(f"{exc}; {context}") from exc
            for key in outputs:
                value = getattr(result, key)
                if value is None or not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"nonfinite inference {key}; {context}")
                outputs[key].append(value.cpu())
    artifact = {
        "schema_version": 1,
        **{k: torch.cat(v) for k, v in outputs.items()},
        "metadata": {k: data["metadata"][k] for k in ("sample_ids", "frame_ids", "channel_ids")},
        "checkpoint_sha256": file_hash(checkpoint),
        "input_sha256": file_hash(input_path),
        "device": str(runtime.device),
        "dtype": str(runtime.complex_dtype),
        "checkpoint_dtype": state["dtype"],
        "input_dtype": str(data["H"].dtype),
        "explicit_dtype_conversion": dtype is not None and dtype != state["dtype"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tensors(output, artifact)
    return {
        "output": str(output),
        "samples": len(data["metadata"]["sample_ids"]),
        "dtype": artifact["dtype"],
        "device": artifact["device"],
    }
