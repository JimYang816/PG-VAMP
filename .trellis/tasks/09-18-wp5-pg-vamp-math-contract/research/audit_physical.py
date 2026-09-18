"""WP5 forward-only receipt; reuse a generated WP3 manifest, no training."""

import json
import platform
import sys
from pathlib import Path

import torch

from pgvamp_ofdm.algorithms import MMSEDetector, PGVAMPDetector, VAMPDetector
from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.records import tensor_hash

torch.set_num_threads(4)
dataset = EffectiveDataset(Path(sys.argv[1]), "test")
report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
    "cuda_available": torch.cuda.is_available(),
    "scope": "WP5 shared physical forward and label isolation; no training/system smoke",
    "manifest": sys.argv[1],
    "runs": [],
}


def prediction_hash(output):
    payload = {key: getattr(output, key) for key in ("x_soft", "class_hat", "bits_hat")}
    if output.probabilities is not None:
        payload["probabilities"] = output.probabilities
    return tensor_hash(payload)


for detector in (MMSEDetector(), VAMPDetector(), PGVAMPDetector()):
    sample = dataset[16]
    epsilon = dataset.records[2]["path_epsilon"]
    assert torch.count_nonzero(epsilon) > 0
    original_input = tensor_hash(detection_inputs(sample))
    outputs, input_hashes = [], []
    for variant in ("original", "modified", "absent"):
        if variant == "modified":
            sample["x"] = torch.zeros_like(sample["x"])
            sample["bits"] = torch.zeros_like(sample["bits"])
        if variant == "absent":
            sample = {key: value for key, value in sample.items() if key not in ("x", "bits")}
        payload = detection_inputs(sample)
        input_hashes.append(tensor_hash(payload))
        with torch.no_grad():
            output = detector.detect(**{key: value[None] for key, value in payload.items()})
        assert torch.isfinite(output.x_soft).all()
        outputs.append(prediction_hash(output))
    assert len(set(outputs)) == len(set(input_hashes)) == 1
    report["runs"].append({
        "detector": type(detector).__name__, "sample_id": sample["sample_id"],
        "shape": list(sample["H"].shape), "dtype": str(sample["H"].dtype),
        "path_epsilon": epsilon.tolist(), "input_hash": original_input,
        "variant_input_hashes": input_hashes, "variant_prediction_hashes": outputs,
        "learned_scalars": sum(p.numel() for p in detector.parameters()),
        "diagnostics": {key: value.tolist() for key, value in output.diagnostics.items()
                        if isinstance(value, torch.Tensor)},
    })
assert len({(r["sample_id"], r["input_hash"]) for r in report["runs"]}) == 1
target = Path(__file__).with_name("physical-forward-receipt.json")
target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
