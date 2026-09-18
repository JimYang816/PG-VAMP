"""Record reviewer's fresh physical payload and full-output label isolation."""
import hashlib
import json
import os
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "TBB"
import torch
from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector
from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.records import tensor_hash

torch.set_num_threads(4)
root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
out = Path(__file__).resolve().parent
receipt = json.loads((out/"check-validation.json").read_text())
base = receipt["commands"][0]["argv"][-1].split("=",1)[1]
manifest = root/base/"data0/compact/manifest.json"
dataset = EffectiveDataset(manifest,"test")
report = dict(manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
              path_epsilon=dataset.records[2]["path_epsilon"].tolist(),detectors=[])
for model in (MMSEDetector(),VAMPDetector()):
    sample = dataset[16]
    outputs = []
    input_hashes = []
    for labels in ("original","changed","absent"):
        if labels == "changed":
            sample["x"] = -sample["x"]
            sample["bits"] = 1-sample["bits"]
        elif labels == "absent":
            del sample["x"],sample["bits"]
        payload = detection_inputs(sample)
        input_hashes.append(tensor_hash(payload))
        result = model.detect(**{k:v[None] for k,v in payload.items()})
        assert tensor_hash(payload) == input_hashes[-1]
        values = dict(x_soft=result.x_soft,class_hat=result.class_hat,bits_hat=result.bits_hat)
        if result.probabilities is not None:
            values["probabilities"] = result.probabilities
        assert all(torch.isfinite(t).all() for t in values.values())
        outputs.append(tensor_hash(values))
    assert len(set(outputs)) == len(set(input_hashes)) == 1
    report["detectors"].append(dict(algorithm=model.name,sample_id=sample["sample_id"],
        input_sha256=input_hashes[0],all_output_sha256=outputs[0],
        label_cases=["original","changed","absent"],shape=list(payload["H"].shape),
        dtype=str(payload["H"].dtype),device=str(payload["H"].device),
        counts={k:v.tolist() for k,v in result.diagnostics.items() if isinstance(v,torch.Tensor)}))
assert report["detectors"][0]["input_sha256"] == report["detectors"][1]["input_sha256"]
(out/"check-physical.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
