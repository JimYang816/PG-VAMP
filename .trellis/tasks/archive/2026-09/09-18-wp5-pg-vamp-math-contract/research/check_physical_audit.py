"""Fresh-loader physical inputs and all prediction hashes after review fixes."""
import json
from pathlib import Path

import torch

from pgvamp_ofdm.algorithms import MMSEDetector, PGVAMPDetector, VAMPDetector
from pgvamp_ofdm.data.dataset import EffectiveDataset, detection_inputs
from pgvamp_ofdm.data.records import tensor_hash

torch.set_num_threads(4)
manifest = Path("runs/wp5-check-final/data0/compact/manifest.json")
records = []
for detector in (MMSEDetector(), VAMPDetector(), PGVAMPDetector()):
    dataset = EffectiveDataset(manifest, "test")
    sample = dataset[16]
    epsilon = dataset.records[2]["path_epsilon"]
    assert torch.count_nonzero(epsilon) > 0
    variants = [sample,
                {**sample, "x": torch.zeros_like(sample["x"]), "bits": torch.zeros_like(sample["bits"])},
                {k: v for k, v in sample.items() if k not in ("x", "bits")}]
    inputs, predictions = [], []
    for variant in variants:
        payload = detection_inputs(variant)
        assert payload["H"].shape == (400, 400)
        inputs.append(tensor_hash(payload))
        with torch.no_grad():
            result = detector.detect(**{k: v[None] for k, v in payload.items()})
        fields = {k: getattr(result, k) for k in ("x_soft", "class_hat", "bits_hat", "probabilities")
                  if getattr(result, k) is not None}
        assert all(torch.isfinite(v).all() for v in fields.values())
        predictions.append(tensor_hash(fields))
    assert len(set(inputs)) == len(set(predictions)) == 1
    records.append(dict(algorithm=type(detector).__name__,sample_id=sample["sample_id"],
                        shape=[400,400],epsilon=epsilon.tolist(),input_hashes=inputs,
                        prediction_hashes=predictions,parameters=sum(p.numel() for p in detector.parameters())))
assert len({(r["sample_id"],r["input_hashes"][0]) for r in records}) == 1
print(json.dumps(dict(manifest=str(manifest),threads=torch.get_num_threads(),cases=records,
                     scope="untrained forward only; no checkpoint/training/system smoke"),indent=2))
