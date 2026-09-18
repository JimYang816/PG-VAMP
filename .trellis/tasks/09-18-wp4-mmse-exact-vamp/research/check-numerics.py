"""Independent reviewer audit: NumPy solves and explicit four-point posterior."""
import json
import os
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "TBB"
import numpy as np
import torch
from pgvamp_ofdm.algorithms import MMSEDetector, VAMPDetector

torch.set_num_threads(4)
records = []
alphabet = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
for seed in (903, 904, 905):
    rng = np.random.default_rng(seed)
    n = 7
    h = (rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))) / np.sqrt(2*n)
    y = rng.normal(size=n) + 1j*rng.normal(size=n)
    noise = 0.8
    H, Y, S = torch.from_numpy(h)[None], torch.from_numpy(y)[None], torch.tensor([noise], dtype=torch.float64)
    mmse = MMSEDetector().detect(H,Y,S).x_soft[0].numpy()
    expected = np.linalg.solve(h.conj().T@h + noise*np.eye(n), h.conj().T@y)
    np.testing.assert_allclose(mmse, expected, atol=1e-9, rtol=1e-8)
    result = VAMPDetector().detect(H,Y,S,return_diagnostics=True)
    r = np.zeros(n, dtype=complex)
    gamma = 1.0
    errors = {}
    for layer, actual in enumerate(result.diagnostics["layers"]):
        a = h.conj().T@h/noise + gamma*np.eye(n)
        x2 = np.linalg.solve(a, h.conj().T@y/noise + gamma*r)
        alpha2 = gamma*np.trace(np.linalg.solve(a, np.eye(n))).real/n
        c = 1-alpha2
        r1 = (x2-alpha2*r)/c
        gamma1 = gamma*c/alpha2
        logits = -gamma1*np.abs(r1[:,None]-alphabet)**2
        weights = np.exp(logits-logits.max(axis=-1,keepdims=True))
        probabilities = weights/weights.sum(axis=-1,keepdims=True)
        mean = (probabilities*alphabet).sum(-1)
        vbar = (probabilities*np.abs(alphabet-mean[:,None])**2).sum(-1).mean()
        alpha1 = gamma1*vbar
        expected_states = dict(r2=r, gamma2=gamma, xhat2=x2, alpha2=alpha2, c=c,
            r1=r1, gamma1=gamma1, xhat1=mean, probabilities=probabilities,
            vbar=vbar, alpha1=alpha1)
        for key,value in expected_states.items():
            observed = actual[key][0].numpy()
            np.testing.assert_allclose(observed,value,atol=1e-9,rtol=1e-8)
            errors[key] = max(errors.get(key,0),float(np.max(np.abs(observed-value))))
        if layer < 7:
            candidate = (mean-alpha1*r1)/(1-alpha1)
            precision = 1/vbar-gamma1
            if 1-alpha1 >= 1e-6 and precision >= 1e-10 and np.isfinite(candidate).all():
                r, gamma = candidate, min(precision,1e8)
    records.append(dict(seed=seed,n=n,iterations=8,mmse_max_error=float(np.max(np.abs(mmse-expected))),
                        linear_condition=float(np.linalg.cond(h.conj().T@h/noise+np.eye(n))),
                        layer_max_errors=errors))
path = Path(__file__).with_name("check-numerics.json")
path.write_text(json.dumps(records,indent=2),encoding="utf-8")
print(json.dumps(records,indent=2))
