"""Independent NumPy scalar-definition/solve audit; run from repository root."""

import json
import platform

import numpy as np
import torch

from pgvamp_ofdm.algorithms.pg_vamp.linear import linear_layer

torch.set_num_threads(4)
rng = np.random.default_rng(91852)
receipts = []
for n in (8, 16, 32):
    for jitter in (0.0, 0.125):
        H = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2 * n)
        y = rng.normal(size=n) + 1j * rng.normal(size=n)
        r2 = 0.2 * (rng.normal(size=n) + 1j * rng.normal(size=n))
        M = rng.uniform(size=(n, n))
        np.fill_diagonal(M, 1)
        sigma, gamma, mu = 0.7, 1.3, 0.63
        d = np.sum((1 - M**2) * np.abs(H)**2, axis=0)
        ell = np.array([
            sum((1 - M[k, i] * M[k, j]) * abs(H[k, i]) * abs(H[k, j])
                for k in range(n) for j in range(n) if i != j)
            for i in range(n)
        ])
        gated = M * H
        G = gated.conj().T @ gated + np.diag(d)
        Gbar = G + np.diag(ell)
        eye = np.eye(n)
        A = H.conj().T @ H / sigma + gamma * eye
        P = Gbar / sigma + (gamma + jitter) * eye
        solved = np.linalg.solve(P, eye)
        B = (1 + mu) * solved - mu * solved @ A @ solved
        W = B @ H.conj().T / sigma
        c = np.trace(W @ H).real / n
        K = W / c
        innovation = W @ (y - H @ r2)
        E = eye - K @ H
        covariance = E @ E.conj().T / gamma + sigma * K @ K.conj().T
        variance = np.trace(covariance).real / n
        actual = linear_layer(
            torch.tensor(H)[None], torch.tensor(y)[None], torch.tensor([sigma], dtype=torch.float64),
            torch.tensor(r2)[None], torch.tensor([gamma], dtype=torch.float64),
            torch.tensor(M)[None], torch.tensor(mu, dtype=torch.float64), jitter=jitter,
        )
        expected = dict(d=d, ell=ell, G=G, Gbar=Gbar, P=P, W=W, K=K, c=c,
                        innovation=innovation, xhat2=r2+innovation, r1=r2+innovation/c,
                        gamma1=1/variance, alpha2=1-c)
        errors = {}
        for key, value in expected.items():
            observed = actual[key][0].numpy()
            np.testing.assert_allclose(observed, value, atol=1e-9, rtol=1e-8)
            errors[key] = float(np.max(np.abs(observed-value)))
        psd_min = float(np.linalg.eigvalsh(Gbar-H.conj().T@H).min())
        upper_min = float(np.linalg.eigvalsh(np.linalg.solve(A,eye)-B).min())
        b_min = float(np.linalg.eigvalsh(B).min())
        assert psd_min >= -1e-9 and upper_min >= -1e-9 and b_min > 0
        np.testing.assert_allclose(B, B.conj().T, atol=1e-9, rtol=1e-8)
        def objective(x):
            return np.linalg.norm(y-H@x)**2/sigma+gamma*np.linalg.norm(x-r2)**2
        assert objective(r2+innovation) <= objective(r2)+1e-9
        receipts.append(dict(n=n,jitter=jitter,errors=errors,majorizer_min=psd_min,
                             inverse_order_min=upper_min,b_min=b_min,condition_P=float(np.linalg.cond(P))))
print(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                     threads=torch.get_num_threads(),atol=1e-9,rtol=1e-8,cases=receipts),indent=2))
