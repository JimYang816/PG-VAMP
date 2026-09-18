# Algorithm and physical-model guide

This is a navigation and interpretation guide to the immutable
[engineering specification](CODEX_ENGINEERING_SPEC.md), especially §§6, 8–17.
Its equations and numerical protections remain authoritative.

## From waveform to detection (§§2–10)

The 8192-point unitary FFT represents a 512-position physical grid at 96 kHz.
Of those positions, 400 contain unknown data QPSK, 64 known pilots and 48 zeros.
QPSK is `x=((1-2*b_R)+j*(1-2*b_I))/sqrt(2)`, class `2*b_R+b_I`.
The full frame has eight OFDM blocks and 6400 uncoded bits.

Each physical path evaluates the continuous transmitted analytic waveform at
`(1+epsilon_l)*t-tau_l0`, multiplied by complex gain `a_l`. Hence path Doppler
`epsilon_l*(fc+q*Delta_f)` depends on carrier q. The analytical grid operator uses
the **8192-term** finite Fourier sum from §6.5. Static multipath reduces to a
diagonal grid operator. CP support is checked for every path and block, including
the final block; a valid initial delay alone is insufficient.

For grid observation `Yg=Hg*Xg+Wg`, select data rows and subtract the exact known
pilot contribution:

```text
H = Hg[data_idx, data_idx]                     # 400 x 400
y = Yg[data_idx] - Hg[data_idx, pilot_idx] @ p
y = H @ x + w;  E[w w^H] = sigma2 I
```

This uses perfect CSI and ideal complex I/Q with oracle windows. It discards
observations at other receive rows. It is not an optimal all-512-observation
receiver or estimated-CSI model. Independent waveform evaluation followed by FFT
checks the analytical H, with relative error below 1e-9 for nondegenerate
complex128 acceptance cases (§6.6). LFM matching returns a selected template
start; it does not establish earliest-path arrival or replace oracle windows.

Noise is independent complex time-I/Q AWGN, each real component variance
sigma2/2. Unitary FFT preserves complex variance. Es/N0 uses unit data-symbol
energy, `sigma2=10^(-EsN0_dB/10)`; per-frame fading is not normalized away.

## Exact baselines (§§12–13)

**MMSE (linear)** solves `(H^H H + sigma2 I) xhat = H^H y` by Cholesky.
Its soft output is that raw linear estimate; nearest-QPSK decisions determine
bits. `probabilities=None`: no extra QPSK denoiser is attached.

**VAMP-8 with explicit numerical protection** initializes `r2=0, gamma2=1`.
With `gamma_w=1/sigma2`, each exact linear posterior solves
`A xhat2=b`, where `A=gamma_w H^H H+gamma2 I` and
`b=gamma_w H^H y+gamma2 r2`. Production uses one SVD of H per cold detection
call, followed by exact spectral solves. Its matched LMMSE relation is
`alpha2=gamma2 tr(A^-1)/N`, `c=1-alpha2`,
`r1=r2+(xhat2-r2)/c`, `gamma1=gamma2*c/alpha2`.
Production sums alpha2 and c separately to avoid cancellation. A QPSK posterior
and protected extrinsic update follow. The independent reference uses Cholesky.
Both baselines have zero learned parameters. The supplementary VAMP-32 config
must be labeled separately; fixed iteration count is not equal compute budget.

## PG-VAMP-VC (§14)

PG learns exactly `raw_gaps[T]` and `raw_mu[T]`: 2T real scalars, 16 at T=8.
Positive softplus gaps generate strictly decreasing thresholds within the
configured range with a minimum gap; `mu=sigmoid(raw_mu)`. Initial gaps are 1/T,
mu=0.8. For one-based t, let `g_t=softplus(raw_gaps_t)` and
`R=rho_hi-rho_lo-T*min_gap`; then
`rho_t=rho_hi-t*min_gap-R*sum_(j<=t)(g_j)/(1+sum_j(g_j))`.
The column-energy score is `10*log10(max(|H_ki|²/sum_j|H_ji|², 1e-12))`
for nonzero-energy columns; zero columns carry no information. Applying
`sigmoid((score-rho_t)/temperature)` gives directed soft masks; diagonal
entries are one and zero non-diagonal edges stay zero. H and masks are not
symmetrized. Training and inference both use soft masks.

For one layer, with mask M, define:

```text
HD = M * H
d_i = sum_k (1-M_ki^2) |H_ki|^2
G = HD^H HD + diag(d)
ell_i = sum_k sum_(j != i) (1-M_ki M_kj) |H_ki| |H_kj|
Gbar = G + diag(ell)
Pbar = gamma_w Gbar + gamma2 I
A = gamma_w H^H H + gamma2 I
```

Energy compensation gives `diag(G)=diag(H^H H)`. The analytical safety term
ensures `Pbar >= A > 0` in the Loewner order. Neither d nor ell is added to noise.
Production computes ell through the nonnegative quadratic-cost equivalent in
§14.3. It does not replace `(1-M^2)` by `(1-M)^2`.

The centered innovation uses the **full H residual** and the same Cholesky factor
of the actual Pbar for every solve:

```text
u = gamma_w H^H (y - H r2)
q0 = solve_P(u)
innovation = q0 + mu * solve_P(u - A q0)
xhat2 = r2 + innovation
B = (1+mu) Pbar^-1 - mu Pbar^-1 A Pbar^-1
```

Inverse notation describes the operator; code uses solves, never explicit
matrix inverses. This is not a zero-centered `solve_P(b)` or a residual using HD.
The actual conditional divergence and variance calibration are:

```text
W = B (gamma_w H^H)
c = real(trace(W H))/N; alpha2 = 1-c
K = W/c; r1 = r2 + innovation/c
var1 = ||I-KH||_F^2/(N gamma2) + sigma2 ||K||_F^2/N
gamma1 = 1/var1
```

Exact trace and Frobenius sums are required. The exact-VAMP precision identity
above cannot replace this PG variance. It is a local second-moment calibration
for an isotropic prior error independent of observation noise, not a general
finite-dimensional state-evolution theorem. Local conditional divergence fixes
layer parameters, while network gradients still propagate through their actual
dependencies. Numerically uninformative c yields zero information, not a
fabricated clamped observation.

## QPSK, protection and training (§§14.6–15)

For `u_R=sqrt(2)*gamma1*real(r1)` and similarly u_I, the posterior is
`xhat1=(tanh(u_R)+j*tanh(u_I))/sqrt(2)` with per-symbol variance
`v_i=(sech²(u_R)+sech²(u_I))/2`. Stable sech² evaluation avoids cancellation.
Four-class probabilities follow the fixed QPSK mapping. The true divergence is
`alpha1=gamma1*mean(v)`. Non-final layers propose:

```text
r2_candidate = (xhat1-alpha1*r1)/(1-alpha1)
gamma2_candidate = 1/mean(v)-gamma1
```

Accept only finite candidates with denominator at least 1e-6 and precision at
least 1e-10. Invalid/negative/undefined candidates keep the previous valid
message and increment diagnostics. Valid precision above 1e8 is capped while
preserving the candidate mean. Zero posterior variance has an explicit rejection
branch. H=0 gives zero soft output and uniform probabilities. The final returned
estimate is the posterior, not the extrinsic message.

Training minimizes `sum_t [2t/(T(T+1))] mean(|xhat1_t-x|²)` using Adam, lr=1e-3,
no weight decay and gradient norm clip 5. Best checkpoints use validation
final-layer NMSE; labels remain outside detector inputs. No arbitrary forward
detach, finite-step CG, stochastic trace or silent hard-mask conversion is used.
Dense Gram/Cholesky/multiple-RHS work may remain O(N³) per layer and O(N²) memory.
Few learned parameters or soft-edge ratios do not prove sparse acceleration.

## Implementation and independent checks

Paths below are relative to `src/pgvamp_ofdm/`.

| Contract | Modules | Independent evidence family |
| --- | --- | --- |
| Physical mapping/frame | `modulation/`, `waveform/frame.py`, `waveform/ofdm.py` | Allocation, QPSK, FFT/CP and real-passband tests |
| Channel/receiver | `channel/`, `waveform/continuous.py`, `receiver/` | Continuous waveform vs analytical H; pilot/noise/CP tests |
| MMSE/VAMP | `algorithms/mmse.py`, `algorithms/vamp.py` | Normal-equation residuals; `reference/dense_vamp_cholesky.py` |
| PG | `algorithms/pg_vamp/` | `reference/dense_pg_vamp.py`, PSD/limits/Jacobian/variance/gradient tests |
| Training | `training/`, `inference.py`, `smoke.py` | Strict resume, target isolation, checkpoint and full-size smoke |
| Metrics/report | `evaluation/`, `reporting/` | Hand counts, frame bootstrap, timing boundaries, persisted hashes |

See [validation](../VALIDATION.md) for actual checks. Error rates aggregate integer
counts before division; symbol NMSE aggregates energies before dB. BER/SER and
paired differences resample independent frames, not correlated bits. Zero measured
errors do not establish zero true BER. Hard failures retain IDs and planned
denominators and mark a cell incomplete. Cold-H and same-H timing include declared
preparation costs; throughput is distinct from physical-link goodput (§§16–17).
