# Configuration guide

The [engineering specification](CODEX_ENGINEERING_SPEC.md) §§2–10, 15–17 and 20
is authoritative. This guide maps it to [config.py](../src/pgvamp_ofdm/config.py),
[base.yaml](../configs/base.yaml) and [bundled defaults](../src/pgvamp_ofdm/base.yaml).

## Resolution and runtime

`load_config(path)` recursively overlays YAML on bundled defaults; lists replace
lists. Profiles are overlays, not inheritance chains. `cpu_dev.yaml` is empty.
Omitting `--config` where supported uses defaults. Unknown nested keys and
unsupported values error. Read each run's `resolved_config.yaml` for full settings.

Explicit `--device`/`--dtype` override runtime fields. `train`, `smoke`, `infer`,
`evaluate`, `benchmark` and `demo-frame` default their CLI device to CPU. Pass
`--device cuda:0` explicitly for CUDA; unavailable CUDA errors without fallback.
Inference inherits checkpoint dtype unless explicitly converted and recorded.
Correctness precision is complex128/float64; complex64/float32 is a separate
experiment. AMP/float16 are unsupported. Defaults are batch size 1, zero data
workers, deterministic operation and at most four available CPU threads.
The base seed is 20260917. Reproducibility is scoped to the recorded environment,
not promised bitwise across platforms.

## Profiles and workload

| YAML in `configs/` | Training / validation | Evaluation | Purpose |
| --- | --- | --- | --- |
| `smoke_math.yaml` | N=32, T=2, two updates | Algebra only | Gradient/integration smoke |
| `smoke_system.yaml` | 2 / 1 physical frames, two updates, T=8 | One frame per configured cell | Full-size smoke |
| `cpu_dev.yaml` | 128 / 32 frames, max 300 updates | 3 scenarios × 3 SNRs × 16 frames | `development_only` |
| `main.yaml` | 1024 / 128 frames, max 5000 updates | 5 scenarios × 7 SNRs × 256 frames | `main_simulation`, unexecuted |
| `cuda_example.yaml` | CPU development counts | CPU development grid | Explicit CUDA/complex64 runtime selection |
| `wp3_smoke.yaml` | 1 / 1 frame | 2 scenarios × 2 SNRs × 1 frame | Data acceptance fixture |
| `vamp_reference_32.yaml` | CPU development defaults | VAMP has 32 iterations | Supplementary reference |

All physical profiles retain 512/400/8192/CP2048/eight blocks. `smoke_math` is
the algebra exception. `main` still defaults to CPU. A frame has eight detection
blocks; test SNR copies share physical identity and are not independent frames.
Counts/update budgets do not establish convergence. `demo-frame` uses one frame,
irrespective of dataset count settings.

## Physical parameters

| Quantity | Default and interpretation | Source |
| --- | --- | --- |
| Sampling/grid | 96000 Hz; 6000/512 = 11.71875 Hz spacing; FFT 8192 | §§2–3 |
| Allocation | 400 data + 64 pilots + 47 guards + 1 DC null | §3 |
| Frequencies | Grid 21000–26988.28125 Hz; nonzero outer carriers 21281.25–26718.75 Hz | §3 |
| CP | 2048 samples = 21.333333 ms; useful block 85.333333 ms | §§2, 6.4 |
| Full frame | 1920 silence + 3840 LFM + 2048 guard + 8×10240 OFDM + 2048 silence = 91776 samples / 0.956 s | §4 |
| Payload | 6400 uncoded bits/frame, fixed sign-bit unit-energy QPSK | §4 |
| Channel | 3–6 paths, first delay 2 ms, max 14 ms, exponential power scale 3 ms | §6 |
| Path time scaling | mild ±2e-5; moderate ±1e-4; strong ±2e-4 per path | §6 |
| Noise | sigma2=10^(-EsN0/10); each I/Q variance sigma2/2 | §9 |

Signed `q` maps to `grid_index=q+256`, `baseband_fft_index=q mod 8192` and
`passband_fft_index=2048+q`. Matrices use ascending data/pilot `q`. Unitary FFT
normalization has no added oversampling gain. Rectangular OFDM has sidelobes;
guards do not make it strictly band limited. Arrival padding is separate from
fixed frame duration.

Every path/block/window must satisfy CP support at both FFT endpoints. Invalid
generated channels are resampled with recorded rejection counts. Path gains
have frame-level energy normalization; H columns/Frobenius norm are not
normalized. Noise is never set from measured per-frame signal power.
The receiver uses ideal complex I/Q, `oracle_timing`, perfect CSI, known noise,
data-frequency rows and exact pilot cancellation. Estimated CSI, coding, real
ADC filtering and general `lfm_detect` dataset replay are unsupported.
The standalone synchronization demo does not change those assumptions.

## Data, training and evaluation

`effective_fast` reconstructs the audited analytical operator; `waveform_reference`
independently evaluates continuous waveforms. Both use time-I/Q noise then FFT.
Compact frame records plus a hashed manifest are default storage. Split, channel,
bits, pilots, noise and arrival streams derive from stable hashes. Split physical
frames before expanding blocks/SNR copies; no frame/channel lineage crosses splits.

Defaults `matrix_cache_entries=4`, `shard_frames=64`, `max_channel_attempts=100`
and `max_output_bytes=1073741824` bound cache, shards, rejection and export budgets.
One complex128 400×400 H alone is 2,560,000 bytes. Dense export estimates payload
and serialization reserve; exceeding its budget requires `--allow-large-output`.
`--without-labels` creates inference input. Labeled train/evaluation consumers
reject missing targets. Auditing is explicit.

PG defaults: T=8, soft masks, threshold range (−60,0) dB, minimum gap 0.5 dB,
temperature 3 dB, initial gaps 1/T and mu=0.8. Only `raw_gaps[T]`/`raw_mu[T]`
are trained. Cholesky/exact traces/zero jitter are defaults. Precision bounds are
1e-10 and 1e8; invalid nonlinear messages keep the prior valid value. VAMP uses
eight exact SVD iterations; MMSE returns raw Cholesky linear estimates.
See [the equations](ALGORITHMS.md) before changing algorithms.

Training uses layer-weighted complex MSE, Adam lr=1e-3, weight decay 0, gradient
clip 5, validation every 50 updates and up to 64 validation blocks (main: 1024).
Best selection uses final-layer validation NMSE. `max_steps` is cumulative;
increase it explicitly to extend a completed run. Resume checks physics,
mapping, depth/mask, optimizer, data identity and precision, and restores
optimizer/sampler/RNG state. Allowed runtime changes are recorded.

Training scenarios static/mild/moderate/strong have weights 0.1/0.2/0.5/0.2;
Es/N0 is uniform in [−5,25] dB. Development test uses identity/static/moderate
at [0,10,20] dB. Main uses all five scenarios at [−5,0,5,10,15,20,25] dB.
Evaluation config must match saved test population. Bootstrap defaults to 2000
frame-cluster resamples. Timing defaults to five warmups/twenty repeats;
`benchmark --timing-mode both` separates cold-H and same-H reuse.
