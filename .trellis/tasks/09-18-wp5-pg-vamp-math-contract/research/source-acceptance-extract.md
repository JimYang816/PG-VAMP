# Verbatim specification excerpt for WP5 context

Source: `docs/CODEX_ENGINEERING_SPEC.md`; SHA-256 `A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
Extracted without rewriting on 2026-09-18 to avoid context-injection truncation.
The original remains authoritative; recheck its hash before implementation.

## 20. 可直接落地的配置合同

以下为 `configs/base.yaml` 的默认内容。各 profile 只覆盖必要字段；最终解析后的完整配置必须写入运行目录。未知配置键报错，避免拼写错误静默失效。

```yaml
schema_version: 1
seed: 20260917

runtime:
  device: cpu
  dtype: complex128
  num_workers: 0
  cpu_threads: 4                   # 实际取 min(4, available_cpu_count)
  deterministic: true
  amp: false

waveform:
  sample_rate_hz: 96000
  band_hz: [21000, 27000]
  carrier_hz: 24000
  n_grid: 512
  n_fft_wave: 8192
  cp_samples: 2048
  n_data: 400
  n_pilots: 64
  n_guard_left: 24
  n_guard_right: 23
  dc_null: true
  allocation: default_400_64_47_1
  fft_norm: ortho
  modulation: qpsk_fixed_sign_bits
  symbol_energy: 1.0
  pilot_energy: 1.0
  coding: none
  ofdm_window: none

frame:
  n_ofdm_symbols: 8
  leading_silence_samples: 1920
  lfm_samples: 3840
  lfm_start_hz: 21000
  lfm_stop_hz: 27000
  lfm_window: tukey
  lfm_tukey_alpha: 0.1
  sync_guard_samples: 2048
  trailing_silence_samples: 2048
  arrival_offset_max_samples: 1920

channel:
  model: affine_time_scaling
  min_paths: 3
  max_paths: 6
  first_delay_s: 0.002
  max_delay_s: 0.014
  power_delay_tau_s: 0.003
  path_energy_normalization: per_frame_sum_abs2_one
  epsilon_max:
    affine_doppler_mild: 0.00002
    affine_doppler_moderate: 0.00010
    affine_doppler_strong: 0.00020
  forbid_interblock_interference: true
  validate_cp_support: true

receiver:
  front_end: ideal_complex_iq
  sync_mode: oracle_timing
  lfm_threshold: 0.1
  csi_mode: perfect
  noise_variance: known
  receive_rows: data_q
  pilot_cancellation: exact_known_channel

noise:
  axis: esn0_db_unit_symbol
  definition: E_abs_w_squared_equals_sigma2
  use_measured_per_frame_signal_power: false

data:
  backend: effective_fast
  storage: compact_frame_records
  train_frames: 128
  val_frames: 32
  train_esn0_range_db: [-5, 25]
  train_scenario_weights:
    static_multipath: 0.1
    affine_doppler_mild: 0.2
    affine_doppler_moderate: 0.5
    affine_doppler_strong: 0.2
  split_unit: physical_frame
  audit_waveform_frames: 2
  matrix_cache_entries: 4

pg_vamp:
  depth: 8
  rho_hi_db: 0.0
  rho_lo_db: -60.0
  min_gap_db: 0.5
  temperature_db: 3.0
  init_gap: inverse_depth
  init_mu: 0.8
  mask_mode: soft
  precision_min: 1.0e-10
  precision_max: 1.0e8
  alpha_margin: 1.0e-6
  trace_method: exact
  solve_method: cholesky
  nonlinear_invalid_policy: keep_previous
  jitter: 0.0

vamp:
  iterations: 8
  linear_method: svd_exact
  nonlinear_invalid_policy: keep_previous
  precision_min: 1.0e-10
  precision_max: 1.0e8
  alpha_margin: 1.0e-6
  learned_parameters: false
  adaptive_stopping: false

mmse:
  method: cholesky
  output: raw_linear_estimate
  qpsk_post_denoiser: false

training:
  batch_size: 1
  max_steps: 300
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  grad_clip_norm: 5.0
  loss: layer_weighted_complex_mse
  validation_every_steps: 50
  validation_max_blocks: 64
  checkpoint_metric: final_layer_nmse
  checkpoint_mode: min
  early_stopping: false

evaluation:
  algorithms: [mmse, vamp, pg_vamp]
  scenarios: [identity_awgn, static_multipath, affine_doppler_moderate]
  esn0_db: [0, 10, 20]
  frames_per_cell: 16
  batch_size: 1
  shared_samples: true
  frame_cluster_bootstrap_repeats: 2000
  label: development_only
  timing_mode: per_observation_cold_H
  timing_warmup: 5
  timing_repeats: 20
  save_per_frame_counts: true
```

必须验证的派生关系：

```text
n_fft_wave * (bandwidth_hz / n_grid) == sample_rate_hz
carrier_hz / subcarrier_spacing_hz 为整数
400 + 64 + 24 + 23 + 1 == 512
cp_samples / sample_rate_hz 与所有路径支撑条件一致
rho_hi - rho_lo - depth*min_gap > 0
dtype 为 complex128 或 complex64
device 默认 cpu
```

identity/static 测试场景必须正确覆盖 `channel.model` 对应的退化参数，不能继续沿用非零 epsilon。

`main.yaml` 覆盖：train_frames=1024、val_frames=128、max_steps=5000、validation_max_blocks=1024、五种测试场景、完整 7 点 SNR、frames_per_cell=256、label=`main_simulation`。设备仍是 CPU。

`smoke_math.yaml` 明确使用代数 fixture，不通过上述物理参数校验；`smoke_system.yaml` 保持完整物理尺寸，只减少帧数和更新步数。

---


### WP5：PG-VAMP 数学合同

实现单调门限、软门控、能量补足、安全矩阵、中心化完整残差、精确条件散度、方差校准、解析 QPSK 和非线性保护。

**验收：** 第 23 节的矩阵性质、Jacobian、全图极限、前向和参数梯度检查通过；16 个可学习标量严格成立。


### 23.2 算法数学

| 测试 | 核心断言 |
|---|---|
| MMSE | 正规方程残差、对角闭式、零信道、秩亏 |
| VAMP | SVD 与 Cholesky 层级前向一致、标准精度公式适用 |
| 参数数目 | PG-VAMP 恰好 $2T$ 个实标量；两个基线为 0 |
| 门限 | 严格单调、最小间隔、范围、初始化 |
| 图 | soft mask 嵌套、零边处理、不强制对称 |
| 能量 | $\operatorname{diag}G=\operatorname{diag}H^HH$ |
| 安全性 | $\bar G-H^HH\succeq0$，图细化的 Loewner 单调性 |
| 线性算子 | 残差形式与 `apply_B` 形式一致 |
| $B$ 性质 | Hermitian，$0\prec B\preceq A^{-1}$，用 solve 构造参考 |
| 单层目标 | 固定输入下二次目标不增，不误测成整网 BER 单调 |
| 条件散度 | 实数展开 Jacobian 的 $\operatorname{tr}(J)/(2N)$ 与解析散度一致 |
| 方差 | 非负平方和公式与完整协方差迹一致 |
| 全图极限 | 强制全图时 PG-VAMP 与精确 VAMP 一致，$\mu$ 不影响输出 |
| 对角极限 | identity/对角信道的退化与 hard bits 正确 |
| QPSK | tanh/sech 闭式与四点枚举概率一致 |
| 消息保护 | 负精度拒绝、保持旧消息、固定候选均值截断 |
| 无信息 | $H=0$ 返回均匀概率和零软估计 |
| 梯度 | raw_gaps/raw_mu 的 gradcheck 与整网反向 |
| 不变性 | 同时缩放 $H,y,\sigma^2$ 后检测等价 |

全图与对角信道下部分梯度自然为零，不要求其非零。只有非退化 ICI 夹具上验证可学习参数获得有效梯度。

### 23.3 工程与评测

| 测试 | 核心断言 |
|---|---|
| CPU 默认 | GPU 可用与否都不改变默认 CPU 行为 |
| CUDA 不可用 | 请求 CUDA 时可读报错，无静默回退 |
| dtype/device | 所有中间张量一致；CPU/CUDA 合理容差内等价 |
| checkpoint | 保存/加载预测一致、恢复后更新恰好 $2T$ 参数 |
| 标签隔离 | 修改 test target 不改变检测器输出 |
| 共享样本 | 三算法输入 sample ID 和哈希一致 |
| 指标 | 手工错误例子的 BER/SER/BLER/FER/NMSE 正确 |
| 统计 | 按帧 bootstrap、配对差值、零错误标记正确 |
| 失败处理 | 非有限输出不被静默移出分母后冒充完整结果 |
| 计时 | 含所声明的分解成本，CUDA 同步，记录 batch/线程 |
| CLI | generate/audit/train/evaluate/report/infer 的小闭环可运行 |
| 禁止 API | 正式路径不调用 inverse、CG、随机 trace，不静默 detach |

建议数值基准：

```text
complex128 核心前向: atol=1e-9, rtol=1e-8
small-matrix PSD:    min_eigenvalue >= -1e-9
gradcheck:          eps=1e-6, atol=2e-5, rtol=2e-4
complex64 smoke:    finite 输出/梯度 + 明确的单精度误差容限
```

容差可按条件数合理调整，但必须记录原因，不能为了通过错误实现任意放宽。

---
