# WP8 source excerpts

Authority: docs/CODEX_ENGINEERING_SPEC.md; SHA256 a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b.

Verbatim planning excerpts below; they do not replace the source. Before editing math/config/physical behavior read source §§2–17 and 20 in bounded ranges. Full source exceeds injection limit; do not rely on a truncated injection.


## Source lines 1887–1891

### WP8：完整交付

整理 README、配置说明、算法数学说明、测试记录、实现状态和命令；至少实际运行 CPU 的完整尺寸 smoke。

主训练或完整 SNR sweep 因资源限制未执行时，保留可运行命令与 `未执行` 状态，不能用占位曲线宣称已完成指标分析。完成代码功能与完成充分性能验证应分别验收。


## Source lines 1740–1833

## 21. CLI 必须支持的命令

下面是 Codex 需要实现的目标命令，**不是本文件已经运行过的工程命令**。

### 21.1 安装与参数检查

```bash
python -m pip install -e '.[test]'
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python -m pytest -q
```

`inspect-config` 打印最终设备、精度、采样率、FFT 点数、载波数、频率端点、CP 时长、帧长度、数据比特数，以及潜在磁盘占用估计。

### 21.2 数学 smoke 与完整物理 smoke

```bash
python -m pgvamp_ofdm smoke --config configs/smoke_math.yaml --device cpu
python -m pgvamp_ofdm smoke --config configs/smoke_system.yaml --device cpu
```

`smoke_system` 至少包含真实 512 网格波形、400 维有效系统、三算法前向、PG-VAMP 两次更新、checkpoint 保存/加载和指标计数。它不是性能结论。

### 21.3 生成数据

```bash
python -m pgvamp_ofdm simulate \
  --config configs/cpu_dev.yaml \
  --output data/cpu_dev

python -m pgvamp_ofdm audit-data \
  --manifest data/cpu_dev/manifest.json \
  --waveform-frames 2 \
  --output results/data_audit
```

生成器不自动运行大量训练；训练器也不在找不到数据时偷偷生成另一套数据。

### 21.4 CPU 训练与恢复

```bash
python -m pgvamp_ofdm train \
  --config configs/cpu_dev.yaml \
  --manifest data/cpu_dev/manifest.json \
  --device cpu \
  --output runs/pg_cpu_dev

python -m pgvamp_ofdm train \
  --config configs/cpu_dev.yaml \
  --manifest data/cpu_dev/manifest.json \
  --device cpu \
  --resume runs/pg_cpu_dev/last.pt \
  --output runs/pg_cpu_dev
```

恢复时不可无说明覆盖波形配置、QPSK 映射、mask 或深度。允许变更的运行配置如设备需明确记录。

### 21.5 显式 GPU 训练与 CPU 推理

```bash
python -m pgvamp_ofdm train \
  --config configs/main.yaml \
  --manifest data/main/manifest.json \
  --device cuda:0 \
  --dtype complex64 \
  --output runs/pg_cuda

python -m pgvamp_ofdm infer \
  --checkpoint runs/pg_cuda/best.pt \
  --input data/example_effective.pt \
  --device cpu \
  --output results/inference.pt
```

设备可以切换；dtype 默认继承 checkpoint，显式转换时单独记录并测试。CPU 推理不是重新训练。

### 21.6 三算法评测与报告

```bash
python -m pgvamp_ofdm evaluate \
  --config configs/cpu_dev.yaml \
  --manifest data/cpu_dev/manifest.json \
  --checkpoint runs/pg_cpu_dev/best.pt \
  --algorithms mmse vamp pg_vamp \
  --device cpu \
  --output results/compare_cpu_dev

python -m pgvamp_ofdm report \
  --results results/compare_cpu_dev
```

PG-VAMP 评测缺少 checkpoint 时默认报错；只有显式 `--allow-untrained` 才允许未训练参数，并且算法名必须标注 `PG-VAMP-untrained`。

独立的 `benchmark` 命令采用第 17.3 节时间协议。`demo-frame` 命令输出完整波形、同步结果及真实尺寸的接收示例；`materialize` 命令把紧凑记录导出为旧接口兼容的有效矩阵文件。


## Source lines 1895–1973

## 23. 必须实现的测试清单

### 23.1 通信和生成器

| 测试 | 核心断言 |
|---|---|
| 资源分配 | 400/64/47/1、集合互斥、索引映射可逆 |
| 频率一致性 | $96000/8192=6000/512$ |
| QPSK | 4 类与 2 比特双向精确一致、平均符号能量 1 |
| FFT/IFFT | 单位酉可逆、能量守恒、正确轴处理 |
| 实通带输出 | fs=96 kHz、正频率位置正确、无信道恢复星座 |
| CP | 恰好复制有效尾部、长度 2048 |
| LFM | 长度、扫频方向、模板时移索引、纯噪声场景 |
| 静态多径 | CP 有效时 $H_g$ 非对角能量为浮点误差级 |
| 仿射时缩 | 至少两条不同非零 epsilon 路径；$\nu_{\ell,q}$ 随频率变化 |
| 独立模型一致性 | 波形计算再 FFT 与 $H_gX_g$ 相符；默认 8192 长度 |
| CP 支撑 | 正负时缩、最后一个 OFDM 块、越界拒绝 |
| 导频消除 | 强导频泄漏场景仍满足 $y=H_{DD}x+w_D$ |
| 噪声 | 实/虚方差、FFT 后复方差、频点交叉协方差在统计容差内 |
| SNR | identity-AWGN 理论 BER 与 Monte Carlo 相符，按样本数设容差 |
| 数据重放 | 同一 manifest/种子/环境可重建 |
| split | 物理帧、信道实例、SNR 副本不跨训练与测试 |

统计测试采用固定种子和由样本数解释的容差，避免凭单次噪声观测判定失败。

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



## Source lines 1386–1443

## 18. 图表、表格与自动分析报告

评测命令生成真实计数和逐帧汇总；报告命令只能读取这些结果，不重新训练或调参。

必须输出：

```text
results/<run_id>/
  resolved_config.yaml
  environment.json
  dataset_manifest_hashes.json
  checkpoint_metadata.json
  per_frame_metrics.csv
  aggregate_metrics.csv
  timing.csv
  diagnostics.jsonl
  figures/
    ber_<scenario>.png
    ser_<scenario>.png
    nmse_<scenario>.png
    latency.png
    stability.png
    learned_thresholds.png
    learned_mu.png
    effective_edges.png
    channel_ici_example.png
    waveform_and_sync_check.png
  REPORT.md
```

主要 CSV 列：

```text
run_id, algorithm, scenario, esn0_db, device, dtype,
train_seed, test_seed, checkpoint_hash, manifest_hash,
n_frames, n_blocks, n_bits, n_symbols,
bit_errors, symbol_errors, block_errors, frame_errors,
ber, ber_ci_low, ber_ci_high, ser, nmse_linear, nmse_db, evm_pct,
bler, fer, goodput_bps,
mean_latency_ms, median_latency_ms, p95_latency_ms,
batch_size, cpu_threads, timing_mode,
hard_failures, message_reject_rate, precision_cap_rate,
status
```

`REPORT.md` 必须说明：

1. 信号参数、400/64/47/1 分配、无编码、理想 I/Q/同步与真实信道假设。
2. 实际环境、训练步数、数据规模、训练种子数和基线实现。
3. 三算法的性能、时间和稳定性差异，支持时给出配对区间。
4. PG-VAMP 的门限是否真正变化、门控是否接近全图、安全项是否过度保守。
5. 是否存在训练分布外性能下降、消息拒绝过多或数值失败。
6. 结果支持什么、不支持什么；缺少的数据明确写“未执行”，不填推测值。

不预先写“PG-VAMP 显著优于其他算法”。NMSE 更好但 BER 无改善、BER 相近但耗时更高，也都是有效结论。

对于需要插值得到的“某 BER 下 SNR 增益”，只有两条曲线都实际覆盖目标 BER 且误差数量充分时才计算；不在零错误区或观测范围外外推。
