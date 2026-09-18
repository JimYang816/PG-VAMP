# WP7 source excerpts (verbatim)

Source: docs/CODEX_ENGINEERING_SPEC.md
SHA-256: a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b

Exact excerpts for bounded context injection; original specification remains authoritative. Read additional referenced sections directly when needed.

## 16. 统一比较协议

### 16.1 主实验条件

固定条件：

```text
front_end       = ideal_complex_iq
sync_mode       = oracle_timing
csi_mode        = perfect
noise_known     = true
coding          = none
receive_rows    = data_q
PG-VAMP depth   = 8
VAMP iterations = 8
MMSE            = one exact linear estimate
```

三个算法使用相同的 dtype、设备和 CPU 线程数。第一版主正确性结果使用 complex128；complex64 结果另设实验标签，不跨精度直接比较延迟。

### 16.2 训练与测试分布

训练建议混合：static 10%、mild 20%、moderate 50%、strong 20%，`Es/N0` 在 [-5,25] dB 连续均匀采样。identity-AWGN 主要作为校验，不靠大量理想样本掩盖 ICI 场景。

主评测场景为第 6.1 节的五种信道，SNR 网格：

```text
[-5, 0, 5, 10, 15, 20, 25] dB
```

测试每帧 8 个 OFDM 块使用同一个 SNR，以便定义 FER。不同 SNR 可以复用相同物理帧/数据，实现配对分析；噪声默认使用不同 SNR 子种子。若使用同一个标准噪声再按 SNR 缩放，必须在 manifest 声明，不能视各 SNR 结果完全独立。

`cpu_dev` 默认只测 identity、static、moderate，SNR 为 [0,10,20] dB，每格 16 个独立帧，结果标注 `development_only`。

`main` 初始设置每个场景/SNR 格 256 个独立帧，即每格 1,638,400 个数据比特。该数量只是启动值，低 BER 结论是否充分取决于实际错误数和置信区间。

可配置多个训练种子重复 PG-VAMP 训练；至少能从 CLI 指定不同种子并合并结果。仅运行一个种子时必须标明，不能伪造均值和标准差。

### 16.3 禁止的不公平做法

不得使用以下做法：只给 PG-VAMP 完整 $H$、只给 MMSE 对角 $H$；PG-VAMP 用真实噪声而 VAMP 用错误噪声；各算法使用不同导频消除；只为一个算法选择有利测试样本；在测试集调深度、门限或 checkpoint；遗漏 VAMP 的高效精确 SVD 实现；只计新算法在线时间却给基线计入不相同的数据加载成本。

复杂度比较区分固定深度与固定计算预算。主表的“8 层 vs 8 次迭代”是可解释的固定轮数对比，不代表相同 FLOPs。

### 16.4 不同测试规模的停止规则

默认按预先固定的帧数评测，不按某个算法先达到指定错误数而单独停止。流式统计保证三个算法最终样本集合相同。

后续增加自适应采样时必须记录停止规则和样本数，并处理其对统计解释的影响。第一版不通过动态停止制造“每条曲线样本数不一样但看起来可比”的结果。

---

## 17. 指标、统计与运行时间

### 17.1 主指标定义

**BER：**

$$
\mathrm{BER}=\frac{\sum_{b,m,i,j}\mathbf1\{\hat b_{b,m,i,j}\ne b_{b,m,i,j}\}}
{2N_dMF}.
$$

$F$ 为当前统计格的帧数，$M=8$，每个 QPSK 符号 $j\in\{R,I\}$。只计数据比特，不计导频、同步或 CP。

**SER：**

$$
\mathrm{SER}=\frac{\#\{\text{数据符号类别错误}\}}{N_dMF}.
$$

**符号 NMSE：**

$$
\mathrm{NMSE}=\frac{\sum\|\hat x_{soft}-x\|_2^2}{\sum\|x\|_2^2},
\qquad\mathrm{NMSE}_{dB}=10\log_{10}\mathrm{NMSE}.
$$

这里估计对象是数据符号 $x$，不是信道 $H$；不把它写成信道估计 NMSE。先在全体样本上累加分子分母，再计算 dB，不平均各样本的 dB 值。

**EVM：**

$$
\mathrm{EVM}_{rms}(\%)=100\sqrt{\mathrm{NMSE}}.
$$

在这里的统一定义下，EVM 与 NMSE 是同一误差的不同表达，不把它们当作两项独立的性能证据。不允许用测试标签对输出做事后幅相对齐来改善 EVM。

**OFDM 块错误率与帧错误率：**

$$
\mathrm{BLER}=\frac{\#\{\text{至少一个数据比特错误的 OFDM 块}\}}{MF},
$$

$$
\mathrm{FER}=\frac{\#\{\text{8 个块中至少一处数据比特错误的帧}\}}{F}.
$$

**按整帧验收的 goodput：**

$$
R_{good}=\frac{6400}{0.956}(1-\mathrm{FER})\quad\mathrm{bit/s}.
$$

这是“整帧出错即丢弃”的仿真净有效速率，包含固定物理帧开销；不表示已经实现 ARQ、CRC 或 MAC。

### 17.2 时间与计算吞吐率

至少报告：

- 单个 400 维 OFDM 检测块的 mean、median、p95 延迟。
- 固定 batch size 下的检测吞吐率，单位 blocks/s 和 data bits/s。
- 同步/FFT/导频消除等共同前处理成本，另列一项。
- 峰值 CPU 进程内存与可用时的 CUDA 峰值已分配显存；二者指标口径不同，不直接相减。
- 可学习参数数目；PG-VAMP 为 16，MMSE/VAMP 为 0。另列矩阵工作内存，不以参数数替代内存。

检测计算吞吐率 $800\times\text{blocks/s}$ 是处理器处理速度，与声学链路 goodput 不是同一个量。

### 17.3 冷启动、复用与计时边界

提供两种明确命名的时间口径：

| 口径 | 含哪些成本 |
|---|---|
| `per_observation_cold_H` | 输入已在设备上；包含该次 $H$ 所需的 SVD/Cholesky、图构建、全部层和判决 |
| `same_H_amortized` | 同一 $H$ 处理多个观测；允许缓存真正与观测无关的计算，单独报告首次预处理和后续成本 |

PG-VAMP 即使图可缓存，$\gamma_2$ 随层消息变化，通常不能缓存所有 $\bar P$ 因子；训练中也不能缓存依赖旧参数的图并复用反向图。

VAMP 对同一 $H$ 的 SVD 可复用，但一帧内不同 $T_m$ 的 $H(m)$ 不相同，不能假装只分解一次就处理整帧所有时变矩阵。

常规基准使用 `.eval()` 和 `torch.inference_mode()`，关闭大量诊断拷贝和日志；计时预热后重复多次。CPU 使用 `perf_counter` 或 PyTorch benchmark；CUDA 在计时边界同步或使用正确同步的 CUDA events。

批量延迟除以 batch size 只能叫摊销每块时间，不能当作 batch size=1 的单块在线时延。图表必须列出 batch size、dtype、设备、线程数、预热次数和重复次数。

不把数据生成、绘图、磁盘 IO 或 checkpoint 加载只计入某一个算法；端到端时间若包含这些成本则三者一致。

### 17.4 数值稳定性与模型诊断

三个算法记录最终非有限输出、分解失败和不可用样本数；VAMP/PG-VAMP 额外记录消息拒绝率、精度截断率和无信息率。

PG-VAMP 额外记录每层门限、$\mu$、有效边比例、安全项相对大小和 $c$。有效边比例统一为非零非对角候选边中 $m\ge0.5$ 的比例，并另外给出相对于全部 $N(N-1)$ 位置的比例；不要让分母改变掩盖稀疏程度。

记录信道 ICI 比例：

$$
\eta_{ICI}=\frac{\|H_g-\operatorname{Diag}(\operatorname{diag}H_g)\|_F^2}
{\max(\|H_g\|_F^2,\varepsilon)}.
$$

完整网格和 400 维有效矩阵的 ICI 比例可分别记录，名字必须区分。

### 17.5 置信区间和零错误

OFDM/同帧内错误可能相关。主 BER/SER 的 95% 区间用**以独立帧为簇的 bootstrap**：保留每帧错误数和分母，重采帧，重新计算总错误率。算法差值的区间采用同一组重采帧，实现配对比较。默认 2000 次重采，固定 bootstrap 种子。

若只有极少帧，应标注区间不稳定。一个独立帧就是一个簇，不能把同帧 6400 个比特当作 6400 个独立信道试验。

观察到零错误时：保留真实值 0 和“0 / 已测比特数”；log 图用特殊标记及文字说明，显示用占位值不能回写 CSV。全零 bootstrap 区间不能解释为已经证明真实 BER 为零。

可对独立帧的零帧错误事件给出单侧 95% FER 上界：

$$
p_{FER,upper}=1-0.05^{1/F}.
$$

不在存在相关位错误的情况下，把 `3 / bit_count` 无条件当作严格 BER 上界。

### 17.6 失败处理

合法的消息拒绝/保持上一消息属于算法定义，正常计入检测结果。

最终 NaN、Inf、分解失败属于硬失败。默认将该统计格标为 `incomplete_or_failed`，保留失败样本 ID，不能只删除失败块后继续输出一个看似完整的 BER。可以另报“成功块条件 BER”，但不得冒充全体 BER。

同步模式下同时报告全体 FER 与同步成功条件下的 BER；不得默默丢掉漏检帧。

---

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

---

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

---

### WP7：统一评测与报告

实现配对样本评测、计数、bootstrap、计时、稳定性记录、图表和自动 Markdown 报告。

**验收：** 手工构造的预测可得到正确 BER/SER/FER；零错误与硬失败没有被隐藏；三算法使用相同数据哈希；报告只引用真实结果。

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
