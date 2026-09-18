# WP6 authoritative source excerpts

Verbatim excerpts, not a replacement specification. Source: docs/CODEX_ENGINEERING_SPEC.md.
Source SHA-256: a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b
Read the full source sections on demand; these ranges prevent context injection truncation.

## Source lines 1130-1204

## 15. 训练、设备、检查点与复现

### 15.1 仅训练 PG-VAMP

传统 MMSE/VAMP 不训练。PG-VAMP 的默认监督训练损失固定为：

$$
\mathcal L=\sum_{t=1}^{T}\omega_t\frac1{BN}
\sum_{b=1}^{B}\|\hat x_{1,b}^{(t)}-x_b\|_2^2,
\qquad\omega_t=\frac{2t}{T(T+1)}.
$$

代码采用 `(xhat-x).abs().square().mean()`，不要直接把复杂张量传给未核验的实数 MSE 实现。

初始配置：Adam、学习率 $10^{-3}$、weight decay=0、固定梯度范数上限 5.0、按验证集末层 NMSE 选最佳 checkpoint。训练失败和保护触发不得通过删除样本隐藏。

### 15.2 CPU 默认，GPU 显式选择

```text
--device cpu         # 默认，即使有 GPU 也不自动切换
--device cuda:0      # 用户显式启用
--dtype complex128  # 默认正确性精度，配 float64 实参数
--dtype complex64   # 显式效率模式，配 float32 实参数
```

不使用 `device = 'cuda' if available else 'cpu'` 作为默认策略。请求 CUDA 但不可用时明确报错，不静默回退 CPU。

默认 CPU batch size=1，`num_workers=0`，CPU 线程数为 `min(4, os.cpu_count() or 1)`，进程启动时设定并记录。不得把硬件线程数差异当作算法增益。

所有常量、单位矩阵、buffer 和标量需跟随输入 device/dtype。PyTorch 支持复数张量的相关自动微分与复 Hermitian 系统的批量求解；具体设备兼容性必须由本项目实际测试确认。[R2]

训练和推理都不能强制依赖 CUDA。第一版 GPU 指 CUDA；MPS、多卡和分布式不属于验收范围。

### 15.3 复现与安全加载

设置 Python、NumPy、PyTorch CPU 和已启用 CUDA 的随机种子；记录确定性选项、线程数、设备信息和实际软件版本。PyTorch 不保证不同版本、平台或 CPU/GPU 完全逐位一致，因此只承诺在记录环境下的可复现实验，并以合理误差测试跨设备一致性。[R7]

checkpoint 至少包含：

```text
schema_version
model_state_dict
algorithm_config + waveform_config + subcarrier_mapping
optimizer_state_dict
step, epoch, sampler_position
random_seed + Python/NumPy/Torch RNG states
best_validation_metric
training_manifest_hash, validation_manifest_hash
dtype, device_used_for_training, mask_mode
QPSK class/bit mapping
Python/PyTorch/NumPy versions and code commit/hash
```

RNG 状态采用可安全恢复的张量/基础类型保存；加载不可信 `.pt` 文件不启用任意 pickle 执行。默认先 `map_location="cpu"` 加载，再按显式配置转设备。

恢复训练必须包含优化器状态、采样进度和随机状态，不能仅加载权重后称为严格恢复。

### 15.4 必须记录的训练诊断

每次日志至少记录 loss、验证 NMSE、各层 $\rho_t,\mu_t$、有效边比例、$d,\ell$ 的量级、$c,\alpha_1,\alpha_2$、精度范围、拒绝率、无信息率、精度截断率、梯度范数和耗时。

训练日志为 JSONL；checkpoint 单独存储。非有限数、Cholesky 失败必须输出样本 ID、dtype、矩阵范数和关键状态后停止或进入显式失败处理，不能 `nan_to_num` 后继续生成正常结果。

### 15.5 默认工作规模分层

| Profile | 物理/数学规模 | 用途 |
|---|---|---|
| `smoke_math` | $N=32,T=2$，2 次更新 | 仅代数与梯度连通性，不能用于通信性能结论 |
| `smoke_system` | 512 网格、400 数据、$T=8$，少量帧，2 次训练更新 | 完整尺寸链路与 CLI 验收 |
| `cpu_dev` | 128 训练帧、32 验证帧，最多 300 更新 | 默认开发闭环，不作可靠低 BER 声明 |
| `main` | 1024 训练帧、128 验证帧，最多 5000 更新 | 可复现主实验启动配置；用户可改规模 |

`cpu_dev` 和 `main` 都默认 CPU。`main` 并不自动切换 GPU，不承诺这些训练步数已收敛。

验证频率、验证最大块数、batch size 和 early stopping 规则放入配置。主测试只能在选择好 checkpoint 后运行；不能用测试 BER 选择 checkpoint。

## Source lines 1740-1814

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

## Source lines 1875-1879

### WP6：训练、保存、恢复与推理

完成 CPU 默认训练、显式 CUDA、损失、日志、checkpoint、安全恢复与无标签推理。

**验收：** 数学 smoke 和真实尺寸 smoke 均完成；保存/加载一致；恢复训练继续更新；CPU-only 环境不调用 CUDA。

## Source lines 1946-1972

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

