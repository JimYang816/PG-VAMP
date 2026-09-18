# WP3 source excerpts (verbatim)

Authority: docs/CODEX_ENGINEERING_SPEC.md. SHA-256: a159d20380d4f48785fac15a02b20247d681b4e078a9dd7f8046fdd1f66ac47b.
Extracted for bounded context injection; original source wins. Before implementation/check, read relevant full source sections including physical §§5–9, runtime §15.2–15.3 and config §20; this extract does not replace them.

## Source lines 666–798

## 10. 数据集设计：不要默认把全部稠密矩阵存入磁盘

### 10.1 一个样本与一帧的关系

一个检测样本是一个 OFDM 数据块：

```text
H       complex [400,400]
y       complex [400]
sigma2  real scalar > 0
x       complex [400]        # 训练/有标签评测使用
bits    uint8 [400,2]        # 训练/有标签评测使用
```

一帧包含 8 个样本，它们共享物理路径参数，但对应不同的 $T_m$、数据和噪声，因此 $H(m)$ 一般不同。

### 10.2 按物理帧划分训练、验证和测试

先划分物理帧/信道实例，再展开为 OFDM 样本。同一帧的不同符号、同一路径实例的不同 SNR 副本，不得跨越训练/验证/测试集合。

独立随机流至少包括：`split`、`channel`、`bits`、`pilots`、`noise`、`arrival_offset`。种子用 SHA-256 等稳定散列从主种子与元数据派生，不使用 Python 的进程随机 `hash()`。

测试三算法读取相同的 `sample_id`、$H$、$y$、$\sigma^2$ 和比特。禁止每个算法重新随机生成自己的测试样本。

### 10.3 默认紧凑记录格式

采用版本化 manifest + `.pt` 分片。分片仅包含张量和基础 Python 类型，不序列化自定义类对象。

```text
data/<dataset_name>/
  manifest.json
  train/frame_records_000.pt
  val/frame_records_000.pt
  test/frame_records_000.pt
  audits/                         # 少量完整波形和一致性检查
```

每帧记录：

```text
schema_version
frame_id, channel_id, split, scenario
path_count
path_gain_complex[L]
path_delay_s[L]
path_epsilon[L]
data_bits[8,400,2]
pilot_symbols[8,64] or pilot_seed
esn0_db[8] or explicit sweep specification
noise_seed_real[8], noise_seed_imag[8]
arrival_offset_samples
waveform_config_hash, generator_version
cp_valid, generation_rejections
```

manifest 保存完整通信参数、频点索引、生成分布、拆分规则、种子、软件版本、文件校验和、样本数量及 SNR 定义。

`EffectiveDataset.__getitem__` 从物理记录确定性构造 $H$、导频贡献和 $y$。生成操作在 `no_grad` 下进行，输出 $H$、$y$、$\sigma^2$ 不带可学习参数。

默认 `num_workers=0`，按需用小型 LRU 缓存保存最近的矩阵。缓存键必须包含帧、符号时刻、模型版本、频点映射和 dtype，不能只用 `frame_id`。

### 10.4 为什么默认不用全量矩阵文件

一个 $400\times400$ 的 complex128 矩阵占：

$$
400^2\times16=2{,}560{,}000\ \text{bytes}.
$$

8192 个这样的矩阵约需 20.97 GB，尚未包含完整网格矩阵、波形和其他字段。大量样本优先保存物理参数与种子。

仍须支持显式物化格式，便于与旧 PG-VAMP 接口对接：

```python
{
    "schema_version": 1,
    "H": ...,        # [K,N,N] complex
    "y": ...,        # [K,N] complex
    "sigma2": ...,   # [K] real
    "x": ...,        # [K,N], 推理文件可缺少
    "bits": ...,     # [K,N,2], 推理文件可缺少
    "metadata": ...,
}
```

物化命令先估算输出字节数，超过用户配置的容量阈值时要求显式 `--allow-large-output`，不能无提示生成几十 GB 文件。

### 10.5 输入校验

检查：形状、方阵维度、dtype 配对、finite、$\sigma^2>0$、QPSK 能量和类别、样本数量、索引一致、配置哈希和 split 无重叠。

推理允许没有 `x`/`bits`；训练和误码评测缺少标签必须报错。普通推理 API 不接收标签。

不在数据加载器中暗中归一化 $H$。显式整体缩放时必须同时变换：

$$
H'=aH,\qquad y'=ay,\qquad\sigma'^2=|a|^2\sigma^2,
$$

并测试预测尺度不变性。

---

## 11. 三算法统一接口

```python
@dataclass
class DetectionResult:
    x_soft: torch.Tensor                 # complex [B,N]
    class_hat: torch.Tensor              # int64 [B,N]
    bits_hat: torch.Tensor               # uint8/int64 [B,N,2]
    probabilities: torch.Tensor | None   # real [B,N,4]
    diagnostics: dict

class Detector(Protocol):
    def detect(
        self,
        H: torch.Tensor,                 # [B,N,N]
        y: torch.Tensor,                 # [B,N]
        sigma2: torch.Tensor,            # [B]
        *,
        return_diagnostics: bool = False,
    ) -> DetectionResult: ...
```

三个算法共享：数据预处理、标签映射、QPSK 判决、复噪声定义、数值 dtype、设备、测试样本与计数器。

三者不共享：可能泄露 PG-VAMP 学习结果的内部状态。VAMP/MMSE 不读取 PG-VAMP checkpoint 的层参数。

`x_target`、`bits`、SNR 类别标签、路径真值不作为检测 API 的额外输入。检测器只获得约定好的 $H,y,\sigma^2$；真实信道信息已经包含在 $H$ 中。

---


## Source lines 1228–1245

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


## Source lines 1763–1777

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


## Source lines 1857–1862

### WP3：数据集与划分

实现紧凑帧记录、manifest、split、随机流、重放、按需矩阵生成、物化格式与输入校验。

**验收：** 相同记录可重建、不同 split 无帧/信道泄漏、不同算法取得完全相同样本、容量预估正确。


## Source lines 1897–1918

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

## Source lines 1946–1974

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
