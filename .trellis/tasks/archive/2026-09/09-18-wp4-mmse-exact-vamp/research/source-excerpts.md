# WP4 authoritative source excerpts

Verbatim excerpts from docs/CODEX_ENGINEERING_SPEC.md. Source wins in any conflict.
SHA-256: A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B.
Read source sections 10.5, 15.2, 17.6 and 20 directly as needed; this excerpt avoids injection truncation.

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

## 12. 基线一：传统 MMSE，准确命名为线性 MMSE

### 12.1 数学定义

对于单位能量、零均值、独立符号协方差 $I$：

$$
\boxed{\hat x_{LMMSE}=(H^HH+\sigma^2I)^{-1}H^Hy.}
$$

这是线性 MMSE 估计，不是对所有 QPSK 联合状态求和得到的精确非线性 MMSE。图例写 **`MMSE (linear)`**。

实现用 Cholesky 解方程：

```python
G = H.mH @ H + sigma2[:, None, None] * I
rhs = (H.mH @ y.unsqueeze(-1))
L = torch.linalg.cholesky(G)
x_soft = torch.cholesky_solve(rhs, L).squeeze(-1)
```

允许在理论 Hermitian 矩阵上做浮点级对称化；不修改原始 $H$。禁止调用 `inverse`/`torch.linalg.inv`。

### 12.2 输出约定

`x_soft` 为原始 LMMSE 线性估计；硬判决为到四点 QPSK 的最近邻。

本基线默认 `probabilities=None`。不额外附加 QPSK denoiser 后仍称为原始 MMSE，也不将任意 softmax 距离当作精确后验概率。三算法共同指标不要求 NLL 或概率校准。

### 12.3 可选等价实现

可提供 SVD 形式以复用静态 $H$ 的分解，但须通过 Cholesky 等价测试，并在时间报告中说明使用哪一种实现以及是否计入分解成本。不能专门选择低效 MMSE 实现来衬托 PG-VAMP。

---

## 13. 基线二：传统 VAMP，必须使用精确 LMMSE 模块

VAMP 的两模块结构和精确线性更新依据 [R1]；本工程采用下述复数散度/QPSK 约定，并把循环起点固定在线性模块。

### 13.1 初始化与次数

$$
r_2=0,\qquad\gamma_2=1.
$$

主比较执行 8 次“线性模块 + QPSK 模块”，对应 PG-VAMP 的 $T=8$。默认无学习参数、无额外阻尼、无按误码停止。

另提供 `vamp_reference_32` 配置作为补充长迭代参考；报告名字必须与主 `VAMP-8` 区分，不能把一次测试里最好的迭代数事后当作默认结果。

### 13.2 精确线性后验

$$
A=\gamma_wH^HH+\gamma_2I,\qquad
b=\gamma_wH^Hy+\gamma_2r_2,
$$

$$
\hat x_2=A^{-1}b,
\qquad\alpha_2=\frac{\gamma_2}{N}\operatorname{tr}(A^{-1}).
$$

这里允许标准精度关系，因为本模块确实是精确匹配 LMMSE：

$$
c=1-\alpha_2,
\qquad r_1=r_2+\frac{\hat x_2-r_2}{c},
\qquad\gamma_1=\gamma_2\frac{c}{\alpha_2}.
$$

### 13.3 正式推荐 SVD 实现，Cholesky 作为独立 oracle

对同一个 $H$ 分解一次：

$$
H=U\operatorname{Diag}(s)V^H,\qquad\lambda_i=\gamma_ws_i^2.
$$

令 $\tilde y=U^Hy$，$\tilde r=V^Hr_2$，则：

$$
\Delta=V\left[\frac{\gamma_ws_i(\tilde y_i-s_i\tilde r_i)}{\lambda_i+\gamma_2}\right]_{i=1}^N,
\qquad\hat x_2=r_2+\Delta,
$$

$$
\alpha_2=\frac1N\sum_i\frac{\gamma_2}{\lambda_i+\gamma_2},
\qquad c=\frac1N\sum_i\frac{\lambda_i}{\lambda_i+\gamma_2}.
$$

分别求和计算 $\alpha_2$ 与 $c$，避免 $1-\alpha_2$ 的相消；$r_1=r_2+\Delta/c$。

使用 `torch.linalg.svd(H, full_matrices=False)`，不训练 SVD 分解。每次检测所需的 SVD 成本必须计入冷启动延迟；若跨多个观测复用，则另外报告摊销场景。[R1]

### 13.4 非线性模块与数值保护

使用第 14.6 节相同的解析 QPSK 后验和非线性外信息规则，但不能引入 PG-VAMP 的门控、安全预条件或训练参数。

负候选精度或非法消息时保留上一有效 $(r_2,\gamma_2)$，并记录次数。报告名注明 **“VAMP，带显式数值保护”**；不把保护机制说成新的传统 VAMP 收敛定理。

$H=0$ 或数值无信息时返回均匀 QPSK 概率和零软符号，不用极小分母伪造信息。

最后输出 QPSK 后验均值和概率，而不是外信息。

### 13.5 重要比较预期

在有限深度下，PG-VAMP 不一定优于精确 VAMP。精确 VAMP 的线性模块更准确，SVD 还可能使其多轮迭代更高效。结构化 OFDM 信道也不自动满足 VAMP 状态演化所需的矩阵分布假设。[R1]

工程必须允许并如实报告“PG-VAMP 没有获得增益”的结果，不能通过弱化传统 VAMP 来保证新算法胜出。

---


### 14.6 QPSK 后验与非线性外信息

$$
u_R=\sqrt2\gamma_1\operatorname{Re}r_1,
\qquad u_I=\sqrt2\gamma_1\operatorname{Im}r_1,
$$

$$
\hat x_1=\frac{\tanh u_R+j\tanh u_I}{\sqrt2},
\qquad v_i=\frac12(\operatorname{sech}^2u_{R,i}+\operatorname{sech}^2u_{I,i}),
$$

$$
\bar v=\frac1N\sum_i v_i,\qquad\alpha_1=\gamma_1\bar v.
$$

上式第一行中的变量为 **`u_R`** 与 **`u_I`**，与路径频移 $\nu_{\ell,q}$ 无关。实现统一使用变量名 `u_real`、`u_imag` 避免字体混淆。

稳定计算：

$$
\operatorname{sech}^2u=\frac{4e^{-2|u|}}{(1+e^{-2|u|})^2}.
$$

四类概率：

$$
\pi_{ia}=\operatorname{softmax}_{a\in\mathcal S}\{-\gamma_1|r_{1i}-a|^2\}.
$$

候选外信息：

$$
r_{2,cand}=\frac{\hat x_1-\alpha_1r_1}{1-\alpha_1},
\qquad\gamma_{2,cand}=1/\bar v-\gamma_1.
$$

候选均值有限、分母 $1-\alpha_1\ge10^{-6}$、候选精度有效且至少 $10^{-10}$ 时接受；否则保留上一有效消息并记录拒绝。

超过精度上限 $10^8$ 的有效候选按**保持候选均值不变**的方式截断。$\bar v=0$、非有限候选或未定义除法必须走显式保护分支，不允许生成 NaN 后依赖 `torch.where` 隐藏；统一采用保守拒绝并记录 `posterior_variance_underflow`。

不要裁剪真实 $\alpha_1$ 后仍叫作解析散度。无信息输入的输出为零均值、均匀概率。

### 14.7 完整循环与输出

```text
r2 = 0; gamma2 = 1
for t in range(T):
    monotone rho[t], mu[t]
    soft mask -> energy compensation -> safe majorizer
    centered linear estimator
    exact conditional divergence + variance calibration
    QPSK posterior mean, probabilities, variance, true alpha1
    if not last layer:
        nonlinear extrinsic candidate -> accept/reject/precision cap
return final QPSK posterior mean and probabilities
```

最终输出不是外信息均值。调试时允许保留各层输出，常规推理默认不保存所有大中间矩阵。


### WP4：传统 MMSE 与传统 VAMP

实现 Cholesky MMSE、精确 SVD VAMP、独立 Cholesky VAMP oracle、共同 QPSK 模块和消息保护。

**验收：** 与显式小矩阵线性求解相符；SVD/Cholesky VAMP 层级输出相符；零/秩亏信道正确；无训练参数。


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
