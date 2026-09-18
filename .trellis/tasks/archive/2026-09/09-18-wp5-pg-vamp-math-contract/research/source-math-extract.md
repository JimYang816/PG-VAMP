# Verbatim specification excerpt for WP5 context

Source: `docs/CODEX_ENGINEERING_SPEC.md`; SHA-256 `A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`.
Extracted without rewriting on 2026-09-18 to avoid context-injection truncation.
The original remains authoritative; recheck its hash before implementation.

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


## 14. PG-VAMP-VC：核心数学合同

本节就是本工程 **PG-VAMP-VC 的正式算法合同**。Codex 必须直接按本节实现；不需要再读取其他执行书或从外部文档补齐公式。默认通信配置下，$N$ 为预处理后的数据维度 400。

### 14.1 只有 $2T$ 个可学习标量

```text
raw_gaps[T]
raw_mu[T]
```

默认 $T=8$，恰好 16 个可学习实标量。不学习噪声方差、信道、QPSK 星座、任意矩阵、GNN 或 MLP。

令 $\Delta_\rho=0.5$ dB，$\rho_{hi}=0$ dB，$\rho_{lo}=-60$ dB，$R=\rho_{hi}-\rho_{lo}-T\Delta_\rho>0$：

$$
d_t^{gap}=\operatorname{softplus}(a_t),
\qquad
\rho_t=\rho_{hi}-t\Delta_\rho
-R\frac{\sum_{j=1}^{t}d_j^{gap}}{1+\sum_{j=1}^{T}d_j^{gap}},
\qquad \mu_t=\operatorname{sigmoid}(v_t).
$$

公式采用 $t=1,\ldots,T$；代码用 `t+1` 对齐。初始 $d_t^{gap}=1/T$，初始 $\mu_t=0.8$，用稳定的逆 softplus/logit 设置原始参数。

### 14.2 图分数与软门控

$$
p_{ki}=\frac{|H_{ki}|^2}{\sum_j|H_{ji}|^2},
\qquad s_{ki}=10\log_{10}\max(p_{ki},10^{-12}),
$$

$$
m_{ki}^{(t)}=\operatorname{sigmoid}\left(\frac{s_{ki}-\rho_t}{\tau_g}\right),
\qquad\tau_g=3\ \mathrm{dB}.
$$

主对角 $m_{ii}=1$；零幅度非对角边置零；零能量列按原数学规范处理。$H$ 和 $M$ 一般都不对称，不能人为对称化。

训练和推理均默认 `mask_mode=soft`。本版主评测不启用 hard mask。

### 14.3 能量补足与安全矩阵

$$
H_D=M\odot H,
\qquad d_i=\sum_k(1-m_{ki}^2)|H_{ki}|^2,
$$

$$
G=H_D^HH_D+\operatorname{Diag}(d),
$$

$$
\ell_i=\sum_k\sum_{j\ne i}(1-m_{ki}m_{kj})|H_{ki}||H_{kj}|,
$$

$$
\bar G=G+\operatorname{Diag}(\ell),
\qquad\bar P=\gamma_w\bar G+\gamma_2I.
$$

必须保留：

$$
\operatorname{diag}(G)=\operatorname{diag}(H^HH),
\qquad\bar P\succeq A=\gamma_wH^HH+\gamma_2I\succ0.
$$

$d$ 不能写成 $(1-m)^2$ 的能量和；$d,\ell$ 也不能额外加进 $\sigma^2$。

$\ell$ 用非负 $O(N^2)$ 等价式实现。令 $F=|H|$、$U=M\odot F$、$R_F=(1-M)\odot F$：

$$
\ell_i=\sum_k\left[(R_F)_{ki}\sum_{j\ne i}F_{kj}
+U_{ki}\sum_{j\ne i}(R_F)_{kj}\right].
$$

不写三重 Python 循环；不要把 $d,\ell$ 的二次成本误称为整个稠密算法只有二次成本。

### 14.4 中心化线性估计与完整信道残差

令 `solve_P` 使用当前层同一次 Cholesky 分解，定义：

```text
apply_A(V) = gamma_w * H^H @ (H @ V) + gamma2 * V

u          = gamma_w * H^H @ (y - H @ r2)
q0         = solve_P(u)
e0         = u - apply_A(q0)
innovation = q0 + mu * solve_P(e0)
xhat2      = r2 + innovation

apply_B(V):
    q = solve_P(V)
    return q + mu * solve_P(V - apply_A(q))
```

这对应：

$$
B=(1+\mu)\bar P^{-1}-\mu\bar P^{-1}A\bar P^{-1},
\qquad\hat x_2=r_2+B\gamma_wH^H(y-Hr_2).
$$

禁止改回零中心 `solve_P(b)`；禁止仅用 $H_D$ 计算真实残差；禁止将内部完整系统求到收敛后仍保留原创新解释。

### 14.5 实际算子的散度与方差校准

```text
W       = apply_B(gamma_w * H^H)
F2      = W @ H
c       = real(trace(F2)) / N
alpha2  = 1 - c
K       = W / c
r1      = r2 + innovation / c
var1    = ||I - K@H||_F^2 / (N*gamma2) + sigma2*||K||_F^2/N
gamma1  = 1 / var1
```

矩阵和均值创新量都必须对应同一个实际 $\bar P$，一次分解、多右端项复用。精确 trace/Frobenius 范数为第一版默认，禁止随机迹估计。

不允许：

$$
\alpha_2\leftarrow\gamma_2\operatorname{tr}(B)/N
\quad\text{或}\quad
\gamma_1\leftarrow\gamma_2(1-\alpha_2)/\alpha_2.
$$

这两项不是本版中心化近似模块的正确组合。

$c$ 在工作精度下无信息时，返回 `r1=0, gamma1=0`，并计数。不能简单 `clamp(c)` 后继续冒充有信息观测。判断阈值须由 dtype、矩阵量级和原规范明确给出，不能无说明地增大到影响正常样本。

上述方差校准采用以下局部二阶矩模型：

$$
r_2=x+v,\qquad \mathbb E[vv^H]=\gamma_2^{-1}I,\qquad
\mathbb E[ww^H]=\sigma^2I,\qquad\mathbb E[vw^H]=0.
$$

在固定当前层 $H,\gamma_2,M,\mu$ 的条件下，平方和公式与该局部模型一致；但实际多层消息在有限维网络中不保证严格白化或与观测噪声严格零互协方差，因此它是本算法定义采用的局部方差校准规则，而不是任意有限维系统的状态演化定理。

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

### 14.8 自动微分与禁止项

局部条件散度的定义固定当前层 $H,\gamma,\rho,\mu$，但整网损失的梯度仍通过它们的实际依赖传播。

不能在模型前向中随意 `.detach()` 精度、门限、mask、Cholesky 因子或方差。仅日志副本允许 `detach()`。

第一版禁止：有限步 CG 替换、根据右端项提前停止、跨层线性求解 warm start、随机 trace、AMP/float16、静默 hard-mask 推理、未经说明的 jitter。

若确需为 $\bar P$ 加非负 jitter，必须作为显式配置记录，并使所有 solve、算子、散度和方差使用同一个实际矩阵；不能给真实 $A$ 暗中加 jitter 后继续引用原模型。

### 14.9 复杂度声明

本版仍包含稠密 Gram 构造、Cholesky、多右端项解和精确方差，单层可能为 $O(N^3)$，存储至少为 $O(N^2)$。只有 16 个可学习标量不等于计算轻量。

软图的有效边比例只是诊断指标，不代表已经使用稀疏存储或获得实际稀疏加速。

---


### 17.4 数值稳定性与模型诊断

三个算法记录最终非有限输出、分解失败和不可用样本数；VAMP/PG-VAMP 额外记录消息拒绝率、精度截断率和无信息率。

PG-VAMP 额外记录每层门限、$\mu$、有效边比例、安全项相对大小和 $c$。有效边比例统一为非零非对角候选边中 $m\ge0.5$ 的比例，并另外给出相对于全部 $N(N-1)$ 位置的比例；不要让分母改变掩盖稀疏程度。

记录信道 ICI 比例：

$$
\eta_{ICI}=\frac{\|H_g-\operatorname{Diag}(\operatorname{diag}H_g)\|_F^2}
{\max(\|H_g\|_F^2,\varepsilon)}.
$$

完整网格和 400 维有效矩阵的 ICI 比例可分别记录，名字必须区分。
