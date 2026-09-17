# PG-VAMP CP-OFDM 完整仿真与检测工程：Codex 执行任务书

**版本：v2.0 · 自包含完整工程版**  
**语言与框架：Python + PyTorch**  
**默认运行设备：CPU；训练与推理均须支持显式切换 CUDA GPU**  
**算法：PG-VAMP-VC、传统线性 MMSE、传统 VAMP**  
**文档日期：2026-09-17**

> 本文件是供 Codex 实现的工程规格，不是已经实现、训练或完成性能测试的项目报告。所有通信参数、数据规模与优化器设置，除用户明确指定的部分外，均为第一版的设计启动值，不是已经验证的最优值。
>
> 本版把“512 个频率”明确解释为：**21–27 kHz 设计频带内的 512 个 OFDM 子载波网格位置**，不是在 96 kHz 采样率下直接使用 512 点 FFT，也不是 512 个位置全部承载数据。默认分配 **400 个数据、64 个导频、47 个边缘保护空载波、1 个中心空载波**。
>
> 第一版采用**不加信道纠错编码的 QPSK**。先观察检测器本身的误码率，不让译码器掩盖检测差异。导频、同步头、循环前缀属于物理层开销，不等同于纠错编码。

---

## 0. 文档定位、实施优先级与范围

### 0.1 本文件是唯一工程实施规格

**Codex 构建本工程时，以本文件作为唯一必须读取的实施规格。** 本文已经包含本版需要的通信参数、数据生成规则、PG-VAMP 数学合同、MMSE/VAMP 基线、训练与推理、评测指标、测试要求、工程目录和 CLI 约定。

实施规则：

1. **本文件自包含且优先级最高。** Codex 不需要读取任何其他工程执行书，也不需要做多文档优先级合并。
2. `PG_VAMP_完整数学推导_标准Markdown公式.md` 仅作为 PG-VAMP 理论推导的**审计与溯源材料**；本文件第 14–15 节已经给出实现所需的正式数学合同和训练合同。即使该推导文件不在 Codex 工作区，也不得以此为理由阻塞工程实现。
3. 本版工程从本文定义的目录、reference 正确性实现、测试和 CLI 开始组织；任何工作区已有文件只有在用户明确要求迁移或复用时才纳入本工程。
4. 所有实现决策、验收条件和实验报告都必须能够仅由本文解释，不得存在隐含的外部工程规格。

### 0.2 本版必须完成的范围

本版直接实现以下完整闭环，而不是在旧工程上追加补丁：

- 96 kHz、21–27 kHz、512 频率网格的 CP-OFDM/QPSK 仿真；
- LFM 同步头与基础同步器；
- 多径、路径相关仿射时缩/Doppler、AWGN 的可复现物理参数化仿真；
- `waveform_reference` 与 `effective_fast` 两条一致性数据路径；
- 导频消除与 400×400 数据检测系统构造；
- 传统线性 MMSE；
- 传统 VAMP（精确 LMMSE 线性模块）；
- PG-VAMP-VC；
- PG-VAMP 训练、checkpoint、CPU/GPU 推理；
- 三算法统一 BER、SER、NMSE、EVM、BLER/FER、goodput、延迟和数值稳定性评测；
- 自动测试、图表和 Markdown 结果报告。

PG-VAMP 的中心化更新、解析安全项、实际条件散度、方差校准以及仅 $2T$ 个可学习标量，是本文件中的硬性算法合同，不允许为了兼容任何旧实现而修改。

### 0.3 正确性参考在本工程内新建

本工程必须自行建立 `src/pgvamp_ofdm/reference/dense_pg_vamp.py`，作为**小规模、稠密、可读、优先正确性**的 PG-VAMP 独立参考实现，再用正式模块与其做逐层前向和参数梯度等价测试。

同样建立 `dense_vamp_cholesky.py` 作为传统 VAMP 的独立 Cholesky 参考，用来核对正式 SVD 实现。

这些 reference 文件属于本版工程的一部分，不依赖工作区已有 oracle，也不引用“历史上已经通过多少个测试”的结论。Codex 只能报告本次实际执行得到的测试结果。

### 0.4 第一版结论边界

主比较采用：**仿真真实信道已知、噪声方差已知、理想帧定位、理想复数 I/Q 接收接口、CP 有效**。

它回答的是：在同一可控 ICI 检测问题下，三个检测器的误码、误差和计算成本如何。

它不回答：真实海试表现、未知完整 ICI 信道的估计能力、真实模拟前端的端到端误码率，或带译码反馈的系统性能。LFM 同步模块必须实现，但其性能和检测器主比较分开报告。

## 1. 工程完成目标

实现以下可运行闭环：

```text
随机比特
  -> 固定标签 QPSK
  -> 数据/导频/空载波映射
  -> 96 kHz CP-OFDM 波形 + LFM 同步头
  -> 多径 + 路径相关时缩/Doppler + AWGN
  -> 同步模块 / 理想同步评测模式
  -> 理想 I/Q、去 CP、FFT、明确的频点提取
  -> 已知导频贡献消除
  -> 同一 (H, y, sigma2)
       -> MMSE
       -> VAMP
       -> PG-VAMP-VC
  -> 判决、统一计数、指标、图表、自动报告
```

必须同时提供两种数据路径：

- `waveform_reference`：独立时域/连续时间波形计算，用来验证物理链路及频域模型，并执行小规模完整帧演示。
- `effective_fast`：由同一物理路径参数解析构造完整频域 $H$，快速生成训练和评测样本。不是随意生成一个随机矩阵。

两条路径必须先通过无噪声一致性测试，再允许 `effective_fast` 用于主训练和主评测。

工程运行不能依赖 IDE、MATLAB、专有仿真软件、云服务或真实数据下载。第一版不要求接入真实声卡。

---

## 2. 默认通信参数与“512”的准确解释

### 2.1 参数表

| 参数 | 符号 / 配置名 | 默认值 | 含义 |
|---|---|---:|---|
| 波形采样率 | $F_s$ / `sample_rate_hz` | 96000 Hz | 用户指定的高采样率 |
| 设计频带 | `band_hz` | [21000, 27000] Hz | 正频率通带范围 |
| 中心频率 | $f_c$ / `carrier_hz` | 24000 Hz | 频带中心 |
| 网格带宽 | $B_g$ | 6000 Hz | $27000-21000$ |
| 子载波网格数 | $N_g$ / `n_grid` | 512 | 包含数据、导频与空载波 |
| 子载波间隔 | $\Delta f$ | 11.71875 Hz | $B_g/N_g$ |
| 高采样率 FFT 长度 | $N_w$ / `n_fft_wave` | 8192 | $F_s/\Delta f$ |
| 有效符号时长 | $T_u$ | 85.333333 ms | $1/\Delta f=N_w/F_s$ |
| CP 长度 | $N_{cp}$ / `cp_samples` | 2048 samples | 96 kHz 下的长度 |
| CP 时长 | $T_{cp}$ | 21.333333 ms | $N_{cp}/F_s$ |
| 单个含 CP 符号时长 | $T_{sym}$ | 106.666667 ms | $(N_w+N_{cp})/F_s$ |
| 数据子载波数 | $N_d$ | 400 | 每个 OFDM 符号 800 个数据比特 |
| 导频子载波数 | $N_p$ | 64 | 已知单位能量 QPSK，不额外升功率 |
| 边缘空载波数 | $N_{guard}$ | 47 | 左 24、右 23 |
| 中心空载波数 | $N_{dc}$ | 1 | 复基带 DC，对应通带 24 kHz |
| 调制 | `modulation` | QPSK | 单位符号能量 $E_s=1$ |
| 纠错编码 | `coding` | `none` | 第一版固定为未编码 |
| 每帧数据 OFDM 符号数 | $M$ | 8 | 8 个检测块 |
| 默认检测维度 | $N$ | 400 | **不是 512，也不是 8192** |

子载波、保护带、导频、CP 和过采样应是不同的配置概念，不能混写。[R4]

### 2.2 为什么不能直接设 `nfft=512, fs=96000`

这样会得到：

$$
\Delta f=\frac{96000}{512}=187.5\ \mathrm{Hz},
$$

6 kHz 频带只有约 32 个频率间隔，而不是 512 个。

本版采用：

$$
\Delta f=\frac{6000}{512}=11.71875\ \mathrm{Hz},
\qquad N_w=\frac{96000}{11.71875}=8192.
$$

可将其理解为“6 kHz、512 点复基带网格的 16 倍高采样率表示”，但**正式波形路径直接采用 8192 点 IFFT/FFT**，避免第一版额外引入重采样滤波器和噪声着色。

不要实现一条 512 点低采样率链路后，不经核验就宣称它与 Doppler 下的 8192 点接收算子逐元素完全相同。

### 2.3 为什么先选 400 个数据子载波

本版的工程取舍是：

$$
400+64+47+1=512.
$$

400 个位置传数据，保留导频布局和边缘余量，且把核心检测矩阵控制为 $400\times400$。相对于全 512 维检测，在相同稠密立方阶算子假设下，维度比例对应的代数工作量比例约为：

$$
(400/512)^3\approx0.477.
$$

这只是维度缩减的数量级解释，不是运行时间实测值，也不证明 400 是最优载波数。

64 个导频是为后续信道估计留出的起点，**本版不宣称它们足以从一个 OFDM 符号估计任意稠密 $512\times512$ ICI 矩阵**。主比较使用仿真器提供的完整真实信道。

---

## 3. 频率索引与资源映射：不得自由发挥

### 3.1 三种索引分开保存

用有符号频率编号：

$$
q\in\{-256,-255,\ldots,255\},\qquad f_q=f_c+q\Delta f.
$$

定义：

```text
q                    : 有符号物理网格编号，[-256,255]
grid_index           : q + 256，512 维中心排列中的索引
baseband_fft_index   : q mod 8192，8192 点基带 FFT 的索引
passband_fft_index   : 2048 + q，正频率通带 FFT 的索引
```

所有索引采用 Python 的 0-based 编号。算法矩阵内部按 $q$ 从小到大排序，必须同时保存 `data_q` 和 `pilot_q`。矩阵位置相邻不代表物理频率位置必然连续。

512 个网格点为 $21000$ Hz 至 $26988.28125$ Hz。OFDM 栅格不能同时把 21 kHz、27 kHz 两个端点都当作 512 个等间距点并仍维持上述 $\Delta f$。

### 3.2 固定默认分配

```python
q = torch.arange(-256, 256, dtype=torch.int64)
guard_q = q[(q < -232) | (q > 232)]       # 24 + 23 = 47
active_q = q[(q >= -232) & (q <= 232) & (q != 0)]  # 464
j = torch.arange(32, dtype=torch.float64)
pilot_pos = 1 + torch.floor((j + 0.5) * 232 / 32).to(torch.int64)
pilot_q = torch.sort(torch.cat([-pilot_pos, pilot_pos])).values  # 64
# data_q = active_q 中剔除 pilot_q 后的有序集合，长度 400
```

`pilot_pos` 必须为：

```text
4, 11, 19, 26, 33, 40, 48, 55, 62, 69, 77, 84, 91, 98,
106, 113, 120, 127, 135, 142, 149, 156, 164, 171, 178,
185, 193, 200, 207, 214, 222, 229
```

最外侧非零载波中心为 21281.25 Hz 和 26718.75 Hz。LFM 仍按 21–27 kHz 扫频。

边缘留空不等于严格带限。矩形截断的 OFDM 符号存在频谱旁瓣；必须绘制 PSD 并报告带外能量，不能声称 21–27 kHz 以外能量严格为零。第一版默认不加 OFDM 窗口或模拟带通滤波器；添加这些算子后必须重新验证有效 $H$ 和噪声协方差。[R4]

### 3.3 强制校验

资源分配模块检查：数量相加为 512、集合互不相交、无重复、边界正确、数据/导频非空、排序固定、正负 FFT 索引一致。

数学单元测试可使用 $N=8,16,32$ 的代数 fixture，不必假装它们满足本节的真实通信配置。真实通信 smoke test 必须至少运行一次 512 网格、400 维检测。

---

## 4. QPSK、编码与帧结构

### 4.1 唯一的比特映射

本工程的 QPSK 类别与比特映射固定如下，其他模块不得自行采用另一套标签顺序：

$$
x(b_R,b_I)=\frac{(1-2b_R)+j(1-2b_I)}{\sqrt2},
\qquad c=2b_R+b_I.
$$

| 类别 | 比特 $(b_R,b_I)$ | 星座 |
|---:|---|---|
| 0 | 00 | $(+1+j)/\sqrt2$ |
| 1 | 01 | $(+1-j)/\sqrt2$ |
| 2 | 10 | $(-1+j)/\sqrt2$ |
| 3 | 11 | $(-1-j)/\sqrt2$ |

符号顺序：帧内 OFDM 符号顺序、该符号内 `data_q` 升序、每个符号先 $b_R$ 后 $b_I$。平局时选编号最小的类别。

导频为通过独立固定种子生成、收发双方已知的 QPSK 序列，单位能量，不与数据标签共享随机流。

### 4.2 第一版不加哪些编码

`coding: none`，不实现卷积码、LDPC、Polar、Turbo、译码反馈或比特交织。不得把 QPSK 比特映射叫作纠错编码。

预留 `coding/` 扩展接口即可；未知 `coding` 配置应报错，不可静默忽略。后续加入 FEC 时再明确码率、块长、交织、译码迭代数与 LLR 约定，并同时报告编码前后 BER。

第一版 FER 是通过仿真真实比特判断的帧错误率，并不表示已实现 CRC 检错。

### 4.3 默认完整帧

```text
前置静默 1920 samples (20 ms)
LFM 同步头 3840 samples (40 ms)
同步保护间隔 2048 samples (21.333333 ms)
8 × [CP 2048 samples + OFDM useful 8192 samples]
尾部静默 2048 samples (21.333333 ms)
```

总长：

$$
N_{frame}=91776\ \text{samples},\qquad T_{frame}=0.956\ \mathrm{s}.
$$

每帧有效数据：

$$
N_{bits,frame}=8\times400\times2=6400.
$$

不计同步和帧保护间隔时的数据速率为 7500 bit/s；包含上述全部固定帧开销时，理想无误码载荷速率约为 6694.56 bit/s。两者都不是实际 goodput。

可随机增加 `arrival_offset_samples` 模拟录音前的额外等待。该偏移不计入定义好的发送帧持续时间；记录长度和发送帧长度必须分开。

### 4.4 LFM 定义

令 $0\le t<T_L=0.04$ s，$f_0=21000$ Hz、$f_1=27000$ Hz：

$$
\beta=\frac{f_1-f_0}{T_L}=150000\ \mathrm{Hz/s},
$$

$$
p_a(t)=A\,w(t)\exp\!\left[j2\pi\left(f_0t+\frac{\beta t^2}{2}\right)\right],
\qquad p_{RF}(t)=\sqrt2\operatorname{Re}p_a(t).
$$

$w(t)$ 为固定 Tukey 窗，默认 `alpha=0.1`，并保存其定义与离散长度。$A$ 为固定幅度归一化系数，使离散同步头的平均功率等于配置确定的 OFDM 平均功率，而不是逐帧按照数据标签归一化。线性 chirp 的相位是瞬时频率的积分。[R5]

离散时刻使用 `torch.arange(n_lfm)/sample_rate_hz`，不用包含两个端点的 `linspace(0,T_L,n_lfm)`。

---

## 5. 波形、接收接口与 FFT 归一化

### 5.1 统一采用单位酉 FFT

所有核心 OFDM 变换使用 `norm="ortho"`，保证：

$$
\|Fz\|_2^2=\|z\|_2^2.
$$

PyTorch 官方 FFT 文档规定了 `ortho` 的 $1/\sqrt n$ 归一化。[R3]

将 512 网格符号放到 8192 点复基带频谱：

```python
spectrum = torch.zeros(..., 8192, dtype=complex_dtype, device=device)
spectrum[..., q % 8192] = X_grid
s_bb = torch.fft.ifft(spectrum, dim=-1, norm="ortho")
s_cp = torch.cat([s_bb[..., -2048:], s_bb], dim=-1)
```

不乘未经解释的 16、$\sqrt{16}$ 或 $\sqrt{N_w/N_g}$。本版的单位能量指每个频域 QPSK 符号，而不是每个高采样率时域样本的功率为 1。

有效区间内平均发送复包络功率为：

$$
P_{OFDM}=\frac{N_d+N_p}{N_w}=\frac{464}{8192}.
$$

### 5.2 实通带波形

保存可检查的 96 kHz 实通带发送波形：

$$
s_{RF}[n]=\sqrt2\operatorname{Re}\{s_{bb}[n]e^{j2\pi f_ct_n}\}.
$$

载波相位以整帧统一时间轴为准，不在每个 CP 和同步片段上随意重置。

复包络网格不强制共轭对称。实通带波形通过取实部自然出现负频率镜像；不得把 400 个独立 QPSK 符号减半。

必须实现无信道时的实波形频点恢复测试：取有效符号、正频率索引 `2048+q`，单位酉 FFT 后乘 $\sqrt2$ 应恢复网格符号。此测试只说明无信道/相应整栅格情况下的归一化正确。

### 5.3 主评测的接收接口

正式主评测采用**理想复数 I/Q 波形接口**：仿真器直接保留解析通带信号，理想下变频后得到复包络，再去 CP、8192 点 FFT、提取 512 网格。

这是有意选择的第一版边界，不是假装已经实现真实实数 ADC、模拟滤波、Hilbert 变换和镜像抑制的全部细节。

时变信道下，直接对一个有限长实通带窗口提取正频率，可能包含镜像泄漏；实际有限长 I/Q 滤波也可能引入额外延迟和有色噪声。**不能在这种前端下仍未经验证地套用本文的复线性白噪声 $y=Hx+w$。**

因此：实通带波形生成和 LFM 演示必须有；检测主曲线必须标注 `front_end=ideal_complex_iq`。真实实数接收前端是后续扩展，不作为 v1 隐含功能。

### 5.4 噪声注入位置

主路径在理想复数 I/Q 的接收时间样本上注入独立复高斯白噪声。单位酉 FFT 后所选频点仍具有同一复噪声方差。

实通带导出若额外加噪，须单独存储 `sigma2_real`。不得直接把复噪声方差复制成任意实通带模型的噪声参数。对“实白噪声 + 正频率 FFT 乘 $\sqrt2$”的特定前端，所选频点复方差是实样本方差的两倍；只有验证该前端后才使用这项转换。

---

## 6. 物理参数化信道：既产生 ICI，又能核验

### 6.1 必须支持的信道族

| 名称 | 内容 | 用途 |
|---|---|---|
| `identity_awgn` | $H=I$、AWGN | 星座、SNR、FFT 和误码校验 |
| `static_multipath` | 多径时延和复增益，时缩为 0 | CP 有效时应退化为对角频域信道 |
| `affine_doppler_mild` | 多径 + 小路径相关时缩 | 较弱 ICI |
| `affine_doppler_moderate` | 多径 + 中等路径相关时缩 | 默认训练和评测主场景 |
| `affine_doppler_strong` | 多径 + 较大路径相关时缩 | 较强 ICI |

可增加 `common_cfo` 作为窄带调试信道，但不能把它当作完整路径相关宽带时缩模型。

水声 Doppler 可涉及时间尺度变化；仅给所有子载波乘同一个频移，不能完整表达频率相关的 Doppler。相关原始研究讨论了时间缩放和重采样补偿。[R6]

### 6.2 默认参数分布：全部属于本次设计假设

每帧生成独立物理路径参数；一帧内路径增益、初始时延和时缩参数固定，路径相位和有效时延随绝对时间演化。

| 参数 | 默认分布 / 取值 |
|---|---|
| 路径数 $L$ | 整数均匀分布于 3–6 |
| 最早路径初始时延 | 2 ms |
| 其他路径初始时延 | 在 (2,14] ms 连续采样并排序 |
| 指数功率时延常数 | 3 ms |
| 路径原始增益 | $\tilde a_\ell\sim\mathcal{CN}(0,p_\ell)$ |
| 功率包络 | $p_\ell\propto\exp[-(\tau_{\ell,0}-2\mathrm{ms})/(3\mathrm{ms})]$ |
| 帧级归一化 | $a_\ell=\tilde a_\ell/\sqrt{\sum_j|\tilde a_j|^2}$ |
| mild 时缩范围 | 每条路径 $\epsilon_\ell\sim U[-2\times10^{-5},2\times10^{-5}]$ |
| moderate 时缩范围 | 每条路径 $\epsilon_\ell\sim U[-10^{-4},10^{-4}]$ |
| strong 时缩范围 | 每条路径 $\epsilon_\ell\sim U[-2\times10^{-4},2\times10^{-4}]$ |
| 外分布压力测试 | $|\epsilon_\ell|\le5\times10^{-4}$，另行标注 |

在 24 kHz 中心处，上述三个默认范围分别对应最大约 0.48、2.4、4.8 Hz 的路径频移；频移随实际 $f_q$ 变化。

时延是用于相对基准的仿真时延，不据此声称真实海深、传播距离或环境已知。这里用 2 ms 的正最小延迟，是为了在所选 FFT 窗口和时缩范围下留出单块有效模型的时间裕量，不是一个通用水声常数。

路径增益归一化用于消除整体发射/传播尺度的无关漂移；**不对每一列 $H$ 分别归一化，也不把每个生成的 $H$ 强行变成单位 Frobenius 范数**。多径相消与深衰落必须保留。

### 6.3 整帧连续时间算子

令 $s_a(t)$ 表示第 5 节定义的整帧解析通带信号，含 LFM、CP、OFDM 和零填充。忽略接收录音的额外整数偏移时，定义：

$$
r_a(t)=\sum_{\ell=1}^{L}a_\ell\,
 s_a\big((1+\epsilon_\ell)t-\tau_{\ell,0}\big)+\eta_a(t).
$$

含义：

- $a_\ell$：第 $\ell$ 条路径的复幅度，吸收固定传播相位。
- $\tau_{\ell,0}$：帧时间原点处的路径时延，单位秒。
- $\epsilon_\ell$：无量纲时缩参数；正值在本约定中提升接收频率。
- $(1+\epsilon_\ell)t-\tau_{\ell,0}$：接收时刻对应的发送端时间。
- 不额外加入时缩的能量补偿系数；幅度约定由 $a_\ell$ 完整定义。

该模型是有限路径、帧内仿射时缩、固定路径增益的可控仿真模型，不包含完整海洋传播、非平稳噪声、路径出生消失或换能器非线性。

### 6.4 单块有效性必须先检查，不能只看最大初始时延

设第 $m$ 个 OFDM 有效区起点为 $T_m$，接收 FFT 取样局部时间为 $t_n=n/F_s$。对应的发送符号局部时间为：

$$
\xi_{\ell,m}(n)=(1+\epsilon_\ell)t_n-\tau_{\ell,m},
\qquad \tau_{\ell,m}=\tau_{\ell,0}-\epsilon_\ell T_m.
$$

单块 CP 模型要求对每个路径、每个数据 OFDM 块和全部 FFT 样本：

$$
-T_{cp}\le\xi_{\ell,m}(n)<T_u.
$$

由于 $1+\epsilon_\ell>0$，检查 $n=0$ 和 $n=N_w-1$ 两个端点即可。同步估计引起 FFT 起点变化时，必须重新检查。

不满足则拒绝该配置/帧并给出原因；主比较不允许把跨块干扰偷偷归入 AWGN。配置生成器可重采不合法路径参数，但必须记录重采次数，不能按检测结果筛选帧。

### 6.5 完整频域信道闭式

在上述有效区间内，CP 保证当前符号可以用相同的 Fourier 级数表示。定义：

$$
\nu_{\ell,q}=\epsilon_\ell(f_c+q\Delta f),
\qquad \delta_{\ell,q}=\frac{\nu_{\ell,q}}{\Delta f}.
$$

定义单位归一化有限和：

$$
D_{N_w}(z)=\frac1{N_w}\sum_{n=0}^{N_w-1}
 e^{j2\pi zn/N_w}
=e^{j\pi (N_w-1)z/N_w}\frac{\operatorname{sinc}(z)}{\operatorname{sinc}(z/N_w)},
$$

其中 $\operatorname{sinc}(u)=\sin(\pi u)/(\pi u)$，$\operatorname{sinc}(0)=1$。

对接收网格 $k$ 和发送网格 $q$，均采用第 3 节有符号索引：

$$
\boxed{
H^{grid}_{kq}(m)=\sum_{\ell=1}^{L}a_\ell
 e^{-j2\pi(f_c+q\Delta f)\tau_{\ell,m}}
 D_{N_w}\big(q-k+\delta_{\ell,q}\big).
}
$$

这来自逐路径时缩后的波形与接收 FFT 基函数的内积；每个路径的频移显式依赖发送频率 $q$。

**实现要求：**

- 正式默认 $D$ 的长度是 **8192**，不是 512 或 400。
- 使用广播和按路径累加；不得对 $k,q$ 写 Python 双循环。
- 允许沿路径、频率块或时间块分块，限制峰值内存。
- 当前参数范围内 $z/N_w$ 不接近非零整数，`torch.sinc` 比值可稳定计算；配置扩展若触及奇点，使用周期约化及解析极限，不能任意加 epsilon。
- 生成器先以 complex128 构造和验证，再显式转换存储/模型 dtype。
- 不施加人工固定 ICI 带宽，不随机删除远离对角线的耦合。

### 6.6 独立波形参考：不要用自己的 $H$ 生成自己的“验证信号”

`waveform_reference` 在每条路径对应的真实发送时间 $\xi$ 上，独立计算 OFDM Fourier 级数：

$$
s_{bb,m}(\xi)=\frac1{\sqrt{N_w}}\sum_q X_q^{(m)}e^{j2\pi q\Delta f\xi},
$$

再应用时缩对应的载波相位、路径复幅度、整帧分段选择，得到接收时间样本；随后执行实际 FFT。

LFM 用解析 chirp 在映射后的时间点求值，静默区返回零。整帧 `piecewise waveform evaluator` 必须区分当前块、CP、其他块、LFM 和静默，不可对整段帧盲目循环取模。

直接指数计算只用于参考验证和少量演示，可以按时间块累加，不要求对所有训练帧都执行。以后用插值加速时，必须单独报告插值误差和前端影响。

至少校验：

$$
\frac{\|Y_{waveform}-H^{grid}X_{grid}\|_2}
{\max(\|Y_{waveform}\|_2,\varepsilon)}<10^{-9}
$$

用于 complex128 的非退化、CP 有效测试。不能只在零 Doppler 时校验。

### 6.7 零 Doppler 极限

$\epsilon_\ell=0$ 时，整数频率网格使 $D_{N_w}(q-k)=\mathbf1\{q=k\}$，于是：

$$
H^{grid}_{kq}=\mathbf1\{k=q\}\sum_\ell a_\ell e^{-j2\pi f_q\tau_{\ell,0}}.
$$

静态多径、CP 有效的信道应恢复对角模型。该测试是仿真器的核心验收，不能仅查看热力图判断。

---

## 7. LFM 接收同步：实现，但不要混淆评测层次

### 7.1 必须实现的基本同步器

对已知 LFM 模板 $p[n]$ 和接收信号 $r[n]$，使用归一化滑动相关：

$$
C[d]=\frac{\left|\sum_{n=0}^{N_L-1}r[d+n]p^*[n]\right|^2}
{\left(\sum_{n=0}^{N_L-1}|r[d+n]|^2\right)
 \left(\sum_{n=0}^{N_L-1}|p[n]|^2\right)+\varepsilon_{sync}}.
$$

输出峰值位置、峰值分数、是否超过阈值、候选到达位置。模板匹配可以用 FFT 卷积实现；必须通过已知时移的索引测试，明确返回的是模板起点而非卷积峰尾。

默认阈值先设 `0.1` 作为启动值，只能在验证集和无信号窗口上调整。同步噪声窗口必须加入测试，不能只测“已知一定有帧”的情况。

基础实现采用最大归一化峰作为**所选匹配峰**，不将其自动称为最早物理路径到达时间。多径下最强峰与最早路径可能不同。

### 7.2 两种运行模式

| 模式 | 定时来源 | 用途 |
|---|---|---|
| `oracle_timing` | 仿真注入的帧起点和预定 FFT 窗口 | 三检测器主比较 |
| `lfm_detect` | 只从接收同步头相关输出估计位置 | 同步成功率、到达时间误差和小规模前端演示 |

`lfm_detect` 第一版不需要实现未知 Doppler 的联合优化。若在已知信道模式下用已知路径时延把匹配峰转为发送帧原点，必须标记 `oracle_channel_assisted_timing=true`，写明校正方法；不能把这种演示称为无先验的全盲同步。

使用估计窗口继续检测时，应按实际窗口重新构造有效 $H$，复查 CP 条件，并把同步失败作为帧级失败处理。**不允许错位 $y$ 配理想窗口的 $H$。** 该模式结果与主曲线分文件保存。

没有可辨识的首径时，不为测试强行把最强峰修正成真实首径。基础同步器的局限应作为结果报告。

### 7.3 同步指标

记录：无信号虚警率、帧漏检率、所选峰的时间误差分布、可用 FFT 窗口比例。需要定义时间真值时，分别保留注入帧起点、最早路径到达位置和所选路径/模板峰参考，不能用三个不同概念计算同一个 RMSE。

三个检测算法共享同一个同步器；不能把同步器的结果分别包装成三个检测器各自获得的同步增益。

---

## 8. 将带导频 OFDM 严格转换为方阵 QPSK 检测

### 8.1 不能直接把 512 网格交给原 PG-VAMP

完整网格满足：

$$
Y_g=H_gX_g+W_g.
$$

令 $\mathcal D$ 为 400 个数据位置，$\mathcal P$ 为 64 个导频位置，其他位置为零。用选择矩阵表示：

$$
X_g=S_{\mathcal D}x+S_{\mathcal P}p.
$$

原 PG-VAMP 只接收全部未知量为 QPSK 的方阵系统。不允许把已知导频和空载波交给同一个 QPSK denoiser。

### 8.2 显式接收行选择与已知导频消除

本版对三个检测器统一采用接收数据频点 $\mathcal D$：

$$
\boxed{
H=S_{\mathcal D}^{H}H_gS_{\mathcal D}\in\mathbb C^{400\times400},
}
$$

$$
\boxed{
y=S_{\mathcal D}^{H}Y_g-
S_{\mathcal D}^{H}H_gS_{\mathcal P}p=Hx+w.
}
$$

对应代码：

```python
H_DD = H_grid[..., data_idx, :][..., :, data_idx]
H_DP = H_grid[..., data_idx, :][..., :, pilot_idx]
y = Y_grid[..., data_idx] - (H_DP @ pilots.unsqueeze(-1)).squeeze(-1)
```

`data_idx`、`pilot_idx` 是 512 网格中心排列的索引，不是 8192 点 FFT 索引。

必须测试 $y=H_{DD}x+w_D$，包括强 ICI 和导频泄漏明显的情况。

### 8.3 这个选择的收益与代价

这是明确、可核验的接收处理，不是为了迎合方阵接口随便截取左上角子矩阵。

但它确实丢弃了接收导频/空载波位置上包含的部分数据 ICI 信息。三个算法必须使用同样的 400 行，因此比较公平，但**不能把该接收器称为使用了全部 512 个接收观测的最优方案**。

未来使用全部观测会得到 $512\times400$ 的矩形系统，应单独扩展 PG-VAMP 合同，不能在本版静默添加。

### 8.4 信道信息条件

本版 `csi_mode=perfect`，因此导频消除使用真实 $H_{DP}$；减去的是已知确定性贡献，所选噪声仍为白噪声。

`csi_mode=estimated` 第一版不支持，应明确报错。不能用逐导频单抽头 LS 宣称已估计完整时变 ICI 信道，也不能忽略信道估计误差对导频消除和噪声协方差的影响。

---

## 9. 噪声、能量与横轴定义

统一定义：

$$
w\sim\mathcal{CN}(0,\sigma^2I),\qquad
\mathbb E|w_i|^2=\sigma^2,
\qquad\gamma_w=1/\sigma^2.
$$

实部、虚部方差均为 $\sigma^2/2$。

```python
noise = torch.sqrt(sigma2 / 2) * (
    torch.randn(shape, generator=gen_r, dtype=real_dtype)
    + 1j * torch.randn(shape, generator=gen_i, dtype=real_dtype)
)
```

本版横轴主定义为：

$$
\mathrm{EsN0}_{dB}=10\log_{10}(E_s/\sigma^2),
\qquad E_s=1,
\qquad \sigma^2=10^{-\mathrm{EsN0}_{dB}/10}.
$$

图上标注 **`Es/N0 (dB), unit-energy data QPSK`**，不写含糊的“采样点 SNR”。不根据每帧 $\|Hx\|^2$ 自动调噪，不把每个深衰落都补偿为相同接收 SNR。

另行记录检测系统的平均接收信号噪声比：

$$
\mathrm{SNR}_{rx,dB}=10\log_{10}\frac{\|H\|_F^2}{N_d\sigma^2}.
$$

它是对独立单位能量数据符号取平均的量，不包含已消除的导频信号。

未编码 QPSK 若只按数据符号能量定义比特能量，有：

$$
(E_b/N_0)_{dB}=(E_s/N_0)_{dB}-10\log_{10}2.
$$

包含导频、CP、LFM 的整帧发送能量每信息比特不能只减 3.0103 dB。需要这种口径时用实际发送帧离散能量除以 6400，并明确离散能量/噪声归一化，不将不同口径的曲线混绘。

identity-AWGN 下可用未编码 QPSK 的理论 BER 做噪声定义校验：

$$
P_b=Q\!\left(\sqrt{E_s/\sigma^2}\right)
=\frac12\operatorname{erfc}\!\left(\sqrt{\frac{E_s}{2\sigma^2}}\right).
$$

这是独立 I/Q 符号判决推导的校验项，不是 PG-VAMP 在多径 Doppler 下的理论曲线。

---

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

---

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

## 19. 推荐工程结构

```text
pg-vamp-ofdm/
  pyproject.toml
  README.md
  IMPLEMENTATION_STATUS.md
  VALIDATION.md
  .gitignore
  docs/
    mathematical_spec.md                # 保留原数学合同及来源
    CODEX_ENGINEERING_SPEC.md           # 本任务书
    signal_and_channel_model.md
    evaluation_protocol.md
  configs/
    base.yaml
    smoke_math.yaml
    smoke_system.yaml
    cpu_dev.yaml
    main.yaml
    cuda_example.yaml
  src/pgvamp_ofdm/
    __init__.py
    __main__.py
    cli.py
    config.py
    modulation/
      qpsk.py
      allocation.py
    waveform/
      ofdm.py
      frame.py
      lfm.py
      continuous.py                    # 分段连续时间参考波形
    channel/
      parameters.py
      affine.py
      effective_matrix.py
      noise.py
      validity.py
    receiver/
      synchronization.py
      fft_receiver.py
      preprocessing.py                 # 导频消除与明确索引选择
    data/
      records.py
      manifest.py
      generate.py
      dataset.py
      materialize.py
    algorithms/
      common.py
      mmse.py
      vamp.py
      pg_vamp/
        topology.py
        majorizer.py
        linear.py
        denoiser.py
        messages.py
        model.py
    training/
      loss.py
      trainer.py
      checkpoint.py
    evaluation/
      runner.py
      metrics.py
      bootstrap.py
      timing.py
      report.py
    utils/
      device.py
      random.py
      logging.py
      validation.py
    reference/
      dense_pg_vamp.py                 # 本工程自带的小规模正确性 oracle
      dense_vamp_cholesky.py
  tests/
    test_allocation.py
    test_qpsk.py
    test_waveform.py
    test_lfm.py
    test_channel.py
    test_waveform_matrix_equivalence.py
    test_cp_validity.py
    test_pilot_elimination.py
    test_noise.py
    test_data.py
    test_mmse.py
    test_vamp.py
    test_pg_vamp_math.py
    test_gradients.py
    test_checkpoint.py
    test_metrics.py
    test_fair_evaluation.py
    test_devices.py
    test_cli.py
  scripts/
    run_smoke.sh
    run_smoke.ps1
  data/                                # gitignore
  runs/                                # gitignore
  results/                             # gitignore
```

正式算法实现与 `reference/dense_pg_vamp.py` 分离。reference 以可读、直接、优先正确性为目标；正式实现以批量化和工程接口为目标。测试必须比较两者，但不能边修改 reference 边声称“与参考等价”。

每个公开函数包含类型标注、shape 说明、dtype/device 要求和输入校验。允许层循环、路径循环和参考分块循环；不对常规 batch 写逐样本 Python 循环替代可批量线性代数。

### 19.1 依赖

核心依赖：PyTorch、NumPy、PyYAML、Matplotlib；测试使用 pytest。CPU 进程内存统计可将 psutil 设为可选依赖。SciPy 不是核心仿真器的必需依赖，LFM 与主信号运算用 PyTorch 完成。

优先使用工作区已安装且支持所需 API 的 PyTorch，不为追随网页文档版本无故升级。建议 Python 3.11/3.12 环境，实际兼容范围以测试记录为准。

`pyproject.toml` 配置 `src` 布局、命令入口 `pgvamp-ofdm`，同时支持 `python -m pgvamp_ofdm`。GPU 版本的 PyTorch 安装方式随环境匹配，项目默认安装流程不强制下载 CUDA。

---

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

---

## 22. 分阶段工作包与验收门槛

### WP0：核对工作区与规格

以本任务书建立项目配置、设备选择、目录、reference 正确性实现与最小 CLI。记录本任务书文件哈希和代码版本；不升级无关依赖，不删除工作区无关文件。

**验收：** `inspect-config` 输出正确派生参数；`reference/dense_pg_vamp.py` 与 `reference/dense_vamp_cholesky.py` 可被最小单元测试调用；不存在任何对本文件之外工程规格或外部 oracle 的运行时依赖。

### WP1：调制、帧与 LFM

实现固定 QPSK 映射、400/64/47/1 分配、8192 点 IFFT、CP、整帧、实通带导出、LFM 和基本匹配相关。

**验收：** 星座双向映射、Parseval、CP 复制、频点索引、无信道实通带恢复、无噪声时移定位、帧长度全部正确；输出 PSD 和相关曲线。

### WP2：物理信道与有效模型

实现整帧仿射路径模型、CP 支撑验证、完整 $512\times512$ $H_g$ 闭式、独立波形参考、单位酉接收 FFT、导频消除和400维系统。

**验收：** 静态多径退化、非零路径相关时缩的时域/频域一致性、导频消除、噪声白性通过；不能仅用随机矩阵测试代替物理链路验收。

### WP3：数据集与划分

实现紧凑帧记录、manifest、split、随机流、重放、按需矩阵生成、物化格式与输入校验。

**验收：** 相同记录可重建、不同 split 无帧/信道泄漏、不同算法取得完全相同样本、容量预估正确。

### WP4：传统 MMSE 与传统 VAMP

实现 Cholesky MMSE、精确 SVD VAMP、独立 Cholesky VAMP oracle、共同 QPSK 模块和消息保护。

**验收：** 与显式小矩阵线性求解相符；SVD/Cholesky VAMP 层级输出相符；零/秩亏信道正确；无训练参数。

### WP5：PG-VAMP 数学合同

实现单调门限、软门控、能量补足、安全矩阵、中心化完整残差、精确条件散度、方差校准、解析 QPSK 和非线性保护。

**验收：** 第 23 节的矩阵性质、Jacobian、全图极限、前向和参数梯度检查通过；16 个可学习标量严格成立。

### WP6：训练、保存、恢复与推理

完成 CPU 默认训练、显式 CUDA、损失、日志、checkpoint、安全恢复与无标签推理。

**验收：** 数学 smoke 和真实尺寸 smoke 均完成；保存/加载一致；恢复训练继续更新；CPU-only 环境不调用 CUDA。

### WP7：统一评测与报告

实现配对样本评测、计数、bootstrap、计时、稳定性记录、图表和自动 Markdown 报告。

**验收：** 手工构造的预测可得到正确 BER/SER/FER；零错误与硬失败没有被隐藏；三算法使用相同数据哈希；报告只引用真实结果。

### WP8：完整交付

整理 README、配置说明、算法数学说明、测试记录、实现状态和命令；至少实际运行 CPU 的完整尺寸 smoke。

主训练或完整 SNR sweep 因资源限制未执行时，保留可运行命令与 `未执行` 状态，不能用占位曲线宣称已完成指标分析。完成代码功能与完成充分性能验证应分别验收。

---

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

---

## 24. 本任务书编写时已核验与未核验的内容

为核对本次新增设计，已在当前 CPU 环境执行了独立的参数与波形算式检查。它们**不是工程验收，也不是三算法实验**。

| 核验项 | 结果 |
|---|---|
| 网格分配 | 400 数据、64 导频、48 空载波，计数正确 |
| 高频 FFT 与间隔 | 8192 点、96 kHz、11.71875 Hz，一致 |
| 有效/CP 时长 | 85.333333 / 21.333333 ms |
| 默认帧长度 | 91776 样本、0.956 s、6400 数据比特 |
| 非零时缩信道公式 | 3 路径、8 个分散发送频点、512 个接收频点、8192 点 FFT；独立波形与闭式 $H$ 的相对误差约 $1.45\times10^{-13}$ |
| 无信道实通带恢复 | 默认全部非零网格符号的最大绝对误差约 $8.11\times10^{-13}$ |
| 该测试帧的 CP 支撑 | 8 个 OFDM 块均满足 |

上述算式测试采用 NumPy 独立数值计算。尚未在本任务中生成完整 Python 工程、训练 PG-VAMP、实现两个基线或运行主评测。因此不能从本表推出 BER、加速比或可训练性已经得到验证。

---

## 25. 来源与技术依据

### 理论溯源材料（非 Codex 执行依赖）

`PG_VAMP_完整数学推导_标准Markdown公式.md` 可用于人工审计 PG-VAMP 推导来源。**它不是 Codex 启动或完成本工程的前置文件**；实现所需的算法公式、保护规则、梯度规则、训练损失和边界条件已在本任务书中完整给出。


### 外部核对来源

**[R1]** Rangan, Schniter, Fletcher, *Vector Approximate Message Passing*, arXiv:1610.03082。用于传统 VAMP、精确线性模块、SVD 实现思路与理论矩阵假设边界。

```text
https://arxiv.org/abs/1610.03082
```

**[R2]** PyTorch 官方文档，*Complex Numbers*、*torch.cholesky_solve*。用于复数自动微分与复 Hermitian 批量求解接口。网页的版本不是本工程的实际运行版本。

```text
https://docs.pytorch.org/docs/stable/complex_numbers.html
https://docs.pytorch.org/docs/stable/generated/torch.cholesky_solve.html
```

**[R3]** PyTorch 官方文档，*torch.fft.fft*。用于 FFT 归一化约定。

```text
https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html
```

**[R4]** MathWorks 官方文档，*comm.OFDMModulator*。用于区分 FFT 网格、数据/导频/保护载波、CP 与过采样等通信参数概念；本工程不依赖 MATLAB，也不声称本任务的具体分配来自标准。

```text
https://www.mathworks.com/help/comm/ref/comm.ofdmmodulator-system-object.html
```

**[R5]** SciPy 官方文档，*scipy.signal.chirp*。用于核对线性 chirp 的瞬时频率与积分相位定义；本工程的生成器使用 PyTorch 实现。

```text
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
```

**[R6]** Mirhedayati Roudsari and Bousquet, *A Time-Varying Filter for Doppler Compensation Applied to Underwater Acoustic OFDM*, Sensors 19(1):105, DOI 10.3390/s19010105。使用作者摘要中关于宽带时间缩放、导频和重采样补偿的论述；本次没有采用该文的具体实验参数或性能结果。

```text
https://pubmed.ncbi.nlm.nih.gov/30597988/
```

**[R7]** PyTorch 官方文档，*Reproducibility*。用于区分记录环境下的复现与跨版本/跨平台逐位一致性。

```text
https://docs.pytorch.org/docs/stable/notes/randomness.html
```

核对日期为 2026-09-17。具体 OFDM 数值、导频分配、路径参数范围、数据规模和训练配置均为本任务提出的启动设计。

---

## 26. 可直接发送给 Codex 的入口指令

```text
请只依据 PG_VAMP_CP_OFDM_完整工程_Codex执行书_v2.md，
从零构建一个可运行的 Python + PyTorch 完整工程。

这份执行书是本版唯一必须读取的工程实施规格。
不需要读取或合并其他工程执行书。
PG_VAMP_完整数学推导_标准Markdown公式.md 仅是可选的人工理论审计材料，
本执行书第 14–15 节已经给出了实现所需的完整算法与训练合同。

本次不是只实现 PG-VAMP，而是完成：
CP-OFDM/QPSK 仿真数据生成、LFM 同步模块、物理参数化时变信道、
接收 FFT 与已知导频消除、PG-VAMP-VC、传统线性 MMSE、传统 VAMP、
CPU/GPU 训练与推理、统一指标评测、图表和 Markdown 结果报告。

默认训练和推理设备必须为 CPU，GPU 只能显式指定。
默认通信配置：96 kHz、21–27 kHz、512 子载波网格、8192 点波形 FFT、
400 数据 + 64 导频 + 47 边缘空载波 + 1 中心空载波、QPSK、CP 2048、
40 ms LFM、每帧 8 个 OFDM 数据块、无信道纠错编码。

先按本执行书建立 reference/dense_pg_vamp.py 和
reference/dense_vamp_cholesky.py，作为本工程自己的独立正确性参考；
只能报告本次实际运行得到的测试数量与结果，不能继承历史项目的测试结论。

先完成 waveform_reference 与 effective_fast 的一致性，
不得拿随机稠密矩阵冒充水声 OFDM 数据。
使用路径相关仿射时缩与 8192 点 Dirichlet 闭式构造完整 ICI 信道。
严格检查 CP 支撑条件，超出单块模型必须报错。
主评测采用理想 I/Q、理想同步和完整真实信道，LFM 同步性能另报。

检测前显式用 H_DP 消除已知导频，只取 data_q 对应的接收行，
得到三个算法共同使用的 400×400 QPSK 系统。
不得把 512 网格的导频/零载波直接送入 QPSK denoiser，
也不得使用不明确的左上角矩阵裁剪。

PG-VAMP 必须保持 2T 个可学习标量、单调 soft gate、软能量补足、
解析安全项、中心化完整残差、实际条件散度和平方和方差校准。
不能把它改成标准 VAMP、普通展开梯度下降、GNN 或有限步 CG。
传统 VAMP 使用精确 LMMSE，提供 SVD 正式实现与 Cholesky 等价参考。
传统 MMSE 使用完整有效 H，不退化成弱化的单抽头基线。

按 WP0–WP8 顺序实现和测试；先通过数学与物理 smoke，
再做训练、统一配对评测与结果报告。三算法使用完全相同的测试输入。
数据默认保存物理参数和种子，避免无提示存储大量稠密矩阵。

请实际执行可运行的 CPU 测试与完整尺寸 smoke，记录真实命令与结果。
如果主训练或完整 SNR sweep 没有运行，明确写未执行，
不要虚构 BER、训练结果、测试数量、加速比或“PG-VAMP 必然胜出”。

最终返回工程文件清单、可复制运行命令、实际测试结果、
已有指标文件/报告路径，以及具体未完成事项。
```
