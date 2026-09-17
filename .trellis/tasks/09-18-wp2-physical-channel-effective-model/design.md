# WP2 technical design — approved and implemented

## Boundaries and flow
显式 config/runtime/随机源 → 帧级路径 + WP1 FrameWaveforms → CP 检查。
独立路径一：连续分段求值 → 仿射传播 → 理想下变频/时间噪声 → 实际 ortho FFT → Y_grid。
路径二：§6.5 闭式 → H_grid。使用相同窗口和 allocation 选择 H_DD/H_DP、消除导频。
物理一致性通过之后，后续 WP3 才可使用 effective_fast。

## Proposed modules and interfaces
| 模块 | 责任与合同 |
| --- | --- |
| channel/parameters.py | 单帧 PathParameters：gain complex128[L]、delay_s/epsilon float64[L]、scenario。显式路径或显式 generator 采样；逐帧接口避免可变 3–6 路径的隐式 padding。 |
| channel/validity.py | 路径、FrameLayout、实际 FFT 窗口 → 支撑结果/可读异常，定位块/路径/端点；校验有限值、shape/device、1+epsilon>0。默认非法帧直接拒绝；未来若重采需返回次数。 |
| waveform/continuous.py | WP1 grid/layout/LFM 定义 + 任意 float64 时间 → 整帧解析通带值；按时间分块直接指数求和，不依赖 H/Dirichlet。 |
| channel/affine.py | 在 (1+epsilon)t-tau0 求值，按复增益累加，接收绝对时间下变频；无额外时缩能量系数。 |
| channel/effective_matrix.py | 相同路径/块时间/窗口 → complex128 H_grid[512,512]；广播 k/q、按路径累加，可频率分块，禁止 k/q Python 双循环。 |
| channel/noise.py | EsN0→sigma2，显式独立实/虚随机流→CN(0,sigma2)，可返回 SNR_rx，不能改变信号幅度。 |
| receiver/fft_receiver.py | 明确窗口 I/Q [...,8192] → ortho FFT → [...,512]，使用 allocation 的 baseband FFT 索引。 |
| receiver/preprocessing.py | H_grid、Y_grid、pilots、allocation、sigma2 → H_DD[...,400,400]、y[...,400] 和方差/窗口元数据；批次轴一致，不隐式广播错配输入。 |

## Time, precision and independence
- 保留 WP1 半开 FrameLayout 和全局载波相位。额外录音整数偏移与物理时间分开，
  传播求值前移除该偏移，不能重置路径演化。
- 先按真实发送时间选片段；OFDM/CP 使用所属块 useful 起点的局部时间直接
  Fourier 级数，静默及帧外返回零，不对整帧取模。
- LFM 使用 WP1 局部 chirp、离散归一化常数及对称 Tukey；连续窗按
  u=t_local*Fs/(NL-1) 延拓，在窗定义外归零，整数点与 WP1 一致。
  片段末端至下一片段之间不得发生余弦窗越界反弹。
- 默认窗口 R=T_m。偏移窗口 R=T_m+Delta 时，发送局部时间为
  xi=(1+epsilon)(Delta+n/Fs)-tau_m，CP 检查用该 xi。绝对时间下变频下，
  §6.5 每路径每列另乘 exp(j*2*pi*(q*df+epsilon*(fc+q*df))*Delta)。
  此扩展必须经独立波形验证，不允许错位 y 配理想窗口 H。
- D_N 长度 8192，默认范围使用 sinc 比值；扩展至奇点需周期约化/解析极限，
  或明确拒绝支持范围外输入，不加任意 epsilon。
- float64/complex128 构造及验证后才显式转换；CPU 默认，拒绝 algebra fixture，
  不隐式探测 CUDA 或迁移 dtype/device。
- 参考只共享 config/layout/allocation 元数据，不共享 H/闭式核；期望值独立构造。

## Compatibility, risks and rollback
保持 WP1 输入形状与帧长度，不重写 WP0 算法 reference。不新增数据生成/检测器 CLI；
可新增专用 run_wp2_audit.py 保存少量完整物理帧证据，不称作 smoke_system。
直接指数求和昂贵，按路径/时间分块，不能靠缩小 FFT 或放宽 1e-9 规避验收。
统计测试预定固定 seed、样本数与容差，complex64 单独记录。
回滚以新增模块和接入点为单位；失败优先查时间/相位/索引，不改执行书，
不通过 H 合成参考信号。WP3 的持久化和后端整合仍由后续任务负责。
