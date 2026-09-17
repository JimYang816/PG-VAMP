# WP2 — Physical Channel and Effective Model

## Goal and authority
在已验收 WP1 发送链路之上建立可独立核验的仿射物理信道、接收 FFT 和
400 维检测输入，为 WP3 提供可靠物理基础。
唯一实施规格是 [CODEX_ENGINEERING_SPEC.md](../../../../../docs/CODEX_ENGINEERING_SPEC.md)
§§5–9、19–23；本规划不修改其公式、参数及完整验收门槛。

## Authorization and dependency
用户于 2026-09-18 明确授权创建 WP2 child task，**仅 planning，不开始 implementation**。
父任务为 `09-17-pgvamp-complete-engineering`；前置 WP1 已完成归档，工作提交
`414de3f`，历史全量 166 passed / 2 skipped，证据见归档 WP1 research。
规划轮未重跑上述测试，不将它们视作 WP2 验收。
用户随后明确回复“规划批准，可以进入 implementation。”，已授权最新规划实施；
任务现为 in_progress，后续以实际 WP2 检查证据验收。

## Requirements
| ID | 范围与来源 |
| --- | --- |
| R1 | 五类信道 identity_awgn、static_multipath、affine_doppler_mild/moderate/strong；按 §6.2 采样帧级路径，仅归一化路径总能量，不归一化 H 的列或 Frobenius 范数。 |
| R2 | 整帧逐路径仿射时间映射，逐路径/逐块检查 CP 支撑，覆盖最后块和实际窗口偏移；越界明确拒绝，不能当作 AWGN。§§6.3–6.4、7.2。 |
| R3 | 完整 512×512 H_grid，D 长度 8192，保留全部 ICI、频率相关 Doppler 和绝对块时间；先以 complex128 构造核验。§6.5。 |
| R4 | 独立分段连续波形加真实接收 FFT，覆盖 LFM、CP、各块、静默及帧外零，不使用 H 或 Dirichlet 核生成参考信号。§6.6。 |
| R5 | ideal_complex_iq / perfect CSI / oracle_timing；单位酉 FFT、显式网格索引、真实 H_DP 导频消除，输出 H_DD[400,400]、y[400] 和已知 sigma2。§§5、8。 |
| R6 | 时间 I/Q 复 AWGN，Es=1、sigma2=10^(-EsN0/10)，不按每帧接收功率调噪；随机源显式可复现。§9。 |
| R7 | 保存实际物理审计参数、种子、窗口、dtype/device、误差、噪声统计、命令及源指纹；区分 complex128 正确性和 complex64 实验。§§21–23。 |

## Acceptance criteria
| ID | 可观测结果 | 映射 |
| --- | --- | --- |
| A1 | 五类路径合法、分布/排序/归一化符合合同；identity 为 I，静态多径非对角能量为浮点误差级。 | R1/R3；§§6.1–6.2、6.7、23.1 |
| A2 | 正负时缩、最后块、CP 下界包含/上界排除、实际窗口偏移及非法参数有测试，拒绝原因定位到块/路径/端点。 | R2；§§6.4、7.2、23.1 |
| A3 | 默认 8192/512 尺寸、至少两条不同非零 epsilon、CP 有效且非退化的 complex128 用例中，独立波形 FFT 与 H_grid X_grid 相对误差 <1e-9；频移随 q 变化。 | R3/R4；§§6.5–6.6、23.1 |
| A4 | 整数采样点重建 WP1 波形，独立非整数时间/边界测试覆盖 chirp、CP、邻块、静默，整帧不被周期化。 | R4；§§5–6 |
| A5 | 强 ICI、显著导频泄漏下，无噪/含噪均满足 y=H_DD x+w_D；索引/形状正确，estimated CSI 明确拒绝。 | R5；§8、23.1 |
| A6 | 固定种子和样本数解释的容差下，实虚方差、FFT 后复方差、跨频点协方差正确；identity-AWGN 硬 QPSK Monte Carlo BER 与理论在计数容差内一致。 | R6；§9、23.1 |
| A7 | WP0/WP1 回归、静态检查、实际 CPU 完整尺寸物理审计通过；CUDA 无硬件明确 skipped，未运行实验如实标注。 | R7；§§21–23 |

## Out of scope and deferred work
WP3 数据集、manifest、split、缓存/后端调度；WP4–WP8 检测器、训练、checkpoint、
评测 sweep、最终报告和完整系统 smoke。WP2 提供后续使用的物理接口。
不做真实 ADC/Hilbert/模拟滤波、estimated CSI、矩形检测或人工 ICI 截断。
不扩展盲 Doppler 联合同步或完整 lfm_detect 评测；保留实际窗口 CP/H 合同，
不将 WP1 匹配峰等同于最早物理路径。§23.1 数据重放/split 留给 WP3，不能删除。

## Reviewed planning
父任务及执行书已定义范围，无阻塞产品决策。design.md 与 implement.md 保存技术
设计和执行门槛；本次规划不构成物理链路验收。
