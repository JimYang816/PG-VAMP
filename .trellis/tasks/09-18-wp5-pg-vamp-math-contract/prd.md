# WP5 PG-VAMP Mathematical Contract

## Goal and authorization
交付符合执行书的正式 PG-VAMP-VC 检测器及独立数学验收，为 WP6 提供可微模型。
用户于 2026-09-18 先授权创建 WP5 child task，仅 planning；随后明确批准最新规划，
授权进入 implementation。按该批准执行 WP5，不扩展到 WP6。

## Authority and prerequisites
唯一实施规格：`docs/CODEX_ENGINEERING_SPEC.md` §§11、14、17.4、20、22/WP5、23。
本轮核对 SHA-256：`A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B`。
父任务为 `09-17-pgvamp-complete-engineering`。WP4 已完成归档及独立验收；其
`research/check-report.md` 记录 313 passed / 5 CUDA skipped，属于历史证据，本轮未重跑。
WP0 已有独立 DensePGVAMP；WP4 已有 DetectionResult、生产 QPSK/messages 和精确 SVD VAMP；
pg_vamp 配置已有全部默认常数。详细定位见 `research/planning-evidence.md`。

## Requirements
- R1：§11 无标签统一 API；默认 CPU/complex128，显式 complex64/CUDA；不隐式转换、
  不修改 H/噪声/物理数据。返回最终后验均值、概率和统一硬判决。
- R2：只学习 raw_gaps[T]/raw_mu[T] 共 2T 实标量，默认 T=8 共 16；遵守 §14.1–14.2
  初始化、门限范围/间隔、soft gate、零边/零列和非对称合同。
- R3：§14.3–14.5 能量补足、安全矩阵、中心化完整残差、实际条件散度与平方和方差；
  同层共用实际矩阵/分解，精确 trace，不替换为传统 VAMP 精度公式。
- R4：§14.6–14.8 解析后验、真实 alpha1、保留旧消息的拒绝、保持候选均值的精度截断、
  显式 underflow/无信息保护和完整整网梯度；非有限算子/分解失败明确报错。
- R5：逐层诊断覆盖消息计数、rho/mu、两种分母的有效边比例、安全项相对大小及 c；
  常规推理不保存全部大矩阵；不从软边数宣称稀疏加速或 BER 收益。
- R6：独立 oracle 逐层前向和参数梯度对照，以及独立矩阵性质/Jacobian/极限验证；
  生产与 reference 不共享数值实现。

## Acceptance criteria
| ID | 可观察结果 | 来源 / requirement |
| --- | --- | --- |
| A1 | 不同合法 T 下参数恰为 2T，默认 16；两基线仍为 0；初始化、门限严格单调/间隔/范围、mask 嵌套、零边/零列及非对称通过 | §§14.1–14.2、23.2 / R2 |
| A2 | diag(G)=diag(H^H H)，Gbar-H^H H PSD，图细化 Loewner 单调；O(N²) ell 与独立小矩阵定义一致 | §§14.3、23.2 / R3 |
| A3 | 中心化残差与 apply_B 等价，B Hermitian 且 0≺B≼A^-1，固定单层输入二次目标不增；每层一次分解，无显式逆 | §§14.4、23.2 / R3 |
| A4 | 固定 H/gamma2/M/mu，对 r2 的实展开 Jacobian trace/(2N) 等于 alpha2；平方和方差等于完整协方差迹且非负 | §§14.5、23.2 / R3 |
| A5 | N=8/16/32 非退化 ICI 下与独立 DensePGVAMP 的逐层均值/精度/散度/概率及两组参数梯度一致；gradcheck 和完整反向通过 | §§0.3、14.8、23.2 / R6 |
| A6 | 强制全图逐层等于精确 VAMP 且不依赖 mu；identity/对角硬判决、零/秩亏/混合 batch、弱信道和联合尺度不变性通过 | §23.2 / R2–R4 |
| A7 | 四点枚举后验、负/非法精度拒绝、旧消息保留、固定均值截断、underflow 和危险除法前保护具有有限 backward；无信息零输出/均匀概率/零梯度，硬失败不冒充无信息 | §§14.5–14.8、23.2 / R4 |
| A8 | shape/dtype/device 校验、CPU 默认、单精度有限输出/梯度、可用时 CUDA 对照；禁止 API/detach/hard mask/隐式 jitter 检查通过 | §§11、14.8、23.3 / R1、R4 |
| A9 | WP3 真实非零时缩 400 维样本上三算法共用输入 ID/hash，前向有限；修改或移除标签不影响全部预测；诊断符合 R5 | §§11、17.4、23.3 / R1、R5 |
| A10 | 独立 check、相关及全量回归、Ruff/format/mypy 通过；保存真实命令/环境/容差/失败/skip，文档区分数学验收和系统实验 | §§22–23 / R1–R6 |

complex128 前向 atol=1e-9/rtol=1e-8，PSD 下界 -1e-9，gradcheck
eps=1e-6/atol=2e-5/rtol=2e-4；单精度容差另行说明，调整须有条件数依据。
全图/对角允许自然零梯度；有效非零学习梯度只在非退化 ICI fixture 上要求。

## Out of scope
WP6 训练循环、优化器、checkpoint/恢复/infer CLI、两次参数更新及完整 smoke_system；
WP7 评测/统计/计时/报告；WP8 完整交付。A9 物理前向不替代 WP6 系统 smoke。
不加入 hard mask 主评测、稀疏求解、CG、随机 trace、AMP/float16、矩形系统、GNN/MLP 或额外参数。
最初 planning 轮未修改产品、测试、配置或 spec，未执行数值实验、提交或归档。
