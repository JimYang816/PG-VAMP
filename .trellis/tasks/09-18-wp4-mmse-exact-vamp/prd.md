# WP4 — Full-H MMSE and Exact VAMP Baselines

## Goal and authority
交付可独立核验、没有训练参数的完整 H 线性 MMSE 与精确 VAMP，为 WP5–WP7 提供公平基线。
唯一实施规格：`docs/CODEX_ENGINEERING_SPEC.md` §§11–13、14.6、20、22/WP4、23；本任务不得改写数学合同。
源 SHA-256：A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B。

## Authorization and dependency
用户于 2026-09-18 授权创建 WP4 child task，**只做 planning，不开始 implementation**。
父任务：`09-17-pgvamp-complete-engineering`。该 planning-only 轮次已结束；用户随后明确批准最新规划并授权 implementation。
前置 WP3 已完成归档；历史验收 248 passed / 3 CUDA skipped，本轮未重跑。
当前授权覆盖 WP4 implementation 和质量检查；工作提交按后续完成流程审核。

## Confirmed facts at planning baseline
- WP0 的 `reference/dense_vamp_cholesky.py` 已有独立逐层状态及保护计数，尚非正式检测器。
- `modulation/qpsk.py` 已定义唯一映射、最近邻判决和最低 class index tie-break。
- `utils/validation.py` 已验证 batched 方阵、有限值、正 sigma2、配对 dtype 和同设备。
- WP3 `detection_inputs` 只返回 H/y/sigma2；单样本需显式添加 batch 维。
- 正式 algorithms 目录和 `vamp_reference_32` 配置尚不存在。

## Requirements
| ID | Required outcome | Source |
| --- | --- | --- |
| R1 | DetectionResult / Detector.detect 统一接口；只输入 H[B,N,N]、y[B,N]、sigma2[B]，输出规定 shape/dtype 的软符号、classes、bits、概率及 diagnostics | §11 |
| R2 | 完整 H 的 Cholesky MMSE，原始线性软输出、最近邻硬判决、probabilities=None；名称 MMSE (linear)，不添加后验 denoiser | §12 |
| R3 | 精确 SVD VAMP，每次 detect 对 H 分解一次；默认 8 层、r2=0/gamma2=1；alpha2 和 c 分别求和；最终输出后验均值/概率；提供独立命名 32 层配置 | §13 |
| R4 | 生产共享解析 QPSK、真实 alpha1、显式消息保护；拒绝非法候选并保持旧消息，精度截断不改变候选均值；零/数值无信息返回零软符号和均匀概率，记录保护计数 | §§13.4、14.6 |
| R5 | 两基线训练参数为 0，不读 PG checkpoint，无阻尼/自适应停止；不修改 H、不加隐式 jitter、不用 inverse/CG/随机 trace | §§12–14、23 |
| R6 | CPU 默认，complex128 正确性基准，complex64 显式验证；输入非法、分解失败或算子非有限明确报错，不静默降精度/跨设备/丢弃样本 | §§10.5、15.2、17.6、23.3 |
| R7 | 两真实基线消费相同 WP3 样本且不修改输入；修改或移除标签不影响检测结果，外部记录 sample ID/hash | §§11、23.3 |

## Acceptance criteria
| ID | Observable evidence | Requirements |
| --- | --- | --- |
| A1 | MMSE 对独立小矩阵 solve、正规方程残差、对角闭式一致；identity/零/秩亏/mixed batch 正确；raw 输出、概率 None、tie-break 可验证 | R1–R2 |
| A2 | N=8/16/32 的 SVD VAMP 对独立 Cholesky oracle 逐层核对 xhat2、alpha2、c、r1/gamma1、后验和下一层消息；8/32 层配置明确区分；complex128 atol=1e-9、rtol=1e-8 | R3 |
| A3 | QPSK 对四点枚举均值/概率/方差一致；负/过低精度、非法分母、非有限候选、方差下溢、均值保持截断与保留旧消息直接测试；危险运算前分支，保护 backward 有限 | R4 |
| A4 | 零/秩亏/弱信道/mixed batch 分支正确；H,y 乘 a 且 sigma2 乘 abs(a)^2 时检测等价；零训练参数和禁止 API 检查通过 | R3–R6 |
| A5 | shape/dtype/device/finite/sigma2 失败矩阵通过；complex64 容差单独说明；可用 CUDA 一致性测试，无硬件明确 skip | R1、R6 |
| A6 | 至少一个非零时缩 WP3 真实 512 网格/400 未知量样本上两基线前向有限、输入 ID/hash 一致、标签隔离通过；不称完整系统 smoke 或三算法验收 | R7 |
| A7 | WP0–WP3 回归、Ruff/format/mypy 和独立 review 通过；保存实际命令、结果、环境、失败/跳过和源码指纹 | 全部 |

## Out of scope and deferred acceptance
不实现正式 PG-VAMP、训练/checkpoint/infer、统一 evaluate/report、性能 sweep、WP6 完整系统 smoke 或 WP5–WP8 任务。
不添加可选 SVD MMSE 或跨调用分解缓存，不运行主训练/主评测，不宣称 BER 增益或速度优势。
§23 PG 数学/梯度/全图极限由 WP5 验收；三算法共同输入由 WP5–WP7 补齐；公平计时及失败分母由 WP7 集成，不删除这些义务。

## Planning disposition
执行书与前序任务已确定产品范围，无阻塞性产品问题。技术边界及执行前核查见 design.md / implement.md。
规划已批准，进入 WP4 实施；实际结果以 research 验收证据为准。

## Verification disposition
WP4 A1–A7 已独立验收，313 passed / 5 CUDA skipped，静态检查通过。完整证据见 research/check-report.md。CUDA 无硬件；用户已确认工作提交及正常归档/journal，后续 WP 未启动。
