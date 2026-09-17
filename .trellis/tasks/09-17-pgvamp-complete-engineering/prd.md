# PG-VAMP CP-OFDM Complete Engineering Project

## Goal and authority
从零构建执行书规定的 Python + PyTorch CP-OFDM/QPSK 仿真、物理信道、
三检测器、训练/恢复/推理、统一评测与报告闭环。

唯一工程实施规格：[docs/CODEX_ENGINEERING_SPEC.md](../../../docs/CODEX_ENGINEERING_SPEC.md)。
本文只组织目标、工作包和跨阶段验收，不能修改、弱化或重新解释数学合同、
通信参数、测试要求或顺序。详细公式见执行书 §§14–15，完整门槛见 §§22–23。
初次规划 source SHA-256：
A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B。

## Parent responsibility
父任务只保存总目标、执行书引用、WP 地图和跨阶段验收证据。
不直接作为代码实现任务，不分派父任务实现，不以父任务启动绕过子任务审核。
各 WP 的设计、实现计划、上下文和执行记录由对应子任务持有。
WP0 已完成归档；用户于 2026-09-18 批准 WP1 规划并授权实施。
WP2–WP8 只记录地图，不创建任务或代码。

## WP map — dependency order is mandatory
| WP | 责任与验收摘要（不得替代执行书 §22） | 前置 | 任务状态 |
| --- | --- | --- | --- |
| WP0 | 配置、设备、目录、独立 references、最小 CLI；派生参数与最小 reference 测试 | bootstrap 已审核 | [WP0 completed / archived](../archive/2026-09/09-17-wp0-foundation-references/prd.md) |
| WP1 | 固定 QPSK/分配、OFDM/CP/帧、LFM；映射/能量/索引/定位/帧长、实通带恢复、PSD/相关曲线 | WP0 验收 | [WP1 implemented / verified](../archive/2026-09/09-18-wp1-modulation-frame-lfm/prd.md) |
| WP2 | 仿射物理信道、CP 支撑、完整 H、独立波形、导频消除；非零时缩一致性/噪声 | WP1 验收 | 未创建 |
| WP3 | 紧凑数据、manifest、split、随机流、重放；无泄漏、同样本、容量估计 | WP2 验收 | 未创建 |
| WP4 | 完整 H 的 MMSE 与精确 VAMP；线性求解及 SVD/Cholesky 逐层等价 | WP3 验收 | 未创建 |
| WP5 | 正式 PG-VAMP 数学合同；矩阵/Jacobian/梯度/极限与 2T 参数 | WP4 验收 | 未创建 |
| WP6 | CPU 默认训练、显式 CUDA、checkpoint/恢复/推理；两类 smoke | WP5 验收 | 未创建 |
| WP7 | 配对评测/统计/计时/稳定性/报告；真实计数和失败可追溯 | WP6 验收 | 未创建 |
| WP8 | 文档、状态、真实 CPU 完整尺寸 smoke；未运行实验明确标注 | WP7 验收 | 未创建 |

Trellis parent/child 链接不是依赖调度器。后续子任务必须显式写入上表前置，
通过前一阶段验收并获得相应授权后推进，不能并行跳过 WP 顺序。
WP0 reference 是执行书要求，不代表提前实施 WP4/WP5 正式检测器。

## Cross-stage acceptance
- 配置与通信维度不漂移：96 kHz、21–27 kHz、512 网格、8192 FFT、
  400/64/47/1、CP 2048、8 数据块、固定未编码 QPSK（§§2–4、20）。
- CPU 默认，CUDA 显式启用且不可用报错，complex128 正确性基准；
  complex64 另标实验，不自动跨设备或改变 checkpoint dtype（§§15、21）。
- 独立 waveform_reference 与同物理参数 effective_fast 在非零时缩、
  有效 CP 下先通过一致性，再用于主训练/评测；随机矩阵只作代数 fixture（§6）。
- 独立本地 references 保持正确性与可读性；正式 PG-VAMP 逐层前向及参数
  梯度等价、正式 VAMP SVD/Cholesky 等价；不削弱基线（§§0.3、12–14、23）。
- PG-VAMP 只训练 2T 实标量，保留中心化完整残差、解析安全项、实际条件
  散度、方差校准及消息保护；公式不得在下游文档重解释（§14）。
- 按物理帧/信道 split；稳定独立随机流、manifest/checksum、完整恢复状态；
  三算法共享 H/y/sigma2/sample IDs/hash，标签不进入检测器（§§10–11、15）。
- 指标以真实计数为依据，按帧 bootstrap、配对比较；计时口径公平，
  硬失败不从分母静默消失，零错误不解释为零真实 BER（§§16–18）。
- 数学 smoke 不能替代真实 512/400 尺寸 smoke；执行全部适用 §23 测试。
  阶段性延期测试要指明所属 WP，不能永久删减验收。
- 只保存实际执行命令、环境、结果及产物；源码完成与充分性能验证分别验收。
  未执行主训练/sweep 标注“未执行”；执行书 §24 不算本工程测试（§§18、22–24）。

## Review state
WP0 已提交为 ecbda13 并归档（4b90d5f），归档 task.json 为 completed。
历史 CPU 目标/全量各 75 passed、1 skipped（无 CUDA），Ruff/format/mypy 通过；
实际回执与源码指纹见归档 WP0 research/final-validation.json 和 final-source-manifest.json。
这些为已读取的历史证据，本次未重跑，不构成 WP1 验收。
WP1 最新规划已获用户明确批准并实施；独立完整检查通过，目标 91 passed/1 skipped，
全量 166 passed/2 skipped（CUDA 无硬件），Ruff/format/mypy 与双 CLI 通过。
完整尺寸 complex128/complex64 发送审计均为 6400 bits 零错误、模板起点零偏差。
保存波形的种子重放、NumPy FFT/相关、PSD 积分与哈希核对均通过。
实际证据位于 WP1 research/check-review.md、check-validation.json、check-artifacts.json；
用户已确认工作提交方案，按完成流程将证据保留在 archive/2026-09/ 下；
不把无信道恢复当作完整物理信道或系统验收。
父任务保持总规划容器；WP2–WP8 未创建或启动。
