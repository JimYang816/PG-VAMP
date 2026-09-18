# WP7 — 配对评测、统计、计时与报告

## Goal and authority
交付可复核的三算法公平比较闭环：真实逐帧计数、配对统计、明确计时边界、稳定性诊断及只引用实际结果的报告。唯一规格为 docs/CODEX_ENGINEERING_SPEC.md §§16–18、21.6、22 WP7、23.3；本文不得降低其要求。

## Background and authorization
父任务为 09-17-pgvamp-complete-engineering。前置 WP6 已验收归档，工作提交 e73a983，历史独立验收 430 passed / 7 CUDA skipped，本轮未重跑。WP3 已有物理帧/SNR 副本和稳定重放，WP4/5 已有完整 H 检测器，WP6 已有严格 checkpoint 与推理；正式 evaluate/report/benchmark 尚缺。
2026-09-18 用户明确授权创建 WP7 child task，**只做 planning，不开始 implementation**。保持 planning、不设 active，不运行 task.py start。

## Requirements and acceptance
| ID | 要求及可观察验收 | 来源 |
| --- | --- | --- |
| R1 | 固定帧数；三算法相同 sample IDs、H/y/sigma2 哈希、完整 H、device/dtype/线程/前处理。修改 target 不改变预测。主协议 ideal_complex_iq/oracle_timing/perfect CSI/known noise/无编码，PG T8 与 VAMP 8 次，MMSE 一次精确估计。保留 cpu_dev/main 场景、SNR 和规模标签。 | §§11、16、23.3 |
| R2 | 手工预测验证 BER/SER/BLER/FER、符号 NMSE/EVM 与 goodput；整数错误数和分母可复算；8 块为完整帧，合计能量后算 NMSE/dB，不做标签幅相对齐。 | §17.1 |
| R3 | BER/SER 95% 区间按独立帧成簇 bootstrap，默认 2000 次、固定记录种子；算法差值共用重采帧。真实零错误保留 0/已测分母，log 标记不改 CSV；少帧标区间不稳定，零 FER 上界按源公式，不能无条件给相关位错误套 3/bit_count。 | §17.5 |
| R4 | NaN/Inf/分解失败保留 ID、原因、计划分母和成功数量，格状态 incomplete_or_failed；成功子集不得冒充完整指标。合法消息拒绝正常计数。若报告同步模式，须同时保留全体 FER 与同步成功条件 BER，不丢漏检帧。 | §17.6 |
| R5 | benchmark 支持 per_observation_cold_H 与 same_H_amortized 两个正式命名口径，包含声明的分解/图/层/判决，复用另列首次准备与后续成本。B1 mean/median/p95、固定 batch 吞吐、共同前处理、CPU 进程峰值/CUDA 峰值 allocated、矩阵工作内存、参数数和完整计时元数据可核验；区分 batch 摊销、计算吞吐与链路 goodput。 | §§17.2–17.3 |
| R6 | 保存硬失败/不可用数量；VAMP/PG 消息拒绝/精度截断/无信息率及机会分母；PG 每层门限、mu、两种有效边比例、安全项、c；网格 ICI 与有效矩阵 ICI 区分命名。 | §17.4 |
| R7 | evaluate 生成 §18 全部基础产物、逐帧/聚合表；report 仅读持久结果生成全部适用图和 REPORT.md，不训练、调参或重跑检测。报告说明假设、真实环境/训练步数/种子/规模/基线、性能时间稳定性、门控变化与局限；缺数据写“未执行”，不造优势或曲线。 | §18 |
| R8 | §21.6 evaluate/report CLI 及独立 benchmark 可用；PG 缺 checkpoint 默认报错，仅 --allow-untrained 可用 PG-VAMP-untrained。支持 CLI 指定种子及多训练种子结果合并，拒绝不兼容实验；单 seed 不制造跨 seed 均值/标准差。 | §§16.2、21.6 |
| R9 | CPU 默认且不调用 CUDA；显式不可用 CUDA 报错；精度实验分开，checkpoint/物理配置兼容。真实 CPU 512/400 小闭环及独立产物核验、相关 §23.3 测试和完整回归通过。 | §§15、21–23 |

## Out of scope and evidence limits
本轮不编辑产品代码，不训练/评测/benchmark，不提交归档。后续 WP7 不改变物理/检测器数学合同，不增加自适应停止、test 调参、hard mask，不启动 WP8。主评测限定 oracle_timing，不扩展 WP3 当前不支持的 lfm_detect 紧凑重放。
完整主训练和充分 SNR sweep 未执行时明确记录“未执行”；小闭环只能验收功能，不能宣称性能充分验证或 PG 优势。

## Planning convergence
范围及验收由源规格与本次授权确定，无阻塞性用户决策。技术风险见 design.md，执行顺序见 implement.md。后续明确批准最新规划后才可进入 implementation。

## Implementation approval
2026-09-18 用户后续明确批准最新规划并授权 implementation。前述 planning-only 为创建时历史边界；现已激活 WP7，按本规划实现并独立验证，WP8 仍未授权。
