# WP8 完整交付与 CPU 验收

## Goal
让使用者能从交付文档安装、理解配置和三算法、复现有限规模完整链路，明确区分功能完成与充分性能验证。唯一权威为 `docs/CODEX_ENGINEERING_SPEC.md`，WP8 门槛见 :1887–1891。

## Authorization and dependency
2026-09-18 用户授权创建 WP8 child task，**仅 planning，不开始 implementation**。父任务为 `09-17-pgvamp-complete-engineering`；WP0–WP7 已归档，WP7 工作提交 f7f210f。保持 planning，不设置 active。本 PRD 描述后续获批实施的交付内容。

## Confirmed background
- 起始 HEAD 3c7d68f、main、工作区干净、无 active task。WP7 历史独立验收 478 passed / 7 CUDA skipped、17 条 CLI 成功；本轮未重跑。
- README.md:4 仍称当前 WP6，:53 把 WP7 列为未来工作，状态和归档证据链接需统一。
- 执行书 :1833 要求 demo-frame；src/pgvamp_ofdm/cli.py:70 起的命令注册没有该入口。WP1/WP2 审计脚本提供相关能力，不等于 CLI 已交付。
- src/pgvamp_ofdm/smoke.py:155 起已有物理审计、训练、三算法、checkpoint 和无标签推理，可复用。

## Requirements and acceptance criteria
| ID | Requirement | Observable acceptance |
| --- | --- | --- |
| A1 | 完整交付文档和命令（§§19–22） | README、配置说明、算法数学说明、测试记录、实现状态入口齐全且链接可解析；安装、inspect、两类 smoke、generate/simulate、audit、materialize、train/resume、infer、evaluate、benchmark、report、demo-frame 说明输入/输出/前置/规模。 |
| A2 | 配置与数学解释准确（§§2–17、20） | 明确 512/400/8192、400/64/47/1、CP2048/8 blocks、CPU/precision/profile、物理帧 split、理想 I/Q/同步/perfect CSI；MMSE 原始线性输出、精确 VAMP、PG 2T 参数/中心化残差/安全项/实际散度/方差校准/消息保护不偏离规格，不宣称稀疏加速。 |
| A3 | 完整帧演示 CLI（§21.6） | demo-frame 输出完整尺寸发送/接收波形、同步结果、400 维接收示例及配置/种子/环境/哈希；同步演示与 oracle_timing 分开；不同非零路径时缩下波形/H 一致性满足既有双精度门槛；错误配置和已有输出目录可读拒绝。 |
| A4 | 新的真实 CPU smoke（§§15.5、21.2、22） | 实际运行 512 grid/400 data/8192 FFT/CP2048/8 blocks/T8 smoke，含三算法、两次 PG 更新、checkpoint 往返和指标计数；保存命令/环境/退出码/产物/hash，独立检查保存结果。数学 smoke 不可替代。 |
| A5 | 最终回归及命令闭环（§23） | 全套适用测试、产品 Ruff/format/mypy、有限规模数据→审计→训练/恢复→无标签推理→评测/两种计时→报告通过；新增 demo 有实质测试；失败/skips 如实记录。 |
| A6 | 真实结论与总体验收（§§0.4、18、22–24） | 建立 WP0–WP8/§23 到证据的索引；功能和性能验证独立列示。主训练、完整 SNR sweep、充分多种子统计和 CUDA 实验未执行时标明“未执行”，保留适用命令/前置，不用占位曲线或历史次数宣称本次完成。 |

## Out of scope
本轮不改产品代码/文档/配置/执行书，不运行产品测试、smoke、训练、评测，不提交归档。后续 WP8 不改物理/算法公式、数据/checkpoint 格式和统计协议；不自动执行主训练、完整 sweep、CUDA 实验，不新增海试、信道估计或 FEC。

## Decisions and deferred items
源规格与父任务已明确范围，无阻塞性用户决策。demo-frame 属于既有交付缺口；资源密集实验保留命令及未执行状态。技术设计和步骤分别见 design.md、implement.md。实际验证结果等待未来实施。

## Implementation authorization — 2026-09-18
用户后续明确批准最新规划并授权进入 implementation。以上 planning-only 是创建时历史边界；当前已激活 WP8，按已批准规划实施并独立验收。主训练、充分 sweep 和 CUDA 实验仍不自动执行。
