# WP7 execution plan — IMPLEMENTED / VERIFIED

## Entry gate
- [x] 用户授权 child planning；核对 WP6 历史验收与父子关系。
- [x] 完成 PRD 收敛、设计、计划、证据及 JSONL 上下文。
- [x] 2026-09-18 用户明确批准最新规划并授权 implementation。
- [x] 已刷新上下文并 task.py start；进入 implement，随后独立 check。

## Ordered implementation
1. [x] R1/R8/R9：固定 test 清单、checkpoint/物理/manifest 校验、共享输入 lineage、evaluate CLI；验证 8 块/SNR/ID/hash/标签隔离/缺 checkpoint。
2. [x] R2/R4：纯计数/能量/完整帧聚合、failure ledger 和原始分母；多帧手算与故障注入验证。
3. [x] R3：bootstrap、配对差值、零值/少帧/FER 上界；验证固定种子、帧粒度与共用索引。
4. [x] R5：两计时口径、prepared-state/失效；先核验 oracle 和 cached/uncached，再测时；补 B1/batch、同步、分解调用边界、前处理/内存口径。
5. [x] R6/R7：诊断、真实示例、版本化 §18 全产物、完整性/原子发布；独立复算计数、hash、机会分母。
6. [x] R7/R8：只读报告、图集/配对表/多 seed 合并；探针禁止 report 训练检测，验证零值不改 CSV、失败/缺失/单 seed 不造结论。
7. [x] R9：真实 CPU 512/400 最小完整帧闭环，覆盖 generate/audit/train/evaluate/benchmark/report/infer；必要时两步 checkpoint，保留实际回执。只减帧数/更新，不减 512/400/T8/8 块。
8. [x] 独立 check 对照完整 §§16–18、21.6、22 WP7、23.3，目标测试及全量回归；记录实际环境/argv/结果/skip/失败/产物 hashes。
9. [x] 已更新 CLI 说明、IMPLEMENTATION_STATUS.md、VALIDATION.md 和 WP7 稳定接口 spec。
10. [ ] 用户已批准具体 commit plan，正在执行提交、归档与 journal；不自动启动 WP8。

## Planned commands — 未执行
目标测试文件为拟新增，实际命令在执行回执固化。
```text
python -m pytest -q tests/test_evaluation.py tests/test_statistics.py tests/test_benchmark.py tests/test_reporting.py
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python -m pgvamp_ofdm inspect-config --config configs/main.yaml
python -m pgvamp_ofdm evaluate --config configs/cpu_dev.yaml --manifest data/cpu_dev/manifest.json --checkpoint runs/pg_cpu_dev/best.pt --algorithms mmse vamp pg_vamp --device cpu --output results/compare_cpu_dev
python -m pgvamp_ofdm report --results results/compare_cpu_dev
```
完整 cpu_dev 命令仅为复现示例，未宣称已执行全部规模。最小闭环用持久专用配置及独立输出；保存 generate/audit/train/infer、两种 benchmark 和多 seed report 的实际 argv。benchmark 参数实现时按 §17.3 固化并保存 help；未实现命令不得算通过。

## Review and rollback
缓存适配必须保持原预测；失败则回退适配，不修改 oracle。数据/config/schema 变化先审兼容，再跑 WP3/WP6 回归。CPU-only 下 CUDA 数值项明确 skip，不用 mock 冒充硬件证据。错误实验标失败，保留用户数据。功能通过与充分 BER/性能验证分开，主训练/完整 sweep 未执行时如实标记。

## Final review result
独立检查通过：478 passed / 7 CUDA skipped；Ruff/format/mypy 通过，17 条真实 CLI 命令通过。源规格和独立 oracle 未修改。计数/哈希与图像复核通过；详见 research/check-report.md。未运行主训练、充分 sweep 或 CUDA 数值实验。
