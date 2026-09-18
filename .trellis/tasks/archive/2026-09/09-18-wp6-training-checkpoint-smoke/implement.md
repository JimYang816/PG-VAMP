# WP6 execution plan — APPROVED

## Gate
- [x] 用户授权创建 child task，仅 planning；核对前置和接口，完成 PRD/design/context。
- [x] 用户于 2026-09-18 明确批准最新 planning summary 并授权 implementation。
- [x] 获批后复核 git/source/spec 指纹与 WP5 前置，启动本 child task，按 Trellis implement/check 子代理流程执行。

## Ordered work
1. [x] A1/A2：兼容的可微逐层输出与完整小型诊断；独立 loss 手算/梯度/2T 测试。先跑 WP5 数学回归。
2. [x] A3：安全 checkpoint schema、RNG/sampler/optimizer 与 atomic last/best；覆盖损坏/缺字段/类型/映射/depth/dtype/hash 错误。
3. [x] A1–A4：deterministic batches、Adam/clip、val NMSE/best、early-stop、JSONL 与失败处理。连续 4 步 vs 2+resume 2 步，对比参数、optimizer、样本顺序、下一 RNG 值；覆盖 epoch 边界/尾 batch/验证事件。
4. [x] A4/A5/A7：train/resume/infer CLI、种子与 provenance；无标签/改标签一致、dtype 继承/转换、CPU forbidden CUDA、CUDA 不可用及有硬件条件测试。
5. [x] A6/A7：数学/完整系统 smoke、最小计数和人工错误样例；真实执行两类 CPU smoke 及 simulate/audit/train/resume/materialize/infer 小闭环，保存维度/更新数/产物 hash。
6. [x] 独立 check：核对 §§15/21/22 WP6/23 与回归，保存真实命令/环境/结果/失败/skip。统计/报告等剩余 §23 项仍归 WP7，不降低门槛。见 research/check-report.md：430 passed / 7 CUDA skipped，lint/format/mypy、11 条 CLI 和独立产物核验通过。
7. [x] 更新必要使用说明、IMPLEMENTATION_STATUS.md、VALIDATION.md 和稳定接口 specs，main/full sweep 标注未执行；回到用户审核，不自动创建 WP7。

## Planned checks — not run during planning
测试文件名为拟新增，实施后保存实际命令：
```powershell
python -m pytest -q tests/test_training.py tests/test_checkpoint.py tests/test_inference.py tests/test_smoke.py
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python -m pgvamp_ofdm inspect-config --config configs/smoke_system.yaml
python -m pgvamp_ofdm smoke --config configs/smoke_math.yaml --device cpu
python -m pgvamp_ofdm smoke --config configs/smoke_system.yaml --device cpu
```
实际实施结果见 research/implementation-evidence.md 和 implementation-cli-receipts.json。

小闭环使用隔离产物/小配置：simulate → audit-data → train(max_steps=2) → resume(max_steps=4) → materialize --split val --without-labels → infer；记录完整实际参数。未运行默认 cpu_dev/main 大训练；实际小闭环已完成。
complex128 atol=1e-9/rtol=1e-8；物理 reference relative error <1e-9。complex64/CUDA 容差给出依据，不为失败任意放宽。硬件缺失显式 skip。

## Rollback/review
模型输出扩展后先过 WP5；checkpoint/采样独立验证后再接物理训练；完整尺寸 smoke 资源失败如实记录、不降维冒充。只回退本 WP 编辑，不删除产物或改 oracle/规格。此前 planning 轮未执行功能测试；获批 implementation 后已执行上述检查和小闭环，独立 check 已完成，证据见 research/check-report.md。

## Final execution state
Independent review passed: 430 passed / 7 CUDA skipped; lint, format, mypy and 11 CLI commands passed. See research/check-report.md and main-final-verification.json. User approved work commit followed by normal archive/journal on 2026-09-18; executing approved completion. WP7 not started.
