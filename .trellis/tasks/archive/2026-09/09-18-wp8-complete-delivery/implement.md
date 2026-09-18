# WP8 execution plan — IMPLEMENTED / VERIFIED

## Activation gate
- [x] 用户后续明确批准最新 planning summary；已核对 WP7 归档、工作树和 planning/context。
- [x] 已 task.py start，进入 implementation，按 Trellis 注入分派 implement/check agents。

## Ordered work
1. [x] 建立 §§19–23/WP0–WP8 的文件/测试/命令/证据矩阵，区分功能缺口和实验未执行。（A1–A6）
2. [x] 统一 README/状态/验证当前叙述和归档链接，补配置、算法及复现文档，核对公式/参数/前置。（A1、A2、A6）
3. [x] 基于现有物理 API 实现 demo-frame；覆盖确定性、完整尺寸、非零时缩一致性、同步输出、错误配置/目录拒绝/CLI 测试。（A3）
4. [x] 执行 bounded 验收，保存真实 stdout/stderr/exit/环境/source hashes；修复后运行相关检查，最终全套回归。（A4、A5）
5. [x] 独立 check 审查产物、计数/hash/checkpoint、文档命令、图像、本地链接及 §23 映射。（A1–A6）
6. [x] 据实更新 VALIDATION、状态、父任务整体矩阵及必要 spec；未执行项明确保留，提交/归档按后续授权完成流程。（A6）

## Planned validation — not executed
PowerShell 先 `$env:MKL_THREADING_LAYER='TBB'`，`$py='.venv/Scripts/python.exe'`。输出目录须全新，已有目录改新后缀，不删除历史证据。

```powershell
& $py -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
& $py -m pgvamp_ofdm inspect-config --config configs/main.yaml
& $py -m pgvamp_ofdm smoke --config configs/smoke_math.yaml --device cpu --output runs/wp8-math
& $py -m pgvamp_ofdm smoke --config configs/smoke_system.yaml --device cpu --output runs/wp8-system
# demo-frame 是待实现入口
& $py -m pgvamp_ofdm demo-frame --config configs/cpu_dev.yaml --device cpu --dtype complex128 --output runs/wp8-demo
& $py scripts/wp6_acceptance.py --output runs/wp8-training-cli
& $py scripts/wp7_acceptance.py --output runs/wp8-evaluation-cli
& $py -m pytest -q --basetemp=runs/wp8-pytest -ra
python -m ruff check src tests
python -m ruff format --check src tests
& $py -m mypy src
git diff --check
```

WP6 harness 已包含两类 smoke，可复用该次新执行避免重复；WP7 harness 的少量帧和两种计时/单多 seed 报告不代表充分统计。新增/修改 scripts 单独纳入 lint/format。主训练/main sweep 文档仅静态核对，不执行。CUDA 无硬件明确 skip，不预填历史测试次数。安装按现有验证环境和 fresh 环境分别说明，不无故升级依赖。

## Review and rollback points
demo 编排/CLI 可独立回退；任何物理一致性或计数失败阻止完成声明，保留回执。最终全文检查当前状态和历史日期，父任务完成必须依赖 WP8 实际验收。


## Final gate — 2026-09-18
486 passed / 7 CUDA skipped；Ruff/format(src tests scripts)/mypy 通过。新训练/smoke 11 条 CLI、评测/计时/报告 17 条 CLI、两种精度 demo CLI 均通过；独立保存数组/计数/哈希/图像复核通过。源码及完整证据见 check-report.md 和根 VALIDATION.md。工作提交、归档、journal 待具体 commit plan 一次确认；task 保持 in_progress。

用户已于 2026-09-18 确认具体 commit plan，授权工作提交及正常 WP8 归档/journal。

工作提交 7102c97 已完成；按用户确认进行归档与 journal。
