# WP1 execution plan

## Gate
- [x] 用户明确授权创建 child 并只规划。
- [x] 执行书相关章节、现有 specs、WP0 实现/测试/回执已读取。
- [x] PRD 收敛，design/implement 和 JSONL spec/research 上下文已准备。
- [x] 用户于 2026-09-18 明确批准本次最终规划并授权实施。
- [x] 执行书哈希未变，工作区仅有本任务规划改动；task.py start 已进入 Phase 2。

当前已进入实施；产品验收以本轮实际执行回执为准，不创建 WP2。
planning-validation.md 是前一轮规划检查历史，不声明产品验收。

## Ordered implementation after approval
1. [x] 加载 Phase 2 step，用 trellis-implement；dispatch 首行指定当前 child path，
   遵循 source reading contract，记录起始 Git/环境。
2. [x] qpsk.py、allocation.py 与独立枚举/精确索引测试。（A1,A2）
3. [x] ofdm.py：最后维 FFT、Parseval、CP、平均功率测试。（A3）
4. [x] lfm.py：离散窗、相位与固定功率归一化。（A5）
5. [x] frame.py：FrameLayout、解析/实通带、offset padding；
   独立 FFT 恢复和全局相位、8 块全尺寸帧测试。（A3,A4）
6. [x] synchronization.py：直接相关对照、边缘 lag、全零、纯噪声、
   拒绝分支及候选有效性。（A6）
7. [x] scripts/run_wp1_audit.py：保存波形、PSD/带外能量、相关曲线、
   原始数据及配置/环境/种子/哈希。（A7）
8. [x] 非法输入、CPU 无 CUDA、显式 complex64、条件 CUDA 测试；
   目标测试、全量 WP0 回归、lint/format/type checks。（A8）
9. [x] trellis-check 完整 PRD 复核；实际打开 PNG 看坐标/标签/范围，
   同时核验原始数值；不以画图成功代替数值验收。（A1–A8）
10. [x] 更新 README、IMPLEMENTATION_STATUS、VALIDATION 与必要长期 specs；
    记录真实命令/退出码/误差/计数/环境/失败/跳过/未执行项；
    返回父任务审核 WP1，不自动进入 WP2。

## Validation commands and execution receipts
以下为原计划命令；对应真实执行记录见 research/implementation-validation.json、
research/check-validation.json。最终使用独立的新 basetemp 和 audit 输出目录，
保留早期回执，精确参数以 JSON 回执为准。
PowerShell，复用 WP0 .venv，不升级无关依赖：
```powershell
$env:MKL_THREADING_LAYER='TBB'
.\.venv\Scripts\python.exe -m pytest -q tests/test_qpsk.py tests/test_allocation.py tests/test_waveform.py tests/test_lfm.py --basetemp=runs/wp1-targeted -ra
.\.venv\Scripts\python.exe scripts/run_wp1_audit.py --config configs/cpu_dev.yaml --output runs/wp1-audit
.\.venv\Scripts\python.exe -m pytest -q --basetemp=runs/wp1-full -ra
python -m ruff check src tests scripts/run_wp1_audit.py
python -m ruff format --check src tests scripts/run_wp1_audit.py
.\.venv\Scripts\python.exe -m mypy src scripts/run_wp1_audit.py
.\.venv\Scripts\python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
.\.venv\Scripts\pgvamp-ofdm.exe inspect-config --config configs/cpu_dev.yaml
git diff --check
```
每次重跑 pytest 使用新的 basetemp 目录，避免清理旧证据。新脚本/测试已实现。
CUDA 无硬件记录 skip，mock 不是硬件计算。
不运行 generate/audit-dataset/train/evaluate/report/infer/full smoke；
WP1 审计脚本不替代后续数据集审计和主评测 CLI。

## Review and rollback points
- R1–R8/A1–A8 逐项有真实断言/审计；PSD/相关图附原始数组。
- 核验 512/8192/400、64 pilots、不按数据标签归一化。
- 不把 reference 后验迁入生产共享模块，不改保护/梯度合同。
- 新模块按步骤 2–7 分别修复/回滚；若须大改配置/CLI 返回规划审核。
- 保留历史运行记录，不递归清空工作区，不改源执行书。
- 同步失败不隐式 oracle 校正；WP2 H/CP 支撑仍未执行。

## Final WP1 result and stop boundary
完整独立检查通过，A1–A8 全覆盖，无遗留范围内问题。目标 91 passed/1 skipped，
全量 166 passed/2 skipped；CUDA 无硬件。Ruff/format/mypy、双 CLI 通过。
两种精度的完整发送审计均恢复 6400 bits，错误位和模板起点误差均为 0。
实际回执、修复记录和重放检查见 research/check-review.md、check-validation.json、
check-artifacts.json；主会话图表/校验和复核见 main-artifact-review.json。
完成的是 WP1，无信道回环不能替代 WP2 物理模型或完整 smoke。
用户已确认 Phase 3.4 提交方案；完成工作提交后归档并记录日志，未创建 WP2。
