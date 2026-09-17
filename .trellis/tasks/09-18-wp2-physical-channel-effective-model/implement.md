# WP2 execution plan — approved for implementation

## Entry gate
- [x] 创建 planning-only child task，检查父任务、源规格与 WP1 历史证据。
- [x] 整理 PRD、design、执行计划和真实 spec/research JSONL。
- [x] 用户在最终规划摘要之后明确授权 implementation。
- [x] 重新核对工作区、WP1 前置和规划；授权后已运行 task.py start。

## Ordered implementation
1. [x] 加载适用 backend 指南，分派 Trellis implement agent，限 WP2 新模块、必要接入点和测试。
2. [x] parameters/validity：五类路径、显式 RNG、输入校验、CP 端点/偏移；覆盖 A1/A2。
3. [x] continuous/affine：独立分段波形与时缩，先验证整数点 WP1 恢复，再测非整数点和边界；A4。
4. [x] effective_matrix/fft_receiver：闭式、静态/identity 极限、实际 FFT、首块/末块与偏移窗口非零时缩一致性；A1/A3。
5. [x] preprocessing/noise：400 行、强导频泄漏、时间 AWGN 与统计/BER 验证；A5/A6。BER 使用硬 QPSK，不提前实现 MMSE。
6. [x] 专用 run_wp2_audit.py 保存实际参数/种子/窗口/误差/统计/版本/源哈希；更新 VALIDATION.md、IMPLEMENTATION_STATUS.md 和必要模型说明。
7. [x] Trellis check agent 独立审核 A1–A7、参考独立性、全量回归及审计，保留实际失败/修复记录。
8. [x] 主会话审阅证据、更新长期 specs 和父任务状态。
9. [x] 用户已确认工作提交方案及正常完成流程；执行提交/归档/记录，不启动 WP3。

最终独立检查：28 passed/1 skipped 定向，194 passed/3 skipped 全量；
产品 Ruff/format/mypy 通过。20 窗口 complex128 最大误差 8.152338913635682e-12，
complex64 最大误差 4.6309497747643036e-7。完整 A1–A7 映射和限制见
research/check-report.md；主会话指纹核对见 research/main-final-verification.json。

## Validation commands
以下同步为实际落地文件名；执行结果以 research 内机器回执为准。
basetemp 与审计 output 每次使用新目录，避免清理或覆盖已有证据。

```powershell
$env:MKL_THREADING_LAYER='TBB'
.venv/Scripts/python.exe -m pytest -q tests/test_wp2.py --basetemp runs/wp2-target-UNIQUE
.venv/Scripts/python.exe -m pytest -q --basetemp runs/wp2-full-UNIQUE
C:/Software/Anaconda3/Scripts/ruff.exe check src tests scripts/run_wp1_audit.py scripts/run_wp2_audit.py
C:/Software/Anaconda3/Scripts/ruff.exe format --check src tests scripts/run_wp1_audit.py scripts/run_wp2_audit.py
.venv/Scripts/python.exe -m mypy src
.venv/Scripts/python.exe scripts/run_wp2_audit.py --config configs/cpu_dev.yaml --output runs/wp2-audit-UNIQUE
```

仓库级 `ruff check .` 已尝试，既有 Hook/Trellis 文件报错，未修改无关代码；
本任务检查范围覆盖全部 src/tests 与两个产品审计脚本。全量 pytest 使用工作区
basetemp 绕过系统临时目录权限限制；保留失败与修正命令的回执。

tests/test_wp2.py 内测试映射：parameters/effective→A1；validity/affine→A2；effective/fft/audit→A3；
continuous→A4；preprocessing→A5；noise→A6；全量、静态检查、CPU 审计→A7。
CPU complex128 为主，complex64 独立误差报告，CUDA 无硬件明确 skipped。
方差/协方差阈值按固定样本数与解析统计误差设定，BER 按 bit 数和二项统计设容差，
保存错误数。至少两条不同非零 epsilon、默认 8192 FFT 和全部 512 网格不可省略。

## Review and rollback gates
物理一致性失败停在步骤 4，不进入 WP3、不放宽阈值、不截断 ICI。
导频/噪声失败停在步骤 5，不声称满足白噪声检测模型。
config.py 与 WP1 waveform/receiver 基础接口是高风险接入点，需完整回归；
新增模块分步可回滚，不重写归档 WP1 证据。
§23.1 重放/split→WP3；算法→WP4/WP5；完整系统 smoke→后续 WP，保留总验收义务。
