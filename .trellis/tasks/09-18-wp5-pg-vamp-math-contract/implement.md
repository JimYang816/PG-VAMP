# WP5 execution plan — implemented / independently verified

## Authorization gates
- [x] 用户授权创建 child task，仅 planning；核对源 SHA、WP4 归档和当前接口。
- [x] 编写 PRD/design/执行计划、研究依据和双上下文清单。
- [x] 后续用户明确批准最新规划并授权 implementation（2026-09-18）。
- [x] 实施前核对 git diff、源 SHA 与 WP4 前置；trace 数值界按 design 在实现中独立审计。
- [x] 上述通过后已 task.py start，按 trellis-implement / trellis-check 子代理流程实施。

## Ordered work
1. [x] topology/majorizer/初始化，A1/A2：2T、门限、图、独立 ell、PSD/Loewner。
2. [x] linear 单层及错误/无信息分支，A3/A4：同层 factor 复用、B/目标/Jacobian/协方差。
3. [x] model/forward/detect、生产后验/消息复用、导出和诊断。
4. [x] A5/A6：N=8/16/32 逐层 DensePGVAMP 前向及两组参数梯度、gradcheck/整网反向；
   全图同时对照精确 SVD VAMP 和独立 Cholesky VAMP；对角/尺度/秩亏/混合 batch。
5. [x] A7/A8：保护分支有限 backward、极端尺度、输入/dtype/device/CUDA（无硬件明确 skip）、
   显式 jitter 一致性、禁止 API、一次 Cholesky/层及无跨调用旧图缓存。
6. [x] A9：临时生成固定种子 WP3 非零时缩样本，T=8/N=400 三算法相同 payload；
   每个检测器重新加载新鲜样本，核对原/改动/移除标签时的 ID、输入和全部预测哈希。
7. [x] A10：保存真实证据，独立 check 全范围复核，修复后跑受影响检查和全量回归；
   更新 README/VALIDATION 及适用 specs，明确未执行训练和系统 smoke。
8. [x] 用户已确认工作提交清单及后续归档/journal（2026-09-18）；按正常完成流程执行，不创建或实施 WP6。

## Planned commands — not executed in planning
拟新增 test_pg_vamp.py、test_pg_vamp_properties.py、test_pg_vamp_gradients.py，
扩展 test_detector_inputs.py；可按职责调整文件名，不减少 A1–A10。

```text
.venv/Scripts/python.exe -m pytest -q tests/test_pg_vamp.py tests/test_pg_vamp_properties.py tests/test_pg_vamp_gradients.py tests/test_detector_inputs.py tests/test_reference.py tests/test_vamp.py tests/test_detector_qpsk.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src tests
.venv/Scripts/ruff.exe format --check src tests
.venv/Scripts/python.exe -m mypy src
.venv/Scripts/python.exe -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python .trellis/scripts/task.py validate .trellis/tasks/09-18-wp5-pg-vamp-math-contract
```

A1/A5/A6→模型/梯度；A2/A3/A4→性质；A7→模型及公共后验回归；
A8/A9→输入/物理；A10→全量检查/独立 review。CUDA 无硬件明确 skip。
独立 review 另以 NumPy/小矩阵 solve 核对安全项、B 和完整协方差，不能只有生产/oracle 相符。

## Review and rollback points
梯度对照使用相同非平凡 raw 参数、同一实值测试损失；同时比较两组梯度，不能只看有限/非零。
gradcheck 避开分支边界，边界单独测有限 backward。全图/对角零梯度是正常情形。
Loewner 比较有序 mask 的 Gbar；目标不增仅指固定单层，不要求整网 BER 单调。
沿用 PRD 源容差；条件数调整须记录，不得通过 jitter/detach 或弱化基线掩盖错误。
风险文件为新 pg_vamp、algorithms 导出及可能改动的公共 helper；reference 原则上只读。
技术审计若推翻设计或需要改 oracle，记录依据并回规划审核。
A9 不含训练/checkpoint/指标闭环，不能命名完整系统 smoke。

## Final review result
376 passed / 6 CUDA skipped；Ruff/format/mypy、配置及上下文检查通过。
独立 NumPy 审计最大绝对差 1.0303e-13；真实 400 维三算法输入及全部输出标签隔离通过。
极弱非对角 backward 和诊断尺度修复均有回归。oracle 同类缺陷经过独立复现、源依据、
协调者审核后，仅修复两处 quotient 求值，未共享生产数值逻辑。见 research/check-report.md、
check-oracle-correction.md 及 bug-retrospective.md。WP6 和性能实验未执行。
