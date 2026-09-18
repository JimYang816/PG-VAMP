# WP4 execution plan — IMPLEMENTED / VERIFIED

## Authorization gate
- [x] 用户授权创建 WP4 child task，仅 planning。
- [x] 检查执行书、前置归档、现有代码边界和 oracle。
- [x] 用户后续明确批准最新规划并授权 implementation（2026-09-18）。
- [x] 重新核查源 SHA、git 差异及 WP3 前置；完成无信息误差界技术核查。
- [x] 上述 gate 通过后才 task.py start，并按 Trellis implement/check 子代理流程实施。

## Ordered implementation — completed through review
1. [x] 实现结果/API、输入边界、生产 QPSK/messages；四点枚举及保护测试先独立验证，reference 算子不共享。
2. [x] 实现 Cholesky MMSE 与 A1：solve/残差/对角/零/秩亏/mixed batch、raw 输出、None 概率和 tie-break。
3. [x] 实现精确 SVD VAMP、诊断及 8/32 层配置；验证每 detect 一次 SVD、零训练参数、无隐式缓存。
4. [x] 实现 A2–A5：逐层 oracle、保护分支、dtype/device/错误、尺度不变性、弱信道和禁止 API。
5. [x] 实现 A6：临时目录生成固定种子非零时缩 WP3 小量物理数据，400 维两基线同一 payload；检查 ID/hash、输入不变、替换或移除 target 不影响结果。
6. [x] 保存实际证据；独立 check 子代理核查全部变更、源合同和 A1–A7。修复后重跑受影响检查与必要回归。
7. [x] 更新 README/VALIDATION 和适用 specs，仅记真实结果。
8. [x] 用户已确认工作提交清单及后续归档/journal，按正常完成流程执行。

## Validation commands — execution receipts in research
预计新增 tests/test_mmse.py、test_vamp.py、test_detector_qpsk.py、test_detector_inputs.py；可按职责拆分，不减少验收。

```text
python -m pytest -q tests/test_mmse.py tests/test_vamp.py tests/test_detector_qpsk.py tests/test_detector_inputs.py
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python -m pgvamp_ofdm inspect-config --config configs/vamp_reference_32.yaml
```

A1→MMSE；A2→VAMP 逐层；A3→posterior/messages 和保护 backward；A4→边界/尺度/API/参数；A5→输入和设备；A6→物理集成；A7→回归和独立 review。
记录实际命令、环境、dtype/device、耗时、失败/skip、产物哈希和源码指纹。无 CUDA 明确 skip。
complex128 atol=1e-9/rtol=1e-8；complex64 独立注明容差。条件数相关调整必须有诊断依据，不为通过错误实现放宽。

## Review and rollback
reference 不得导入生产数值算子；oracle 改动须单独论证。无信息分支不得隐藏硬失败/抹掉弱信号。
核查真实 alpha1、保持候选均值、危险 reciprocal 前分支、无 forward detach，未来 PG 可复用且梯度有限。
不提前创造三算法指标、失败分母或计时结论；这些保留 WP7。阶段回退只撤销本 WP 变更，不删除 WP3 数据或降低旧测试门槛。
最终独立检查通过：313 passed / 5 CUDA skipped，Ruff/format/mypy、配置检查、独立 NumPy 逐层及真实物理标签隔离审计通过。完整证据见 research/check-report.md。用户已确认工作提交及正常归档/journal；WP5 未启动。
