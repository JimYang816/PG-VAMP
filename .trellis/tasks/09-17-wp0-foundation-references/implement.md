# WP0 implementation plan — NOT EXECUTED

## Gate before any code
- [ ] 用户在本轮规划展示后明确批准最终 PRD/design/implement。
- [ ] 再核对任务保持正确 parent、执行书未改变、工作区已有改动与环境。
- [ ] 按 research/source-reading-contract.md 分段完整读取原执行书；自动注入受限，不得只读前缀。
- [ ] 重跑本任务 task validation；确认 implement/check 上下文可读。
- [ ] 只有届时才允许 task start 激活本 child；不得激活父任务实现。
本轮不运行上述 start，也不分派实现或写产品代码。

## Ordered steps after approval
1. [ ] 盘点 Python/PyTorch 及所需 API、Git/源文档哈希。
   记录无提交/dirty 状态，确定源文件哈希清单；不无故升级依赖。（A1,A7）
2. [ ] 建立最小 src package、pyproject.toml、pytest 配置、CLI 双入口、
   ignore 规则和状态文档。只创建实际需要的模块，不补 WP1–WP8 空壳。（A1）
3. [ ] 按 §20 建立严格配置 schema、base/profile、递归未知键校验、
   物理/代数模式和派生摘要；测试错误配置和配置 roundtrip。（A2,A3）
4. [ ] 实现 CPU 默认/device/dtype/thread 选择；CPU 不触 CUDA；
   CUDA 请求失败可读报错，显式 complex64 配对正确。（A4）
5. [ ] 实现 inspect-config、可选输出目录与 provenance/resolved config。
   核对所有 §21.1 派生值与容量估计，保证无仿真/训练副作用。（A2,A7）
6. [ ] 独立实现 dense_vamp_cholesky，再实现 dense_pg_vamp；直接依据
   §§13–14，保留完整层循环和保护/梯度，不导入 production algorithms。
   初步测试 identity/diagonal/zero/rank-deficient 和非退化 fixture。（A5）
7. [ ] 增加 QPSK 枚举、全图极限、PG 2T 参数、反向/小规模 gradcheck、
   protection 和 dtype/device 测试；逐项源码审阅禁止 API。（A5,A6）
8. [ ] 执行完整 WP0 pytest 和 CLI 验证，保存真实命令/退出码/测试汇总/
   环境/源码与执行书哈希；更新 README/STATUS/VALIDATION。（A1–A7）
9. [ ] 按完整 PRD 做实现与检查复核；只将 WP0 范围标为已验收，
   列出未执行硬件/物理/生产等价/训练/评测项；回到父任务记录证据，
   不自动创建或实现 WP1。（A7）

每一步失败先记录原因并修复，不绕过下一阶段依赖。
WP0 两 reference 的最小正确性测试必须真的运行，不能用 import-only 测试完成 A5。

## Planned validation commands (all product commands UNEXECUTED)
在实际选定且记录的 CPU 环境中：
```text
python -m pip install -e ".[test]"
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
pgvamp-ofdm inspect-config --config configs/cpu_dev.yaml
python -m pgvamp_ofdm inspect-config --config configs/main.yaml
python -m pgvamp_ofdm inspect-config --config configs/smoke_math.yaml
python -m pgvamp_ofdm inspect-config --config configs/smoke_system.yaml
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml --output runs/wp0-inspect
python -m pytest -q tests/test_config.py tests/test_devices.py tests/test_cli.py tests/test_reference.py
python -m pytest -q
```
安装前核查既有环境；缺失依赖才补齐，不无理由升级。命令成功与否以执行记录为准。
测试中构造坏配置/不可用 CUDA 情形并断言错误，CUDA 硬件计算只在实际可用时运行。
无 CUDA 不阻止 CPU WP0 验收，但不得报告 CUDA 数值等价已通过。
不运行 smoke 子命令、simulate/train/evaluate/report 或全尺寸物理链路，
它们不是本 WP 的交付；配置可解析不等于相关功能完成。

## Review checklist and rollback
- [ ] A1–A7 每项有实际命令、断言或审阅证据，不使用源 §24 历史结果。
- [ ] config 默认逐项对照 §20；CLI 派生值对照 §§2–4、21.1。
- [ ] reference 方程/保护/梯度逐项对照 §§13–14，无标准 VAMP 替代 PG。
- [ ] 源文档未改，无外部 oracle runtime dependency，无后续 WP 业务代码。
- [ ] 数值阈值或容差的任何解释均可回溯执行书，不以通过测试为由放宽。
- [ ] full physical、生产逐层等价、训练/checkpoint、性能验证明确延期到对应 WP。
风险文件：config.py、两 reference、device.py。失败回滚只撤销当前任务
对应补丁；保留原仓库文件和真实失败记录，不使用递归清空工作区。
