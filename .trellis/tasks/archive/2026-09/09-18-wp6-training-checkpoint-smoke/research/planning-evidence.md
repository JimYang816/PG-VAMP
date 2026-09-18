# WP6 planning evidence — 2026-09-18

## Authority and prior state
- docs/CODEX_ENGINEERING_SPEC.md:1130–1204 为训练/设备/checkpoint/log/profiles；1740–1814 CLI；1875 WP6；1946–1972 工程验收和容差。
- WP5 归档 research/check-report.md 是历史独立验收依据。创建前 HEAD=5246e9b、git clean；WP5 work commit=877fcca。本轮未重跑历史测试。
- 唯一授权为 WP6 planning。当前工具清单无 FastCtx 文件工具，因此使用 shell fallback。无需搜索会话历史，既有 PRD/规格/归档已提供依据。

## Inspected integration points
- src/pgvamp_ofdm/algorithms/pg_vamp/model.py:81 forward，:114 逐层状态，:125 xhat1，:166 完整 layers，:168 小型 summaries。需增加轻量可微输出，不用 detached summaries 算 loss。
- src/pgvamp_ofdm/algorithms/base.py:12 DetectionResult；:30 failure helper 的上下文仍需 caller 补充 IDs/范数。
- src/pgvamp_ofdm/data/dataset.py:21 标签隔离 projection；:29 EffectiveDataset；:61 no-grad replay。单样本需明确 batch stack，manifest dtype 与训练 override 需校验和记录。
- src/pgvamp_ofdm/data/materialize.py 已有 load_materialized(purpose=inference)、metadata/lineage/shape 校验与 labeled=False；复用现有 schema。
- src/pgvamp_ofdm/data/manifest.py:45 安全 weights_only CPU load；checkpoint 另需自己的 schema 校验。
- src/pgvamp_ofdm/utils/device.py:20 CPU 默认与显式 CUDA；RNG 操作不能绕过此约束。
- src/pgvamp_ofdm/cli.py:69 现有入口仅 inspect/simulate/audit/materialize；物化无标签参数是 --without-labels。
- configs/smoke_math.yaml 为 N32/T2/2 updates；smoke_system.yaml 仅减少帧数与更新。新增配置须同步 configs/base.yaml 和 src/pgvamp_ofdm/base.yaml。

## Convergence and limitations
无阻塞性范围决策；技术风险为逐层输出驻留、完整诊断分母、严格 resume 与日志原子边界，已在设计和验收覆盖。未运行训练、smoke、性能或功能测试，未修改 src/tests/configs。WP7/WP8 未创建。task 保持 planning 且不设 active。
