# WP7 planning evidence — 2026-09-18

## Baseline and authority
起始 HEAD f455876、git clean、无 active task；WP0–WP6 已归档，WP6 工作提交 e73a983，历史 430 passed / 7 CUDA skipped，本轮未重跑。
执行书锚点：docs/CODEX_ENGINEERING_SPEC.md:1208 协议；1260 指标/计时；1360 CI；1376 失败；1386 产物报告；1816 CLI；1881 WP7；1946 测试。
FastCtx 工具不在可调用清单，使用 PowerShell/rg fallback；无外部搜索或实验。

## Inspected interfaces
- data/dataset.py:23 detection_inputs 仅 H/y/sigma2；:29 EffectiveDataset 保留 records 与完整块重放，只支持 oracle_timing。
- data/generate.py:33 测试 scenario/SNR 格与同物理帧副本；records.py:151 整帧同 SNR，:153 snr_copy 校验。
- algorithms/base.py:12 DetectionResult；vamp.py:22 detect 每次 SVD，mmse.py:17 每次 Cholesky，目前无 prepared-state。
- algorithms/pg_vamp/model.py:81 forward、:204 detect；WP6 spec 记录小型 summaries、T/T-1 机会分母，失败时需部分执行分母。
- inference.py:18 严格 physical checkpoint、兼容性、dtype 转换、ID/hash 与 contextual failures。
- base.yaml:139 evaluation 默认 2000 bootstrap、cold_H、5 warmup/20 repeats；config.py 校验共享输入及模式；cli.py 无 evaluate/report/benchmark。
以上代码路径相对 src/pgvamp_ofdm。pyproject.toml 已依赖 matplotlib；报告不需另加绘图库。

## Convergence and boundaries
源规格已决定范围，无阻塞性用户决策；风险处置见 design/implement。本轮仅写任务 artifacts 与父任务进度，没有改 src/tests/configs/执行书，没有训练、测试、评测或 benchmark。保持 planning、不设 active，WP8 未创建。
