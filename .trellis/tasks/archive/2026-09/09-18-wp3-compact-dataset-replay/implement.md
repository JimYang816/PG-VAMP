# WP3 implementation plan — implemented and independently verified; work commit and completion workflow approved

## Gates
- [x] 用户授权 planning-only child；WP2 历史验收证据已读取。
- [x] PRD、设计、执行计划已准备。
- [x] 用户明确批准最新规划及 implementation。
- [x] 批准后主会话核对 git、执行书指纹及 WP2 接口，并 task.py start。

## Ordered work
1. [x] records/manifest/random：schema、稳定 ID/seed、完整元数据、配置/文件摘要、安全加载和错误消息；建立 A1/A2 fixtures。
2. [x] generate：帧/信道 split、训练分布、配对 SNR、独立流、CP 拒绝计数、分片发布和准确计数；完成 A2/A4。有界且记录的重试，耗尽明确失败。
3. [x] EffectiveDataset：按需 H/y/sigma2、标签隔离、no_grad、正确 LRU 键、访问顺序/缓存污染/跨进程重放；完成 A3/A6。
4. [x] materialize：payload/开销/预算、拒绝/显式放行、紧凑与物化等价、无标签和坏输入矩阵；完成 A1/A5/A6。
5. [x] CLI simulate/generate、audit-data、materialize 和严格配置扩展；保持 inspect-config、双 base.yaml；更新 README/VALIDATION。
6. [x] 完整尺寸独立物理审计，保存工件的独立读取/FFT/消除/哈希复核；完成 A7。未以 H@X 冒充波形证据。
7. [x] Trellis 独立 check 按 §22 WP3 和全部适用 §23 验收；记录实际命令、环境、结果和工件指纹；修复后重验。

## Original planned validation commands
以下为原始规划命令清单；实际已执行命令及结果见 research/implementation-evidence.md。
```powershell
python -m pytest -q tests/test_data_records.py tests/test_data_random.py tests/test_dataset.py tests/test_materialize.py tests/test_data_cli.py
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src
python -m pgvamp_ofdm inspect-config --config configs/cpu_dev.yaml
python -m pgvamp_ofdm simulate --config <small-physical-config.yaml> --output <fresh-data-dir>
python -m pgvamp_ofdm audit-data --manifest <fresh-data-dir>/manifest.json --waveform-frames 2 --output <fresh-audit-dir>
python -m pgvamp_ofdm materialize --manifest <fresh-data-dir>/manifest.json --split test --output <fresh-materialized.pt> --max-output-bytes 1073741824
```
测试文件和新 CLI 为计划接口，当前不存在。小物理配置只减少帧数，不缩小 512/400/8192/8；覆盖 static 和非零不等 epsilon，标明 development_only。
CPU complex128 强制，complex64 单列容差；CUDA 无硬件记 skip。main 数据规模、训练及性能 sweep 未执行。

## Acceptance coverage and rollback
- A1/R1/R6：独立核对记录/manifest/哈希/数量；覆盖错误 dtype/shape/NaN/Inf/非正方差/无效 QPSK/配置索引/缺失标签/未知 schema/篡改。
- A2/R2：frame/channel 跨 split 重复及跨 SNR 泄漏，流隔离和 channel retry 不串扰；不误报 identity 数值巧合。
- A3/R3：同帧不同块 H，缓存开关/逐出/原位修改，独立进程重放及 no_grad；保存精确 payload 哈希。
- A4/R4：配置、计数和 SNR 配对关系；场景频率用固定种子、合理样本量容差，不要求小样本恰好匹配权重。
- A5/R5：两种 dtype 手算 payload、8192 矩阵例；拒绝在生成输出前，显式放行用小文件测，不为测试写几十 GB。
- A6/R5–R7：三个消费者 ID/hash 相同，检测投影仅 H/y/sigma2；实际三算法联测和预测标签隔离由 WP4–WP7 完成。
- A7/R3/R7：非零时缩独立波形、CP 边界/拒绝、CLI 小闭环和回归。
身份/重放/容量合同失败则停止推进；物理审计失败不放宽阈值。回滚新数据模块/CLI/config 修改，保留诊断、不删除既有数据。

## Planning verification
原 planning 轮仅校验规划文件、JSON/JSONL 路径、父子链接及 planning 状态。批准后的 implementation 已执行产品测试和小规模完整物理尺寸闭环；独立 check 尚待主会话分派。

## Final verification
最终独立检查通过：248 passed / 3 CUDA skipped，Ruff/format/mypy 通过；两帧 16 窗口独立审计最大误差 1.4435457958165196e-11，32 个物化样本与紧凑重放逐位一致。详见 research/check-report.md 和 check-final-command-receipts.json。用户已确认工作提交、归档和 journal；WP4 未启动。
