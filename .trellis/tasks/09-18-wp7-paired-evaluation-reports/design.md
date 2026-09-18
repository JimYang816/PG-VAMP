# WP7 technical design

## Boundaries and flow
拟新增 evaluation/（runner、metrics、statistics、timing、artifacts）和 reporting/（plots/report），CLI 调度 evaluate/benchmark/report。流程：manifest/checkpoint 校验 → 固定 test 帧清单 → WP3 重放 → 三算法共享 H/y/sigma2 → 逐帧计数与诊断 → 聚合及配对 CI → 持久结果 → 只读报告。
复用 EffectiveDataset、detection_inputs、load_checkpoint/model_for/physical_contract 和 runtime。从 records 关联 scenario/esn0/snr_copy，不能从缺少 scenario 的 sample 字典猜测。检查每格帧数、8 块、单帧同 SNR、重复 ID、缺帧/空 split、配置与 manifest 预期一致；不临时生成替代数据、不选择有利子集。checkpoint 不可通过 test 选取。

## Counts, statistics and failures
帧键为 run/algorithm/scenario/esn0/frame_id，保留 sample IDs、input hashes、期望/尝试/成功/失败块数、整数错误计数、误差/目标能量。硬失败格保留计划分母；完整指标写 unavailable（CSV 空值并附状态），可单列 conditional_success 指标及其分母，不能编造失败预测或移除失败后称完整。失败 ledger 包含 ID、阶段、异常、dtype/device、必要范数；合法消息保持正常计入。失败格不产生完整结果 CI 或配对优势结论。
纯统计函数按帧重采索引再合计分子分母，同格算法共用索引。独立 bootstrap RNG 由运行 seed 派生并保存，不影响数据/训练流。跨 SNR 同物理帧不当独立试验，保留噪声策略和相关性说明。附加 paired_comparisons.csv 保存差值区间；零错误与零 FER 上界按源规格，少帧警示。不满足两曲线覆盖和充分错误数时不插值 SNR 增益。

## Timing and prepared-state contract
cold_H 走完整正式检测入口；same_H 增加显式只读 prepared-state，普通 detect 保持兼容。VAMP 可缓存 U/S/Vh；MMSE factor 还依赖 sigma2；PG 仅缓存当前参数下观测无关的图/矩阵组成，不缓存随 gamma2 变化的全部因子。
缓存标识覆盖 H 内容/shape/dtype/device、相关 sigma2、模型参数版本/保护设置。H/参数变化必须失效；训练不用缓存旧反向图。使用同 H 多个独立真实噪声观测核对 cached/uncached，不同 T_m 的 H 不得假装相同。分别保存准备耗时、后续耗时、观测数及总摊销。
使用 eval()+inference_mode，设备输入已就绪；预热/重复按配置；CPU perf_counter、CUDA 显式同步。诊断/计数/IO 在计时外；重放及同步/FFT/导频消除共同成本另列，effective_fast 与真实波形前处理区分。B1 在线延迟与 B>1 摊销分列。
CPU 进程峰值 RSS 用平台接口并声明生命周期/基线；不能隔离时不归因给单算法。CUDA peak allocated 独列；矩阵内存估算注明推导与非实测。不得把参数数代替工作内存。

## Diagnostics and artifact schema
复用轻量 summaries/counts，按实际执行机会聚合；正常 T8 每样本无信息机会 T、消息机会 T-1，失败时保存部分执行与缺失。完整 H_g ICI 在数据侧计算，H 的 ICI 单列。只保存有限带 ID/hash 的真实矩阵/波形/同步示例，避免完整诊断矩阵长期驻留。
采用版本化 results bundle，覆盖 §18 全文件/列并附 input lineage、failures、paired CI、统计种子、初始/学习后门限、计时元数据和示例附件。记录实际 argv、环境、代码/spec/manifest/checkpoint hashes；临时写后原子替换，拒绝覆盖别的 run，中断保持 incomplete。
report 校验 schema/hash/计数/完整性，仅消费持久产物，不能导入训练/检测执行入口。生成 §18 图集及 Markdown；无证据写“未执行”，不造占位曲线。零值显示专用标记不回写 CSV。多 seed 入口拟为 report --results <run> [<run> ...] --output <directory>，单目录保持源 CLI 兼容；合并前校验输入集合/配置/计时口径，保留 checkpoint hash/train_seed，不把同一 baseline 重复视作独立 seed。

## Compatibility and risks
新参数优先留在 CLI/结果 schema，不无故改数据/checkpoint schema。确需 config 项时同步两份 base.yaml，补明确旧数据兼容，保留旧 hash。
最大风险是缓存数学回归：必须 cached/uncached 和 WP4/WP5 oracle 双重核验。帧聚合、失败分母、日志污染计时、无证据报告分别用手算、故障注入、计时边界探针和只读报告测试覆盖。回退以 WP7 增量为单位，不改执行书/reference、不撤销旧 WP、不删用户产物。主训练、主 sweep、CUDA 实测仅按实际执行记录。
