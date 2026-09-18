# WP3 design

## Boundaries and data flow
执行书优先。复用 WP1 allocation/QPSK/frame 和 WP2 sample_paths/validate_cp_support/effective_matrix/receive_window/preprocess/complex_awgn；不修改物理公式，不以 H@X 作为独立波形 oracle。
`resolved config → split/frame identities → independent seeds → physical frame records → shards + manifest → EffectiveDataset → detector inputs / labels / materialization`。
新增 `src/pgvamp_ofdm/data/{records,manifest,generate,dataset,materialize,audit}.py`，随机派生放 `utils/random.py`。依次负责 schema/校验、元数据/checksum、帧生成、重建/缓存、预算/导出和独立波形审计。
CLI 增加 simulate、audit-data、materialize（generate 可作为 simulate 别名，覆盖 §23.3 用语），保留 inspect-config；扩展所需严格配置，两个 base.yaml 同步。

## Identities and randomness
生成身份只包含物理生成参数和分布；排除 runtime、backend、storage 与检测器设置，因此切换精度或后端不改变物理样本。完整 resolved config 仍单独保存和校验。
先用 split 流产生确定的全局帧身份/分配表，满足 train_frames、val_frames 和每测试格独立帧数；为每物理帧分配 channel_id，再生成路径。身份命名空间含生成配置摘要、主种子和版本，不能靠给同一实例换 split 前缀绕过泄漏检查。
默认测试按 scenario 建立物理帧集合，在 SNR 网格内复用；每格仍有配置要求数量的独立帧。train/val 独立，使用配置场景混合及连续均匀 SNR。manifest 声明配对策略和独立帧/副本/样本数。
SHA-256 输入为版本化规范 JSON 的 master seed、stream name、frame/channel identity、block、SNR copy、draw/retry index；固定字节顺序截取合法 Torch seed。六类流独立；noise 再分 real/imag；场景和 SNR 使用独立子命名空间。路径拒绝只推进 channel retry，不改变 bits/pilots/noise/arrival。
不依赖全局 RNG 消耗或 worker 顺序；入口记录/设置 Python/NumPy/Torch CPU/显式 CUDA seed，样本使用显式 generators。默认 num_workers=0。
sample_id 由 frame_id + block + SNR copy 定义；channel 实例按生成谱系判别，identity/static 的数值巧合不等于实例泄漏。

## Persistence and integrity
帧 schema v1 严格实现 §10.3 全部字段：schema_version、frame_id/channel_id/split/scenario、path_count、path_gain_complex/path_delay_s/path_epsilon、data_bits、pilot_symbols、esn0_db、noise_seed_real/noise_seed_imag、arrival_offset_samples、waveform_config_hash/generator_version、cp_valid/generation_rejections。
采用显式 pilot_symbols、double precision 路径、uint8 bits 和逐块种子；不序列化 PathParameters 对象，不默认保存 H。
manifest v1 保存 resolved config、waveform/frame/allocation/QPSK 合同、分布、生成/模型版本、源码/规格指纹、环境、流派生版本、split 表、分片相对路径/hash/数量及独立帧和展开数。配置摘要与环境摘要分开，dtype/model/mapping 明确参与重建校验。
用 torch.load(map_location='cpu', weights_only=True)，校验 checksum/schema/字段后显式移设备；分片路径限制于数据集目录。未知 schema、配置不匹配、损坏/重复记录拒绝，不静默修复。
分片先写临时文件再提交，最后发布 manifest；默认拒绝覆盖已有目标。manifest hash 由最终 JSON 字节计算并由外部引用，不递归保存自身摘要。

## Reconstruction and cache
根据 WP1 帧布局和 block 的 T_m 重建完整 H_grid、固定 data/pilot 映射的观测及精确导频消除，按 Es/N0 和独立双流噪声构造 y。默认 oracle_timing/perfect CSI，实际路径/窗口先通过 CP 检查；waveform_reference 用独立连续波形 + 实际 FFT，不悄悄回退 effective_fast。
物理计算遵守 WP2 双精度后显式 cast；H/y/x complex128 配 sigma2 float64，complex64 配 float32。no_grad；标签与仅 H/y/sigma2 的检测输入投影分开。
LRU 键含记录/配置摘要、frame/channel、block/T_m/窗口、model version、mapping、dtype/device。SNR 可共享矩阵但不能共享不同噪声 y；返回值不能让消费者原位修改污染缓存。

## Materialization and validation
计划 CLI：materialize --manifest ... --split ... --output ... --max-output-bytes ... [--allow-large-output]。默认预算 1 GiB 为保护参数，不改变通信/实验合同；同时保护 data.storage=materialized 路径。
张量载荷公式 K*(N*N*c + N*c + r + labelled*(N*c + 2*N))，bits uint8，c/r 按 dtype。分别报告矩阵/其他载荷和 metadata/序列化保守开销预留，不将 payload 误称为精确 .pt 文件大小。超预算在矩阵生成/创建输出前拒绝。
物化 schema 遵守 §10.4，metadata 含 IDs/split/映射/来源 manifest hash。入口显式区分 inference/train/evaluation；推理标签可省略，存在时仍校验；训练/误码评测要求完整且一致的 x/bits。
检查形状/方阵/dtype/finite/variance/QPSK/配置/索引/数量/split。无隐式 H 归一化，不新增缩放功能；未来显式缩放仍须同时调整 H/y/sigma2 并验证预测不变性。

## Audit, compatibility, risks and rollback
`audit-data` 显式运行；未传帧数时读取 data.audit_waveform_frames。simulate 将此策略写入 manifest，不隐式执行昂贵波形审计。验收必须实际执行两帧审计。
audit-data 从保存记录核对完整性/重放/CP，选取规定数量完整帧保存波形、FFT、H、导频贡献/消除结果及噪声证据；非零不等 epsilon complex128 误差 <1e-9。
保留 WP0–WP2 接口和 references；后续检测器使用数据合同但本任务不实现它们。新保护参数缺省有记录，双配置同步。
主要风险是完整波形审计耗时、物化资源量大、跨环境数值差异、错误缓存键和流串扰。以少量完整尺寸审计、预算预检、环境记录、缓存/跨进程/顺序测试控制。
回滚移除新增数据模块与 CLI 分支、恢复新增配置键；保留失败诊断，不覆盖或删除已有用户数据。半成品不可被识别为有效数据集。
