# WP6 technical design — planned, not implemented

## Architecture
CLI → config/runtime → manifest/EffectiveDataset → deterministic batches → production PGVAMPDetector → weighted loss/Adam → validation → JSONL/checkpoints。
推理：安全 CPU checkpoint load → compatibility validation → explicit runtime transfer → WP3 load_materialized(purpose=inference) → H/y/sigma2 → prediction artifact。
新增模块优先放 `src/pgvamp_ofdm/training/{losses,trainer,checkpoint}.py`；推理/smoke 编排单独模块，CLI 只负责解析/错误呈现。复用 WP0 runtime、WP3 数据/安全加载、WP4/5 正式检测器，不用 reference 训练。

## Differentiable outputs and logs
当前 model.py:81 forward 默认只给末层；:166 详细 layers 保留大矩阵。增加显式可选训练输出（例如 return_layer_outputs），仅保留各层可微 xhat1；普通 detect 及预测返回行为兼容。禁止 detach 损失状态、缓存旧参数计算图或新增可学习参数。
小型 detached summaries 补齐 alpha1/alpha2、d/ell 量级、precision ranges 和 counters/denominators，不为训练开启全量大矩阵诊断。拒绝/截断率按实际消息机会计，末层无外信息更新，不能以 T 冒充 T-1。loss 按 t=1..T 规范权重及 complex abs.square.mean 独立实现。
记录 clip 前 norm；检查有限后 clip/step；在完整成功更新边界持久化。日志不保留计算图；未运行验证时 NMSE=null 并记录最近验证 step，不伪造新测量。

## Data, sampling and validation
load_manifest 验证内容与 split；train/val 两个 EffectiveDataset，test 不参与模型选择。重放 no_grad，模型前向有 autograd，batch 显式 stack。数据与训练 dtype 差异只通过记录的转换处理，比较物理/映射合同而非要求整个 config hash 相同。
独立 CPU sampler generator，保存 epoch permutation、next cursor 和 generator state；max_steps 是累计成功更新上限。尾 batch/epoch、验证插入/恢复不能改变下一批样本。验证稳定顺序且受 validation_max_blocks 限制，以总误差能量/总目标能量聚合末层 NMSE，拒绝空 val。best 严格改善，持久化 best step。
early_stopping=false 为默认；启用时提供明确配置 patience/min_delta（同步两个 base.yaml 和严格验证），按验证事件计数并恢复状态；最后一步执行验证，使两步 smoke 产生 best。
非有限 loss/grad 或分解失败记录 IDs、batch/layer、dtype/device、稳定 H 范数与关键状态；用可序列化标记表达非有限值，不写 JSON NaN。不提交失败 step、不跳过样本；保留上一成功 last.pt 并标记 run failed。

## Checkpoint and strict resume
Versioned schema 完整覆盖 PRD R4/§15.3，加 resolved config、执行书 hash、sampler permutation/generator state、early-stop 状态、日志运行身份。Python/NumPy RNG 转为基础类型/张量；CUDA RNG 仅显式 CUDA 时访问。CPU seeding 避免间接调用 CUDA 的统一 seed helper，以 forbidden CUDA stubs 验证。
复用 weights_only=True、map_location=cpu 的安全策略；严格校验字段/types/shapes/finiteness、模型恰好 raw_gaps/raw_mu、optimizer state 与 tensor dtype，再加载和迁移设备，无 unsafe fallback。
不可变恢复合同：物理/接收/QPSK/子载波映射、PG 数学设置/depth/mask、train/val manifest 内容、optimizer 设置、采样规则/batch size。允许显式 device 和增加 max_steps 并记录；严格 resume 拒绝 dtype 转换。推理允许显式转换，不把转换后训练称作严格恢复。
同环境同设备对照验证确定性一致；跨设备只承诺合理容差及完整状态迁移，不保证逐位相同。CPU 加载 CUDA checkpoint 时保留原 CUDA RNG 为 provenance，不调用 CUDA API。
last.pt 每成功更新后同目录临时文件写完再原子替换；best.pt 仅验证改善时替换。新训练拒绝覆盖不相关 run；resume 校验日志/step 身份，崩溃留下的超前日志用显式 resume event 标记无效，不伪装连续成功历史。

## CLI and inference artifact
按 §21 接受 train --config/--manifest/--device/--dtype/--resume/--output；infer --checkpoint/--input/--device/--dtype/--output；smoke --config/--device，另提供可记录的 --seed。默认设备 CPU。
infer 复用 WP3 materialized schema、metadata/lineage/shape validation，匹配 checkpoint 物理/映射合同，不要求与训练 manifest 身份相同。标签可存在但不传入模型。输出 versioned tensors/basic types、IDs、soft/classes/bits/probabilities、input/checkpoint hashes、实际 dtype/device/转换记录。

## Smoke and minimal counts
smoke 显式编排自己的少量数据/fixture；train 缺数据仍报错。数学路径 N32/T2、固定种子非退化复 ICI、两次更新、有限 loss/grad、2T/checkpoint roundtrip，标明 algebra_fixture。
系统路径复用 smoke_system.yaml（2 train/1 val frames），不改完整物理维度/T8。生成真实 waveform，并复用 WP2/3 audit 核验非零不同路径时缩下独立参考 FFT 与 effective H。三算法共享同一 H/y/sigma2 及 hash。训练两次、保存/加载、无标签推理；保存 bit/symbol/block/frame 错误数及总数、NMSE 能量和。FER 按完整帧聚合，残缺帧不能当完整帧；最小计数函数使用人工错误样例核验，暂不扩展 sweep/report/统计框架。

## Compatibility, risks and rollback
现有 WP0–WP5 API/数学/独立 oracle 保持；新增可选输出先跑 WP5 回归。checkpoint 为新 schema，无旧 WP6 格式迁移，未知/损坏版本拒绝。
400 维 dense solve/autograd 昂贵，轻量输出只减少额外驻留，不宣称稀疏加速、不降维。CUDA 实测取决于硬件，缺失记录 skip；CPU 完整 smoke 仍是硬门槛。
发现需要改数学/数据合同则返回 planning 审核。撤回仅限本 WP 新模块/兼容扩展及测试，不删除现有用户数据，不修改执行书/oracle 来通过验收。
