# WP6 — Training, checkpoint, resume, inference and smoke

## Goal and authority
交付可复现的 PG-VAMP 训练、保存/恢复、无标签推理闭环，用数学与完整物理尺寸 smoke 验证工程连通性。
唯一实施规格为 `docs/CODEX_ENGINEERING_SPEC.md` §§10–11、14–15、20–23；本任务不能修改数学合同或降低物理尺寸。
用户于 2026-09-18 授权创建 WP6 child task，**仅 planning，不开始 implementation**。
父任务为 `09-17-pgvamp-complete-engineering`；用户随后明确批准最新规划并授权 implementation；本任务可进入执行。

## Background and prerequisite
WP5 已完成、独立验收并归档，工作提交为 877fcca；历史证据为 376 passed / 6 CUDA skipped，见归档 WP5 research/check-report.md。这不是本任务的新执行结果。
已有正式 PGVAMPDetector、MMSE/VAMP、紧凑数据重放、安全物化输入和 CPU 默认运行时。当前 CLI 尚无 train/infer/smoke，逐层可微状态仅通过详细诊断返回。

## Requirements
- R1 仅训练 PG-VAMP 的 2T 实标量。逐层复 MSE 权重 2t/[T(T+1)]；Adam、lr=1e-3、weight_decay=0、梯度范数上限 5。仅以 val 末层 NMSE 选 best，支持配置的 batch、验证频率/上限及 early stopping，禁止 test 调参。来源 §§14.1、15.1、15.5、20。
- R2 复用显式 manifest、物理帧 split 和标签校验；缺数据不得自动生成，训练/验证缺标签报错，检测器只接收 H/y/sigma2。非有限 loss/输出/梯度或分解失败记录 IDs、dtype、矩阵范数和关键状态后停止并标记失败，不删除样本伪装成功。来源 §§10.5、11、15.4。
- R3 默认 CPU/complex128/float64、batch=1、workers=0、线程 min(4,cpu_count)。CUDA 必须显式且不可用报错，CPU 路径不调用 CUDA。complex64 是显式模式，设备/dtype 切换记录并测试，不使用 AMP。来源 §§15.2、21.5。
- R4 checkpoint 完整保存 §15.3 schema、模型/优化器、算法/波形/映射、step/epoch/sampler、种子及 Python/NumPy/Torch RNG、best metric、train/val manifest hashes、dtype/device/mask、QPSK 映射、软件与代码指纹；仅安全张量/基础类型，先加载至 CPU。
- R5 严格恢复 optimizer、采样和 RNG 后继续更新；不静默覆盖深度、mask、波形、映射、数据身份和精度。允许设备等显式变更并记录；同环境连续与中断恢复结果相符。来源 §§15.3、21.4。
- R6 JSONL 日志与 checkpoint 分离，完整记录 §15.4 loss、验证 NMSE、逐层 rho/mu、有效边、d/ell 量级、c/alpha1/alpha2、精度范围、拒绝/无信息/截断率、梯度范数和耗时，以及配置/环境。未验证步骤标注无新验证值。
- R7 infer 安全读取 checkpoint 和 WP3 物化 H/y/sigma2，不要求 x/bits；eval/inference_mode，无训练。默认继承 checkpoint dtype，显式转换可追溯；输出保留 IDs、预测及来源 hashes。来源 §§10.5、21.5。
- R8 smoke_math 为 N32/T2/两次更新；smoke_system 保持 512 网格/400 数据/8192 FFT/CP2048/8 OFDM 块/T8，包含真实波形、有效系统、三算法同输入前向、两次 PG 更新、checkpoint roundtrip 和真实指标计数。只减少帧数，不作性能结论。来源 §§15.5、21.2、22/WP6。
- R9 实现 §21 train、train --resume、infer、smoke 命令，支持种子覆盖并记录。数学 fixture 不进入物理性能结果。来源 §§16.2、21。

## Acceptance criteria
| ID | 可观察验收 | 要求 |
| --- | --- | --- |
| A1 | 独立手算逐层加权复 MSE；有限梯度、仅 2T 可学习标量；非退化 ICI 两次更新改变参数，基线无训练状态 | R1 |
| A2 | manifest/split/标签异常拒绝；loss/梯度/Cholesky 故障有上下文且停止；JSONL 必需诊断和计数分母齐全 | R2,R6 |
| A3 | 保存/加载预测一致；安全加载拒绝畸形/不兼容状态；连续 k 步与 j 步后恢复至 k 步的模型/optimizer/样本序列/RNG 相符 | R4,R5 |
| A4 | CPU 默认及禁止 CUDA API 的 CPU train/resume/infer/smoke 测试；CUDA 不可用报错；有硬件则跨设备验证，无则明确 skip；complex64 输出/梯度有限且容差有依据 | R3 |
| A5 | 无标签 infer 成功；改/删标签不影响预测；dtype 继承/显式转换和不兼容错误覆盖 | R7 |
| A6 | 实际完成数学与完整尺寸 CPU smoke，保存命令/环境/维度/两次更新/checkpoint/hash/计数；非零时缩波形与有效模型一致性达到 §6.6 | R8 |
| A7 | simulate/audit → train → resume → 无标签 materialize → infer 小闭环；现有 WP0–WP5 回归及 lint/type 检查通过 | R9 |

## Out of scope
WP7 evaluate/report、统一 sweep、bootstrap/配对统计/正式性能计时；WP8 完整交付；main/cpu_dev 全规模训练、收敛或 BER 优势声明；MPS/多卡/分布式；hard mask/数学公式变更。smoke 最小计数与验证 NMSE 不代表 WP7 完成。

## Planning readiness
需求已收敛，无阻塞性用户决策。design.md 保存设计，implement.md 保存实施顺序，research/planning-evidence.md 保存依据。最新规划已于 2026-09-18 获用户明确批准，可进入 implementation。
