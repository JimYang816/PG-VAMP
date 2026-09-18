# WP3 — Compact Dataset, Splits and Replay

## Goal and authority
为后续检测器、训练和配对评测提供可验证、可重放且无 split 泄漏的数据层，避免默认保存全部稠密矩阵。
唯一工程规格是 `docs/CODEX_ENGINEERING_SPEC.md` §§10–11、16.2、19–23；本文不改变数学、物理参数或验收要求。
父任务：`09-17-pgvamp-complete-engineering`。前置 WP2 已完成归档；历史全量 194 passed / 3 CUDA skipped，本次未重跑。
用户于 2026-09-18 明确授权创建 WP3 child task 并仅做 planning。该 planning-only 轮次已结束；用户随后明确批准最新规划并授权 implementation。WP4 仍未授权。

## Confirmed background
WP0 提供严格配置、runtime、inspect-config；WP1/WP2 提供帧、物理路径、完整 H、独立波形、CP 检查、FFT、导频消除与噪声接口。
当前没有 data 包，CLI 只有 inspect-config；simulate/audit-data/materialize 均是本任务未来交付。
默认物理样本为 400×400 的 H、一帧 8 个块，保留 512 网格和 8192 FFT。CPU/complex128 为正确性基准，complex64 单列验收。

## Requirements
- R1 紧凑存储：版本化 manifest + train/val/test `.pt` 分片，仅张量/基础类型；保留 §10.3 全部帧字段及完整通信参数、频点映射、分布、拆分规则、种子、环境、checksum、计数和 SNR 定义。
- R2 随机与 split：至少独立 split/channel/bits/pilots/noise/arrival_offset 流，稳定 SHA-256 派生，不用 Python hash()。先划分物理帧/信道实例再扩展符号/SNR，不允许跨 train/val/test 复用实例。
- R3 重放：同记录/manifest/环境确定性还原 sample_id、H/y/sigma2 和标签；访问顺序、缓存命中、消费者不得改变样本。生成 no_grad，不暗中归一化 H。保存实际拒绝次数及 CP 有效性。
- R4 分布：遵守配置与 §16.2 的训练混合、连续 SNR、评测场景和网格；测试一帧全部块同 SNR，跨 SNR 配对复用同物理帧时使用不同噪声子种子并记录策略。准确区分独立帧数与展开样本数，cpu_dev 标注 development_only。
- R5 物化：支持 §10.4 H/y/sigma2/x/bits/metadata 格式；先估容量，超过可配置阈值须显式 --allow-large-output。推理允许缺少标签，训练/误码评测缺少标签报错。
- R6 校验：形状、方阵、dtype 配对、finite、正 sigma2、QPSK 能量/类别、数量、索引、配置哈希、文件 checksum、split 重叠均检查；加载先到 CPU，安全加载，不反序列化自定义对象。
- R7 数据交付：实现 simulate、audit-data 及显式物化入口；审计少量完整波形并与有效模型核对，保存可独立复核的证据。不同算法获得相同 sample IDs/输入哈希，普通检测输入不包含标签。

## Acceptance criteria
| ID | Observable outcome | Source / requirements |
| --- | --- | --- |
| A1 | 紧凑文件安全 roundtrip；完整字段、索引、配置/文件哈希和数量可独立核对；篡改/未知 schema/非法数据清晰失败 | §§10.3,10.5; R1,R6 |
| A2 | frame/channel 集合在三 split 两两不交；符号及 SNR 副本继承唯一 split；独立流无串扰 | §§10.2,23.1; R2 |
| A3 | 固定环境跨进程重放一致；缓存开关/逐出后 H/y/sigma2/ID 哈希一致，no_grad；同帧不同时刻不错误复用 H | §§10.1–10.3,22 WP3; R3 |
| A4 | 训练/测试分布与 resolved config 相符；测试全帧 SNR 一致；跨 SNR 配对且噪声不同；计数区分独立帧与副本 | §16.2; R4 |
| A5 | 紧凑和物化读取逐样本一致；8192 个 complex128 400×400 H 的矩阵载荷为 20,971,520,000 bytes，其余字段另计；阈值拒绝在大矩阵生成/输出前发生 | §10.4; R5 |
| A6 | 无标签推理可加载，训练/评测明确拒绝；三个独立消费者取得相同 ID/H/y/sigma2/标签哈希；标签不进入检测输入投影 | §§10.5,11,23.3; R5–R7 |
| A7 | 默认完整尺寸非零不等 epsilon 波形/有效模型审计误差 <1e-9 (complex128)，CP 不合格不进入数据集；CLI 小闭环及 WP0–WP2 回归通过 | §§6.6,21.3,23; R3,R7 |

## Out of scope and deferred integration
不实现 WP4/WP5 正式检测器、训练/checkpoint/恢复、三算法性能比较、统计报告或完整系统 smoke；不运行 main 数据规模/训练/SNR sweep 来宣称性能。
WP3 以独立消费者检验共享样本合同；实际三算法输入一致性及修改 target 不影响检测输出在 WP4–WP7 集成验证，不免除 §23.3。
不扩展 CSI 估计、同步算法、信道数学或数据归一化功能。跨设备/版本不承诺 bitwise 相同，遵守记录环境内重放。

## Planning disposition
无阻塞性产品问题。设计选择、容量预算与失败策略见 design.md；尚未执行的验证见 implement.md。
最新规划已获用户明确批准，任务于 2026-09-18 进入 in_progress。
