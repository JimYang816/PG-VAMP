# WP0 technical design

## Authority and boundaries
以 [执行书](../../../docs/CODEX_ENGINEERING_SPEC.md) §22 WP0 为范围，
§§19–21、23 为工程验收依据，§§13–15 为 references 的算法依据。
本设计不复写或替换公式。下列路径是获批的 WP0 实现责任边界；
2026-09-18 已获实现授权，实际完成与验证状态见根目录 VALIDATION.md。
实现与检查都先按 research/source-reading-contract.md 完整读取原文件；
Trellis 单文件注入会截断执行书，不能视为已完成必读要求。

## Architecture / intended file ownership
| 路径 | WP0 职责 |
| --- | --- |
| pyproject.toml | src layout、核心依赖和 pytest extra、pgvamp-ofdm 入口 |
| src/pgvamp_ofdm/{__init__,__main__,cli}.py | 包加载、参数解析、inspect-config；没有训练/仿真副作用 |
| src/pgvamp_ofdm/config.py | typed schema、base/profile 合并、递归未知键检查、物理/代数校验、派生摘要与保存 |
| src/pgvamp_ofdm/utils/device.py | 显式 device/dtype 解析与 CPU 线程设置 |
| src/pgvamp_ofdm/utils/validation.py | 本阶段实际需要的输入校验，不预实现后续数据层 |
| src/pgvamp_ofdm/reference/dense_pg_vamp.py | 可读稠密 PG-VAMP oracle，包括完整层级输出和参数梯度路径 |
| src/pgvamp_ofdm/reference/dense_vamp_cholesky.py | 独立精确 Cholesky VAMP oracle |
| configs/{base,cpu_dev,main,smoke_math,smoke_system,cuda_example}.yaml | §20 基线与 §15.5 profile；存在配置不等于对应功能已实现 |
| tests/test_config.py, test_devices.py, test_cli.py, test_reference.py | WP0 基础与独立 reference 验收 |
| README.md, IMPLEMENTATION_STATUS.md, VALIDATION.md, .gitignore | 安装/已实现命令、WP 状态、真实验证与产物管理 |

§19 其余目录保留为后续地图，不写占位业务代码。CPU 安装不强制拉取 CUDA；
优先可用已装 PyTorch，不无故升级。实现前记录 Python/PyTorch API 能力；
不提前声称某版本已经支持所有操作。

## Configuration flow
CLI arguments → base.yaml → selected profile overrides → explicit CLI overrides
→ strict typed validation → derived physical/algebra summary → runtime selection
→ printed summary / optional resolved-config artifact.

采用 Python dataclasses/显式字段 schema 和 PyYAML safe_load，无需新验证框架。
所有层级字段按 §20 注册，递归合并映射、序列整体替换；未知键在合并前按
对应 schema 检查，不能在覆盖时消失。该解析方式是工程选择，不改变源默认值。
入口 --config 选 profile；--device/--dtype 只在用户传入时覆盖配置。
计划用 --output 可选目录保存 resolved_config.yaml 和 environment/provenance.json；
这是 inspect-config 的工程接口细化，不修改执行书已有命令。
所有运行目录保存最终配置；无 --output 的 inspect-config 可以只读打印。

物理 profile 验证 §20 字段、频率整栅格关系、资源计数/配置合法性、
门限范围和 dtype/device。不调用随机路径采样或波形代码。
计算静态派生量仅服务检查器，不声称完成 WP1 allocation 算法。
CP 配置值在 WP0 检查合法性，逐路径逐块支撑检查由 WP2 在真实路径可用时
完成；本阶段不能打印“所有物理帧 CP 已验证”。使用实际路径的功能尚不存在。
smoke_math 用显式 profile 识别与严格的代数子 schema（N=32,T=2，
2 次更新的配置），避免以文件名或小维度猜测运行模式；不把 fixture 当通信配置。
实现具体 schema 表达时不得新增可静默绕过物理校验的开关。
配置默认值、main 覆盖、scenario 退化约定逐项与 §20 核对。

inspect-config 明确区分设计频带端点与实际网格端点，打印 §21.1 全部值。
容量估计区分 compact records 和显式 dense materialization；用形状、dtype
字节数、块数计算 dense 估计，说明 compact 的可变元数据/序列化开销，不能
把估计当作实际落盘字节数。保存产物不生成数据矩阵。

## Device and precision
默认 CPU/complex128/float64。只有显式 CUDA 请求才探测 CUDA 可用性；
不可用报错。CPU 初始化和 provenance 收集不能顺手访问 CUDA。
线程数在进程启动设置并记录。reference 常量和单位阵跟随输入 device/dtype；
raw_gaps/raw_mu 是配对 real dtype，不能靠全局默认 dtype 掩盖错误。
complex64 是显式路径，误差与有限性检查另列；CUDA 数值等价测试仅在硬件可用
时实际运行，不可用记录 skip，不用 mock 冒充真实 CUDA 计算。

## Independent reference contracts
- 独立小规模 PyTorch 稠密实现，输入 H/y/sigma2 采用 §11 的批量维度
  [B,N,N]/[B,N]/[B]，校验方阵、shape、dtype/device、finite、sigma2>0。
  测试至少 B=1 并覆盖一个多样本 batch；不扩展矩形检测合同。
- PG reference 包含 §14.1–14.7 的全部计算和 §14.8 自动微分要求：
  只有 raw_gaps/raw_mu，实际 centered operator、精确 trace、Frobenius 方差、
  QPSK posterior、保护分支与最终 posterior 输出。不会以标准 VAMP 代替它。
- Cholesky VAMP 使用 §13 精确系统，trace 可通过同一 Cholesky solve 作用于
  identity 求得，禁止 explicit inverse。没有可训练参数。
- 返回可测试的最终均值/概率以及可选逐层状态，供未来正式实现对比；
  不为 WP0 建立生产 Detector 实现或基线 runner。
- reference 内解析 QPSK 及保护计算保留独立可审阅路径；通过四点枚举等
  独立预期核对，不导入未来 production denoiser 或修改 reference 迎合生产输出。
- exact zero/no-information 显式分支；其他阈值严格按源规定。
  §14.5 未给固定常量的数值无信息判据，实作必须依据 dtype/矩阵量级给出
  推导与边界测试记录，不凭经验放大。若无法在源合同内论证，停止该项并提出
  具体问题，不能自行改写算法或以 jitter/epsilon 静默修复。
- jitter 默认 0，若启用仅按源规则配置、记录并统一实际矩阵；
  本阶段不以新增自适应 jitter 兜底分解失败。

## Verification design
WP0 测试是初步 oracle 正确性证据，并非 WP5 完整验收：
- 配置默认/嵌套未知键/非法类型与枚举/矛盾派生关系、profile 合并、保存 roundtrip。
- inspect-config 的两入口与真实期望值核对，错误返回非零退出码。
- CPU 默认不探测 CUDA；通过隔离 mock 测请求不可用分支，
  此 mock 不代表硬件等价测试。检查所有 reference 中间状态 dtype/device。
- 小 N identity、diagonal、zero、rank-deficient、固定非退化复杂矩阵；
  代数随机矩阵带固定 seed 和 fixture 标签，不作物理信道。
- 两 oracle 真实前向；全图 PG 与精确 Cholesky VAMP、QPSK 四点枚举；
  PG 参数计数、有限梯度及小规模 raw 参数 gradcheck。
  强制全图只作验证夹具，不提供 silent hard-mask 运行模式。
- 保护边界（非法候选/无信息/下溢）不制造 NaN 后隐藏；
  §23 建议容差直接沿用，条件数调整必须有记录。
- source/runtime 依赖审计，禁止 inverse、CG、随机 trace、任意 detach；
  不把文本搜索单独当成数值正确性证明。

逐层生产等价、完整 Jacobian/PSD/Loewner 等 WP5 全套断言继续保留给 WP5；
波形/噪声/CP 物理测试留 WP1/WP2，checkpoint/训练/CLI 闭环留 WP6/WP7。

## Provenance, compatibility and rollback
实现开始及验证时记录源文档 SHA-256、Git HEAD/dirty 状态、软件环境。
规划时无 commit 的描述属于历史状态；本次执行起始已有初始提交 2611522。
按实际 Git 状态记录 commit、dirty 和排序代码文件内容哈希，不能编造版本。
未来若在无提交工作区执行，记录 commit=null/uncommitted，同时保留源码哈希。
用户执行书保持不变。仅新增本任务拥有的工程文件；回滚仅针对本任务补丁，
不清理未跟踪目录或整个工作区，不删除用户文件。
主要风险：reference 与生产代码过度复用、配置默认漂移、smoke 范围膨胀；
通过独立实现、source 对照、明确 WP handoff 管控。
