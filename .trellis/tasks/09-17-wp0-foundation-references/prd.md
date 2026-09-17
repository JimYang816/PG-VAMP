# WP0 - Project Foundation and References

## Goal
严格完成执行书 §22 WP0 的可验证项目基础：配置、设备选择、目录、
独立正确性 references 和最小 CLI，为后续工作包提供稳定起点。

## Authority and background
唯一工程实施规格：[CODEX_ENGINEERING_SPEC.md](../../../docs/CODEX_ENGINEERING_SPEC.md)。
WP0 范围由 §22 定义，工程/配置/CLI/测试依据 §§19、20、21、23；
reference 必须同时遵循 §§0.3、13–15 的完整数学与精度合同。
[父任务](../09-17-pgvamp-complete-engineering/prd.md)管理总项目与 WP 顺序，
本子任务才是未来 WP0 实现的执行目标。bootstrap specs 已获用户审核。
当前没有 src/、tests/、configs/ 或 pyproject.toml，也没有 Git 提交；
不能宣称已有函数、通过的测试或代码提交版本。

## Requirements
- R1 基础与溯源：建立 §19 的 Python/PyTorch src package、依赖声明、
  模块入口、测试布局和生成产物忽略规则。记录执行书 SHA-256、
  实际环境和代码 commit/hash；无提交时明确记录并以源文件清单哈希标识代码。
  不升级无关依赖、不删除无关文件、不迁移未授权旧代码。
- R2 配置：以 §20 为完整基线，profile 只覆盖必要字段，拒绝各层未知键、
  非法类型/范围/枚举及矛盾派生参数。物理配置与 smoke_math 代数配置明确区分；
  最终配置可完整保存。不能用缩小物理维度模拟真实尺寸 smoke。
- R3 运行时：CPU 默认（即便有 GPU）；CUDA 仅显式请求且不可用时报错。
  complex128/float64 默认，complex64/float32 显式，AMP 禁用；
  默认 batch=1、workers=0、CPU threads=min(4, available)。CPU-only 路径不调用 CUDA。
- R4 最小 CLI：支持 python -m pgvamp_ofdm 与 pgvamp-ofdm 的 inspect-config。
  输出 §21.1 所有字段及磁盘容量估计；不生成数据、不训练、不跑评测。
- R5 独立 references：本工程内新建 reference/dense_pg_vamp.py 与
  reference/dense_vamp_cholesky.py，可运行小规模稠密代数输入。
  PG reference 完整遵循 §14 层循环、参数化、保护分支和梯度依赖，
  VAMP reference 遵循 §13 精确 Cholesky 数学。不能只是导入壳或恒等输出，
  不能依赖正式算法模块、外部 oracle 或另一份执行书。
- R6 最小验证：以 pytest 调用两 reference，验证基础数学/边界与
  可微性；验证配置、默认设备、错误路径与 CLI。按 §23 原容差基准，
  不把此阶段测试称为完整物理验收或正式算法等价验收。
- R7 状态证据：README/IMPLEMENTATION_STATUS/VALIDATION 记录已实现范围、
  实际命令结果和后续未执行项。只报告本次实际运行结果。

## Observable acceptance
| ID | 验收结果 | 对应 |
| --- | --- | --- |
| A1 | 可安装 src package；两个 CLI 入口可用，references 可导入；工程不依赖外部 oracle/执行书或 CUDA 安装 | R1,R4,R5；§§19、21、22 |
| A2 | inspect-config 对 cpu_dev 输出 CPU、complex128、96 kHz、8192 FFT、512 网格、400/64/47/1、11.71875 Hz、正确频率端点/CP/帧长/比特数与容量估计 | R2,R4；§§2–4、20–22 |
| A3 | 未知键（含嵌套）、错误 dtype/枚举/派生关系均可读报错；smoke_math 明确为代数配置，真实 profile 保留物理规模 | R2；§§15.5、20、23.3 |
| A4 | 可用 GPU 不改变默认 CPU；请求不可用 CUDA 明确报错，无回退；CPU-only 执行无 CUDA 调用；dtype 配对正确 | R3；§§15.2、23.3 |
| A5 | 两 reference 在小规模 complex128 pytest 中真正执行，输出形状/有限性/零信道/对角极限正确；PG 参数数为 2T，非退化夹具保留反向梯度 | R5,R6；§§0.3、13–14、22、23.2 |
| A6 | 有限小矩阵的 reference 内部性质获得独立证据（例如全图与精确 VAMP、QPSK 四点枚举核对），无 inverse/CG/随机 trace/任意 detach | R5,R6；§§14、23 |
| A7 | 保存完整解析配置、执行书哈希、实际代码版本/无提交状态与源码哈希、环境、实际测试结果；无虚构测试数或性能声明 | R1,R2,R7；§§20、22、24 |

A2 的关键期望：网格端点 21000–26988.28125 Hz（设计频带上界仍 27000）；
CP 2048 样本/21.333333 ms；帧 91776 样本/0.956 s/6400 数据比特。
以上来自执行书，不是本任务实测。验收细化没有删除 §23 后续测试义务。

## Out of scope and handoff
不实现正式 modulation/waveform/channel/receiver/data/algorithms/training/evaluation
模块；不提前构建生产 MMSE/VAMP/PG-VAMP、不做数据集、checkpoint trainer、
完整 smoke、BER 曲线或基准。WP0 reference 内的 QPSK 解析计算只服务数学
oracle，不视为 WP1 调制链路完成。
完整资源映射/波形 WP1、物理 CP/时变 H/噪声 WP2、数据 WP3、
正式 oracle 等价 WP4/WP5、训练恢复与两类 smoke WP6、评测 WP7、交付 WP8。

## Planning gate
阻塞产品决策：无；以上范围由执行书和用户本轮限定。
用户于 2026-09-18 明确批准 WP0 规划与实现；本 child 已通过 task start
进入 in_progress。执行仅限 WP0；完成后先汇报 diff、实际测试和未完成项，
不得开始 WP1。初次规划时的仓库状态仅为历史背景；本次实际起始 HEAD
为 2611522，工作区干净，详见根目录 VALIDATION.md。
