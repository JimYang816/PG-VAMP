# Source reading and WP0 evidence

## Authority and reading contract
唯一实施规格 docs/CODEX_ENGINEERING_SPEC.md，
SHA-256 A159D20380D4F48785FAC15A02B20247D681B4E078A9DD7F8046FDD1F66AC47B。
本轮起始 HEAD 0ca6e48447808cfa80c7468a05a69bdc82a9f0cb，工作区起始 clean。
FastCtx 文件工具未暴露，使用本地 PowerShell/rg 读取；没有外部工程规格依赖。

已读取 WP1 直接章节及邻接边界：
| 章节 / 源行 | 用途 |
| --- | --- |
| §§0–1 / 17–93 | 唯一权威、范围、真实证据、整体数据流 |
| §2 / 95–162 | 512/8192/400、默认物理参数 |
| §3 / 164–217 | 精确 allocation/索引/guard、PSD 与旁瓣 |
| §4 / 219–292 | 标签/编码/帧/LFM |
| §5 / 294–353 | ortho、幅度、整帧载波、理想 I/Q 与实前端边界 |
| §6 / 355–506 | 仅 WP2 边界：连续时间/仿射模型、窗口与 CP、独立 H 等价 |
| §7 / 508–545 | 基础相关、峰值含义、模式/指标边界 |
| §9 / 610–664 | 复/实噪声功率口径；只用于同步噪声 fixture 和功率说明 |
| §10.3、10.5、11 / 690–725、753–797 | bits/pilots 形状、输入检查、共享标签；数据集与检测器延期 |
| §14.6 / 1052–1093 | reference 后验与 WP1 映射区别；不迁移算法算子 |
| §15.2–15.3 / 1146–1185 | CPU/dtype、复现与证据 |
| §18 / 1386–1443 | 真实图表证据；不提前实现完整评测报告 |
| §§19–20 / 1446–1738 | 模块布局、依赖、配置完整默认值 |
| §21 / 1740–1835 | 保持 inspect-config；后续 CLI 不在本 WP |
| §§22–24 / 1837–1992 | WP1 精确交付及测试归属；§24 不是当前工程测试 |

之后 implement/check 必须直接读取上述适用原文，尤其 §§2–5、7、22/WP1、23.1。
JSONL 注入长度可能裁剪长文件；摘要不替代原文。先看工具截断标记，
长文件按连续行段补全，不以“已注入”猜测全文已读。外部引用不是执行依赖。
现有 Trellis specs 全部已读：backend/index、directory-structure、runtime-config、
reproducibility、numerical-contracts、quality-guidelines、wp0-inspect-contract；
guides/index、workflow-and-authority、cross-layer-thinking-guide、code-reuse-thinking-guide。
它们低于执行书权威，不改变 WP 顺序。

## Actual WP0 implementation
| 源码锚点 | 现有行为 / WP1 决策 |
| --- | --- |
| src/pgvamp_ofdm/config.py:185、224、269 | strict merge、物理配置校验、静态派生值；没有真实 allocation/waveform；复用但不能把静态计数当波形验收 |
| src/pgvamp_ofdm/utils/device.py:20 | 显式 CPU/CUDA、配对 dtype、线程与确定性；CPU 不探测 CUDA，复用 |
| src/pgvamp_ofdm/cli.py:20、69 | provenance 与唯一 inspect-config；无模拟/训练命令，保持现有合同 |
| src/pgvamp_ofdm/utils/validation.py:6 | 仅参考 H/y/sigma2 的输入检查，不强行用于 waveform |
| src/pgvamp_ofdm/reference/result.py:20、25 | argmax 最小类、R/I bits 顺序；可交叉核对标签，不挪用数值 oracle |
| src/pgvamp_ofdm/reference/dense_pg_vamp.py:17、34、75 | 独立 posterior/protection/完整 PG；与 WP1 不共享被验证算子 |
| src/pgvamp_ofdm/reference/dense_vamp_cholesky.py:17 | 独立 Cholesky VAMP，保留 |
| tests/test_config.py:10、21、109、126 | 两 base 副本一致、默认帧派生值、非法配置、alpha=0 已支持 |
| tests/test_devices.py:12、44、62 | CPU CUDA-call guards、不可用错误、硬件可用条件测试 |
| tests/test_cli.py:13、31 | 双入口、保存配置/来源与非法输入 |
| tests/test_reference.py:76、178、205、235 | 独立枚举、倒数边界、cap backward、无信息零梯度；回归保留 |
| pyproject.toml、tests/conftest.py | pytest/src 布局、Ruff/mypy、4 CPU 线程；已有可复用环境 |

两 reference 与全部现有测试已实际读取；尚无 modulation/waveform/receiver 业务模块。
WP0 历史设计不替代实际代码。后续 WP1 不需修算法，不扩展 denoiser/messages。

## Historical validation, not rerun
归档：
.trellis/tasks/archive/2026-09/09-17-wp0-foundation-references/
- task.json: completed，completedAt=2026-09-18，
  commit=ecbda1345744019a4490a91e061fa59a16420ffb。
- research/final-validation.json: timestamp=2026-09-17T16:33:04.134835+00:00；
  targeted 75 passed/1 skipped（12.22s）、full 75 passed/1 skipped（11.93s），
  exit 0；Ruff/format/mypy/inspect-config/diff-check exit 0。
- research/final-source-manifest.json: WP0 验证时 24 个产品/测试/配置文件指纹。
- VALIDATION.md 记录环境、历史失败和保护修复；其中旧“未提交/归档”段落
  被 archived task.json 和当前 Git 历史取代，本轮不改历史回执。
- 当前 Git：ecbda13=WP0完成，4b90d5f=WP0归档，0ca6e48=journal。

运行条件：.venv/Scripts/python.exe，Python 3.13.9、torch 2.12.0+cpu、
MKL_THREADING_LAYER=TBB。初始 OpenMP 冲突及中间失败已保留在 VALIDATION.md；
不使用 KMP_DUPLICATE_LIB_OK。Ruff 历史由 Anaconda python 启动。
唯一 skip 是 CUDA 硬件不可用；CPU 默认/不可用错误分支已实际测试。
WP1 本轮不运行 pytest，不宣称新的物理/算法/性能验证结果。
