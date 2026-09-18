# WP8 Design

## Document boundaries
README 作为交付导航和最短 CPU 路径；拟新增 docs/CONFIGURATION.md、docs/ALGORITHMS.md、docs/REPRODUCING.md 分别解释配置、公式与模块映射、分级命令及产物。IMPLEMENTATION_STATUS.md 维护功能/性能双状态；VALIDATION.md 增加 WP8 真实证据并保留带日期历史。修复过时当前状态和归档链接，不改写历史结果或执行书。

配置文档核对 base/profile/Config，解释 overlay、随机种子、预算、新输出目录、严格 resume；数学文档引用源规格，映射正式模块及独立 oracle，保留稠密复杂度和 2T 参数约束，不形成第二套规格。

## demo-frame contract and data flow
拟新增 `demo-frame --config <physical yaml> --device cpu --dtype complex128 --output <fresh directory>`。沿用配置种子，仅生成一个完整帧，不执行 profile 全部数据规模；拒绝 algebra profile，dtype/device 沿用 runtime 校验，不静默回退。

package 内新增轻量编排模块（拟 demo.py），调用既有 frame/channel/receiver/synchronization API；借鉴 WP1/WP2 audit 产物组织，不 import scripts 作为安装后依赖。选择 CP 有效、不同非零 epsilon 的可复现物理路径。独立连续波形→接收 FFT→导频消除，与解析 H 比较，不能用 H 生成自己的验证信号。

保存配置/环境/物理路径/随机流、实通带和解析发送波形、接收数组、LFM 相关和模板起点、完整块的 H/y/sigma2/参考符号、摘要与 hash；图像源于保存数组。标签仅用于审计，不进入检测器。同步峰值不冒充最早路径；同步结果与 oracle_timing 接收一致性分别标记，不声称 lfm_detect 数据重放或主评测已支持。

demo 不训练、不强制 PG checkpoint；三算法前向和训练计数由 system smoke 验收。拒绝覆盖输出，失败不发布成功 receipt。新增 CLI 为兼容性增量，已有 API/脚本/产物格式保持。

## Evidence flow and reuse
新 runs/wp8-* 目录→命令 receipts/产物→独立 checker→VALIDATION/状态更新。记录实际 HEAD/dirty/source hashes、环境、argv、时间、退出码；大数组保留 ignored runs，紧凑可定位证据保存在任务 research。

复用 smoke.py 和 WP6/WP7 acceptance scripts，避免重复实现训练、统计或计时。checker 独立核对保存数组的尺寸、输入 hashes、计数、两次更新、checkpoint 往返及报告图像；历史验收只能作背景。

## Risks and rollback
Windows 子进程启动前设 MKL_THREADING_LAYER=TBB；pytest 使用独立 workspace 临时目录。只运行 bounded 验收，但不能缩小物理尺寸。按 §§19–23 核对额外缺口，重大新缺口返回 planning，不将缺失功能包装成实验未执行。文档、demo 编排和验证分组回退，仅撤本任务修改，保留旧证据和失败回执；不放宽阈值或修改双 oracle 掩盖问题。
