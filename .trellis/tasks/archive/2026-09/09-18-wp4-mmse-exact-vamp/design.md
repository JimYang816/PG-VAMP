# WP4 technical design

## Boundaries and data flow
WP3 EffectiveDataset → detection_inputs → 显式 stack/unsqueeze 成 batch → MMSEDetector / VAMPDetector → DetectionResult。
ID/hash 和标签留在测试/后续评测调用方。继承输入 validation 和 QPSK 映射，不改变数据 schema 或物理链。

拟新增 `src/pgvamp_ofdm/algorithms/{__init__,base,mmse,vamp,qpsk,messages}.py`：
- base：DetectionResult dataclass 和 Detector Protocol，遵循 §11。
- mmse：无训练参数 MMSEDetector.detect，完整 H Cholesky solve，复用 hard_decision/classes_to_bits。
- vamp：无训练参数 VAMPDetector，iterations 默认 8 且必须为正整数；SVD 每次 detect 开始执行一次，不跨调用缓存。
- qpsk/messages：正式算法共享解析后验和保护算子，未来 WP5 可复用；本 WP 不创建 PG 模型。
- `configs/vamp_reference_32.yaml`：仅覆盖 vamp.iterations=32 的补充 profile，保持主默认 8 层，后续报告必须单独命名。

## Numerical contracts
MMSE 按 §12：完整 H Gram + sigma2 I，用 Cholesky 求解；不求逆、不附加 QPSK 后验、不修改 H。只允许理论 Hermitian Gram 的舍入级对称化。
VAMP 按 §13.3：reduced SVD、lambda=gamma_w*s²、U^H y、V^H r2；中心化 delta 更新，alpha2/c 独立求和，避免相消；不得替换为近似迭代求解。
QPSK 用 tanh/稳定 sech²；概率可用等价 log-sigmoid 计算，但必须与独立四点枚举核对，class 顺序服从 §4.1。
消息按 §14.6 的分母 1e-6、精度下限 1e-10、上限 1e8；先验证再 reciprocal/divide，避免 where 掩盖 NaN 或梯度污染。拒绝保留旧消息，截断只改变精度，最终层不生成不用的外信息。

数值无信息判断采用与算子尺度和 dtype 相关的舍入误差界，不能用固定 eps 抹掉弱信号。实施前需在 research 中记录 SVD 正项均值 c 的误差界推导，对照 oracle trace-scale 界验证 0、1e-160（无信息）与 1e-20（仍有信息）案例。若不能满足 §13.4 和既有边界证据，回到设计审核，不任意调整阈值。alpha2/算子非有限等硬失败不得伪装无信息。

## Diagnostics and failures
常规模式不保存逐层大矩阵；保留逐样本 no_information、message_rejected、precision_capped、posterior_variance_underflow 计数。
return_diagnostics=True 保存 A2 所需逐层向量/标量；r2/gamma2 为入层消息，无需每层复制 Gram/SVD。
非法输入抛 ValueError；分解/算子硬失败抛可读 FloatingPointError，含算法、适用层号、batch index、dtype/device，调用方关联 sample ID。不得吞掉失败或自动 jitter。

## Independence and compatibility
保留 `reference/dense_vamp_cholesky.py` 独立 Cholesky 路线。生产不能调用 reference 数值算子，reference 不能导入生产 posterior/messages。
若发现 oracle 缺陷，单独记录执行书依据、独立重现和修复理由再评审，不同时改两边制造等价。正式结果不依赖 ReferenceResult。
复用既有配置覆盖，不改变 base.yaml 默认、CLI 或数据 schema；若确需新键，两个 base 模板必须同步并重新审核范围。

## Validation and rollout
先小矩阵解析，再逐层独立 oracle，最后真实 400 维物理输入。双精度按执行书容差；单精度独立记录容差及条件数依据。
SVD 冷启动成本留给后续公平计时；本 WP 没有性能结论。回退只撤销本 WP 新模块/测试/配置，保留前序物理数据和历史证据。
