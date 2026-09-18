# WP5 technical design — approved for implementation

## Modules and API
新增 `src/pgvamp_ofdm/algorithms/pg_vamp/{__init__,topology,majorizer,linear,model}.py`，
由 algorithms 导出 `PGVAMPDetector(nn.Module)`。QPSK/messages 直接复用 WP4 生产模块，
不再建重复 helper；reference 数值实现独立。
`forward(H,y,sigma2,*,return_diagnostics=False)` 和同签名 detect 返回 DetectionResult；
detect 委托 forward，不自动 no_grad，后续训练可对 x_soft 反向。
构造器接收 depth、实 dtype/device 和现有 pg_vamp 数值设置；默认 CPU/float64 参数匹配
complex128 输入，不新建 schema，不接受 hard mask/随机 trace。显式设备使用现有设备解析；
参数和输入 dtype/device 不匹配报错，不静默移动。配置固定策略与 §20 保持一致。
WP3 detection_inputs → 显式 batch → PG 模型 → 公共后验/消息 → DetectionResult。
ID/hash/标签留在调用方，本 WP 不实现损失、优化器或 checkpoint。

## Topology and majorizer
只在 model 注册 raw_gaps/raw_mu，topology 按 §14.1–14.2 生成 rho/mu/M。
零列除法先保护、非对角零边为零、主对角为一；不对称化 H/M。
majorizer 实现 d=sum((1-M²)|H|²)、G、非负 O(N²) ell 与 Gbar。
exclusive row sum 使用前缀/后缀求和，避免 row_sum-self 的严重相消；
测试以独立小矩阵定义求和对照，不能复用同一个 helper 自证。
仅理论 Hermitian P 可做舍入级对称化。jitter 使用已有显式配置，默认 0；
非零时只加到实际 P，全部 solve/W/c/方差共用，记录数值，不改变真实 A。

## Linear layer and differentiation
linear 接收 H/y/sigma2/r2/gamma2/M/mu，按 §14.4–14.5 执行。
每层一次 Cholesky；innovation/W 共用 factor，apply_A 始终用完整 H。
K=W/c，r1=r2+innovation/c，方差为两项 Frobenius 平方和，alpha2=1-c。
独立检查发现极弱非对角输入的直接 quotient backward 溢出；正式数值求值将两个
除 c 写成实部/虚部分别连续两次除 sqrt(c)，保持上述公式、实际 c 和阈值不变。
不跨层缓存 gamma2 相关因子，不跨调用复用旧参数图；前向不 detach。
全图使用测试专用内部拓扑注入，不公开主推理 mask 切换选项。
Jacobian 固定当前层 H/gamma2/M/mu，只对实展开 r2 求导；整网梯度保留所有依赖。

## No-information and hard failures
采用独立 PG oracle 的实际 W/H trace contraction 界，而非 VAMP spectral positive-sum 界：
S=sum_ij |W_ij||H_ji|/N，eta=4N*eps/(1-4N*eps)，tol=max(eta*S,tiny)。
要求 4N*eps<1，先检查算子有限；c>tol 才计算 K/r1/方差，不加 max(S,1) 固定量级下限。
该界仅解释给定已算 W/H 的乘加舍入，不是 Cholesky 总误差或状态演化定理。
c<-tol 是理论正性违反的硬失败；|c|<=tol 返回 r1=gamma1=0 并记录 no_information。
H=0 或无信息整 batch 保留 raw 参数零依赖，backward 返回零梯度。
检验 float64 的 0、1e-160、1e-20 倍单位阵以及非对角消去/秩亏案例。
实施阶段须独立审计此界和极端尺度；出现合同冲突先回 planning，不任意放大阈值。
P/W/innovation/方差非有限、Cholesky 失败或浮点界不适用须带算法/层/batch/dtype/device 报错。

公共 nonlinear_message 采用 WP4 保护：reciprocal/divide 前判断，保留候选均值，
精度下限 1e-10/上限 1e8、分母 margin=1e-6，极端有限输入也要测有限 backward。
如 oracle 边界存在缺陷，独立记录最小复现及源依据再评审，不改两边制造一致。

## Diagnostics
常规保留逐样本 no_information/message_rejected/precision_capped/posterior_variance_underflow；
小标量诊断逐层保存 rho/mu/c/tol/jitter、非零非对角候选边中 M>=0.5 比例及相对 N(N-1) 比例。
空候选或 N=1 时 ratio=0，另记 denominator=0，避免误导分母。
安全项相对大小定义为 ||ell||_2/||diag(G)||_2；均零时记 0 及零基准标记。
两个诊断 norm 先用同一个可消去尺度缩放，避免有限极端能量的平方溢出/下溢。
详细模式供小矩阵测试保存入层 r2/gamma2、M/d/ell/G/Gbar/P/W/K、innovation/xhat2、
c/alpha2、r1/gamma1、xhat1/概率/vbar/alpha1 及保护状态；测试层状态保留梯度。
仅日志副本可 detach，常规推理不保存所有大矩阵，最后一层不生成无用外信息。

## Compatibility and rollback
保持 DetectionResult、MMSE/VAMP 数学和配置兼容。若公共 helper 必须修复，执行全部基线回归。
独立 oracle 不导入生产 helper。先小矩阵数学验收，再真实 400 维三算法前向。
稠密 Cholesky/多右端项/精确方差仍可能 O(N³)，内存至少 O(N²)，不承诺稀疏加速。
回退仅撤销 WP5 文件及导出/文档改动，不删除前序数据、不降低既有验收门槛。
