# WP1 technical design

## Boundaries and data flow
Config + Runtime + explicit seeds -> allocation + bits/pilots -> X_grid
-> ortho IFFT + CP -> frame analytic passband -> sqrt(2)*real export。
相关器只接收 waveform/template，不读取 arrival_offset；审计层比较真值。
以下是待实现接口，不是当前已有 API。

| 文件 | 接口与责任 |
| --- | --- |
| modulation/qpsk.py | bits_to_classes、classes_to_bits、classes_to_symbols、bits_to_symbols、hard_decision；暂不实现 §14.6 posterior/messages |
| modulation/allocation.py | build_allocation(config, device) -> Allocation；map_grid(data, pilots, allocation) -> X_grid |
| waveform/ofdm.py | modulate_grid(X_grid, config) -> useful、with_cp；最后维为时间 |
| waveform/lfm.py | tukey_window、make_lfm(config, runtime) -> analytic/real template、normalization/window metadata |
| waveform/frame.py | build_frame(data_bits, pilot_symbols, config, runtime) -> FrameWaveforms + FrameLayout；录音 offset 独立显式 padding |
| receiver/synchronization.py | normalized_correlation(recording, template, threshold=0.1) -> SyncResult；仅 valid lag |
| scripts/run_wp1_audit.py | 发送波形审计、导出、画图；不是 dataset generate/audit、evaluation/report 或完整 smoke |
| tests/test_qpsk.py、test_allocation.py、test_waveform.py、test_lfm.py | PRD A1–A8；已有 config/device/CLI/reference tests 保留 |

不创建 channel/、waveform/continuous.py、receiver/fft_receiver.py 或 preprocessing.py。
不改 reference 数值算子；共享生产 QPSK 以后由 WP4 接入正式算法。

## Shapes, ordering, errors
- bits 为 bool/uint8/int64 [...,2]，只含 0/1；class 为 int64 [...]、0..3；
  symbols 为 complex128/complex64。hard_decision 支持 finite 复输入，以实/虚严格小于零取位，
  零坐标选正号，等价于最小类平局；与独立四点枚举比较。
- frame bits [B,M,400,2]、pilots [B,M,64]、grid [B,M,512]、
  useful [B,M,Nw]、with_cp [B,M,Nw+Ncp]、frame [B,Nframe]。
  默认 M=8/Nw=8192；支持已验证配置的 M/CP/FFT 参数，不硬编码所有偏移。
- Allocation 包含 int64 q/grid_index/baseband_fft_index/passband_fft_index、
  data_q/pilot_q/guard_q/dc_q 及各组 grid positions；同 device、q 严格升序。
  默认 q=-256..255、q+256、q%8192、2048+q；其他有效配置按 carrier_hz/df+q 推导，
  拒绝 DC/Nyquist/镜像重叠的通带配置。
- 复用 load_config/resolve_runtime；拒绝 algebra_fixture 进入物理帧 API。
  检查 shape、空 batch、位值、有限性、dtype/device、配置一致性、符号能量；
  错误抛可读 ValueError/ConfigError，不隐式移动 device 或归一化标签。

## Seeds and compatibility
核心组帧显式接收 bits/pilots，不消费全局 RNG。审计从配置 master seed
用 SHA-256 对带版本的 bits/pilots/arrival_offset/sync_noise_real/sync_noise_imag
域标签派生独立种子，保存派生规则和实际种子，使用独立 torch.Generator。
固定 pilot seed 生成 [B,M,64] 单位能量 pilots；改变 bits seed 不改变 pilots。
不提前实现 WP3 数据格式或随机流框架。
无需新增配置键；若必须改变 schema，回到规划。两个 base.yaml 保持一致；
inspect-config 继续仅做静态检查。

## OFDM and absolute phase
q%Nw 放置网格，ifft(dim=-1,norm='ortho')，CP 只复制末尾，不乘 16 或 sqrt(16)。
FrameLayout 以半开区间保存 LFM、各 CP/useful、静默边界；默认：
LFM [1920,5760)，首 CP [7808,9856)，首 useful [9856,18048)，
第 m 个 useful start=9856+m*10240，尾静默 [89728,91776)。
OFDM 解析通带使用发送帧全局 n/Fs 载波，不在 CP/块边界重置。
LFM 按 §4.4 以 LFM 起点局部 t 定义相位，不给已是通带的 chirp 再乘载波。
保存解析通带及 sqrt(2)*real；若返回复基带帧，必须由全局下变频定义，
不得把解析 LFM 与基带 OFDM 直接混接。
录音 offset 是发送样本前置零，不因录音原点变化重新调制波形。

无信道恢复独立测试：有效段实波形实际 ortho FFT，取 carrier_bin+q 再乘 sqrt(2)。
默认有效起点载波相位为整周，直接恢复 X_grid。另测非整周期起点配置，
明确消去已知发送时间载波相位 exp(j*2*pi*fc*T_m)，不偷改为逐块重置相位。
此检查不是 WP2 正式接收器，也不证明时变实前端等价。

## LFM discrete contract
N_L=lfm_samples，T_L=N_L/Fs，beta=(f1-f0)/T_L，t=arange(N_L)/Fs。
对称离散 Tukey：u=n/(N_L-1)，alpha=0 返回全 1；
alpha>0 时 u<alpha/2 用 0.5*(1+cos(pi*(2*u/alpha-1)))，
中段为 1，u>1-alpha/2 用 0.5*(1+cos(pi*(2*u/alpha-2/alpha+1)))。
alpha=1 为 Hann；N_L<2 拒绝；保存 symmetric、N_L、alpha 和表达式版本。
A=sqrt(((Nd+Np)/Nw)/mean(w**2))，
p_a=A*w*exp(j*2*pi*(f0*t+beta*t**2/2))，p_RF=sqrt(2)*Re(p_a)。
功率合同为 mean(abs(p_a)**2) 等于配置 OFDM 复包络平均功率。
另外报告实通带功率，不再次归一化，也不依赖当前标签。
在非零窗口内部检查相位差与上扫频；末采样频率小于 f1，不要求采样到 T_L。

## Basic matched correlation
recording [...,L]、一维 template [N_L]，实/复及配对精度、同 device；
要求 L>=N_L、模板 finite 且能量>0；短录音、不兼容 dtype、非有限值拒绝。
依 §7.1 求 d=0..L-N_L；FFT 线性卷积充分补零，核 conj(flip(template))，
从卷积 N_L-1 处截取 valid 段；滑窗能量用前缀和。
负的舍入能量残差仅按 dtype 误差尺度处理并测试；不能掩盖实质计算错误。
固定 epsilon_sync=finfo(real_dtype).tiny，写入结果；全零得分为 0。
不裁剪 C 掩盖错误，检查分母/得分 finite；不可表示的极端幅度明确报错。
SyncResult：scores、peak_start、peak_score、detected（严格 > threshold）、
candidate_arrival（检测成功时为 peak_start，失败为 None 或显式无效 mask）。
同分峰选最小 d；arrival 指模板到达，不是物理首径。
无信道审计可将 peak_start-leading_silence 与注入原点比较，
但不得把真值喂给同步器。

短序列以直接滑动内积独立核对；默认长模板测首端、中间、最后 valid lag。
全零不报检。固定种子纯噪声多个窗口保存峰值/报检数；复白噪声的单窗口
归一化相关 Beta(1,N_L-1) 尾概率及 union bound 用于解释最大峰统计容差；
实噪声另记实测计数，不声称虚警率恒零，不用测试集调 0.1。
不做多径首径校正、未知 Doppler 联合搜索、误定时 H 构造或检测性能报告。

## Audit outputs
待实现命令：
python scripts/run_wp1_audit.py --config configs/cpu_dev.yaml --output runs/wp1-audit

保存 resolved_config.yaml、environment.json（执行书/代码哈希、软件、种子、
device/dtype/线程）、waveforms.pt（仅 tensors/basic types，Fs、布局、索引）、
psd.npz、correlation.npz、figures/psd.png、figures/correlation.png、summary.json。
区分整帧、OFDM useful-only、LFM 的 PSD，不混淆功率统计区间。
实波形双边 periodogram=abs(FFT_unscaled)**2/(Fs*N)，积分乘 df 应为 mean(rf**2)；
带内取 ±[21,27] kHz 并集，报告其外能量比例。图显示正频率 0..Fs/2 并说明镜像；
不因带外能量非零而加 OFDM 窗。
相关图 x 为模板起点样本/秒，标阈值和注入位置，保留原始数组。
complex128 atol=1e-9/rtol=1e-8；整数/CP/长度精确断言。
complex64 FFT atol=2e-5/rtol=2e-4；长通带相位用 float64 时间/相位后显式转换，
记录转换以控制累积误差，不静默放宽容差。
GPU 仅实际可用且显式请求才验证；CPU 路径不探测 CUDA。

## Risks and rollback
风险：全局/局部时间混淆、卷积 lag 偏 N_L-1、重复 sqrt(2)、RNG 耦合、
将 WP1 审计冒称完整 smoke。逐层独立断言与原始数值数据作为验收依据。
无数据迁移；按新增模块/测试/审计脚本撤回当前补丁，保留 WP0 和用户文件。
后续 WP2 接收 FrameLayout/grid/解析通带边界，本次不创建其业务实现。
