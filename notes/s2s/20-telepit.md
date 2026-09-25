# TelePiT：将球面编码、多尺度 ODE 与遥相关注意力直接用于第 3–6 周

> Tengfei Lyu、Weijia Zhang、Hao Liu，*Physics-Informed Teleconnection-Aware Transformer for Global Subseasonal-to-Seasonal Forecasting*，*KDD 2026*，pp. 1054–1065，DOI [10.1145/3770854.3780198](https://doi.org/10.1145/3770854.3780198)，校方机构库记录 2026-04-20 已发表；实验与表格以下依据开放的 [arXiv:2506.08049v3](https://arxiv.org/html/2506.08049v3)（2025-08-11），代码依据[作者仓库固定提交](https://github.com/tfeilyu/TelePiT/tree/0e6e67285231c6b96cfb040d58f76ce374168528)。初读 2026-09-24；正式书目和代码复核 2026-09-25。**正式发表书目已核验，但 ACM 最终 PDF 未成功取得，不能推定正式版数值与 v3 完全一致。**[HKUST 机构库](https://researchportal.hkust.edu.hk/en/publications/physics-informed-teleconnection-aware-transformer-for-global-subs/)

## 1. 任务和设计

TelePiT 不是逐日滚动 42 天，而是由初始全球场**直接预测**第 3–4 周（天 15–28）与第 5–6 周（天 29–42）的两周平均场。此任务显著不同于逐日中期预报：时间平均使天气噪声抵消，也改变 ACC 与 RMSE 的难度。作者假设球面几何、多尺度演化和遥相关有助于超过两周的预测，但其“物理启发”是在潜空间加入平流/扩散形式，并非求解完整流体方程，也不是严格的守恒约束。[任务定义及方法](https://arxiv.org/html/2506.08049v3)

### 数据与标签构造

| 项目 | 原文设定 | 复现注意 |
|---|---|---|
| 数据 | ChaosBench 处理的 ERA5 再分析 | 非原始全分辨率 ERA5 |
| 空间 | 1.5° 全球网格，121 × 240 | 模型内部把经度先聚合 |
| 高空变量 | `z`、`q`、`t`、`u`、`v`、`w`，各取 10、50、100、200、300、500、700、850、925、1000 hPa | 60 个高空通道 |
| 地表变量 | 2 m 气温、10 m 纬向风和经向风 | 总共 63 通道 |
| 时间 | 1979–2016 训练、2017 验证、2018 测试；附录另用 2019 样本外检查 | 两段标签分别是天 15–28、29–42 的逐日场平均 |

因此输入可表示为 `63 × 121 × 240`，输出是两个相同空间网格的未来时段平均场。公开配置将高空变量置于 `era5`、三个近地面变量置于 `lra5`，`oras5` 变量列表为空；不能因此把全部 63 通道笼统说成单一 ERA5 文件。数据代码按年份匹配 `YYYYMMDD.zarr`、文件名排序，并对索引 `idx` 取单时刻输入；`lead_time=15,n_step=28` 读取后续 28 个目标文件，前 14 个与后 14 个分别逐格平均成输出。若文件确为连续逐日，才对应天 15–28 和 29–42；代码没有自行检查缺日或逐日间隔。[配置](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/configs/TelePiT.yaml#L10-L23)｜[数据集索引/标签](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/dataset.py#L65-L82)

标准化不是未知：`S2SDataset` 从 `climatology_era5.zarr`、`climatology_lra5.zarr`、`climatology_oras5.zarr` 按变量/层抽取 `mean` 和 `sigma`，输入及每个目标日都做 `(x−mean)/sigma`，**然后**对 14 日的标准化目标求平均。这些气候态文件未随模型代码提供，统计期、是否排除验证/测试年、格点预处理与缺测 QC 仍无法独立审计。代码即使 `oras5_vars=[]` 也试图打开 ORAS5 气候态文件，是部署前要核实的依赖。评估脚本计划在预测后反标准化，但训练入口引用的 `predict.py` 并不存在（见第 3 节）。[数据集标准化](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/dataset.py#L88-L102)｜[逐样本处理](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/dataset.py#L109-L162)

```mermaid
flowchart LR
  X[ERA5 初始全球场] --> Z[每条纬线先做经度平均]
  Z --> S[球面经纬正余弦嵌入]
  S --> M[可学习多尺度频带分解]
  M --> O[带物理约束的 Neural ODE]
  O --> T[显式遥相关注意力 + 跨尺度交互]
  T --> H1[第 3–4 周平均场]
  T --> H2[第 5–6 周平均场]
```

结构图据原文 Figure 1 和式 2–19 自绘。图中最值得留意的是，经度平均把 `121 × 240` 空间网格压缩成 121 个纬线 token，然后才进入注意力。这显著降低计算量，却可能丢失局地经向非对称信息；末端 MLP 虽再次输出所有经度，并不能自然恢复在输入压缩中丢弃的信息。[方法 §3](https://arxiv.org/html/2506.08049v3)

### 各模块的输入、输出与物理含义

1. **球面位置嵌入（SHE）**：纬度和经度各使用多频率 `sin/cos` 编码。气象场沿经度平均成每条纬线 63 维；经度位置编码也求全局平均，再与纬度编码拼接，得到 `H × D` token 序列。原文称其近似球谐表示，但并未显式求球谐系数或保留所有经向模态。[式 2–8](https://arxiv.org/html/2506.08049v3)
2. **可学习多尺度分解（WD）**：每层两层 MLP + GELU 输出粗近似和细节两组，反复处理后得到 `L+1` 组特征。论文类比小波，但式 9 未明确正交滤波器、降采样或严格可逆变换，因此不能把潜特征严格当作物理频带。[式 9–10](https://arxiv.org/html/2506.08049v3)
3. **潜空间 ODE**：每组特征分别演化，邻纬线二阶差分作为扩散项，一阶中心差分作为平流项，再加可学习外强迫与 MLP 校正；`tanh` 和 `γ` 限制变化率。正文明确写固定步长 Euler 更新。系数是潜特征参数，没有披露真实物理单位。[式 11–13](https://arxiv.org/html/2506.08049v3)
4. **遥相关注意力（TA）**：把纬线状态平均成全局向量，对 `n_p` 个可学习模式作 softmax 加权，再把所得模式投影到 query 空间，并作为各纬线 key 的注意力偏置，强度由 `λ` 控制。模式是从数据学习，**并非直接输入已观测的 MJO、ENSO 或 NAO 指数**；是否代表具体气候遥相关仍需独立验证。[式 14–15](https://arxiv.org/html/2506.08049v3)
5. **输出头**：尺度间拼接、MLP、LayerNorm 后跨尺度取平均；每条纬线上的两层 MLP 输出 `2 × C × W`，分别重排成两个 14 天平均场。[式 16–19](https://arxiv.org/html/2506.08049v3)

**原文内部不一致**：正文给 ODE 极点零填充、固定步长 Euler，而附录理论分析谈到周期边界与自适应步长。公开代码现可确定该版本使用**纬度常数零填充**和 `torchdiffeq.odeint(method='euler',step_size=0.1)`，积分区间 `[0,1]`；这解决了公开实现如何运行的问题，仍不能证明发表模型与这份提交完全同源。求解器异常时退化成 `no_grad` 计算导数的单次 0.1 步，梯度路径不同于正常 ODE。[正文 §3.2、附录 A.5](https://arxiv.org/html/2506.08049v3)｜[ODE 源码](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/models/TelePiT.py#L181-L273)

### 训练流程、参数及微调

训练损失为两个预测时段均权的像素均方误差，式 20 以 `2CHW` 归一化。评测 RMSE 使用纬度余弦权重，但训练式未显式使用该权重。TelePiT 是在上述 ERA5 样本上直接训练；文中没有单列大模型预训练后再微调 TelePiT 的阶段，不能把基线模型的预训练混为其微调。[式 20、实验 §4](https://arxiv.org/html/2506.08049v3)

| 参数/资源 | 公开值 | 未公开或限制 |
|---|---:|---|
| 实现 | PyTorch Lightning；公开代码 `seed_everything(42)`、`deterministic=True`、自动统计可见 GPU 数 | 未核验所发布权重与当前源码完全同版 |
| batch size | 16 | 未交代是单卡还是全局 batch |
| 结构 | 256 维；4 个尺度（分解层数 3），每尺度 6 Transformer block、8 heads、MLP ratio 4、drop/attn_drop 0.1 | 代码模型默认值；不等于单一 6 层整网 |
| 优化器 | AdamW，初始 LR 0.01；β 和 weight decay 未显式覆写，依赖 PyTorch 默认值 | 不能猜测发表实验的所有版本依赖 |
| 日程 | 最多 200 epoch；CosineAnnealingLR `T_max=500` epoch、`eta_min=0.001`；val loss 十轮无改善早停、保留前三和最近 checkpoint | `T_max` 超过训练上限，因此不能说运行 200 epoch 已降到 `eta_min` |
| ODE | 扩散系数初始化 0.001、平流 0.01、强迫 0、校正项系数 0.1；导数末端 `tanh(·)×0.5` | 系数是可学习潜空间参数，不对应固定物理扩散率/风速 |
| ODE 积分 | Euler，积分时间 1.0，固定步长 0.1；极点纬向零填充 | 附录周期边界/自适应求解器叙述与本提交不一致 |
| 遥相关 | 5 个可学习模式，注意力偏置系数硬编码 0.2 | 没有外部 ENSO/MJO 指数输入；附录图 20 有系数灵敏度 |
| 训练硬件 | 4 张 GeForce RTX A40 | 完整训练时长未报告 |
| 复杂度 | 37M 参数、14.5G FLOPs、141.64 MB、32.25 ms/样本 | 条件依赖输入尺寸与硬件，不可直接外推运营成本 |

ClimaX、CirT 按相同设置重新训练；FourCastNetV2、Pangu-Weather、GraphCast 使用 ECMWF AI models 提供的预训练版本，在 A800 80GB 推理。GraphCast 第 5–6 周因显存不足无结果。于是“相同配置”仅适用于部分直接训练的基线，不能推论全部模型已严格统一训练数据、变量集合与算力。[实验 §4、附录 B.2–B.3](https://arxiv.org/html/2506.08049v3)

上述 AdamW/调度/早停来自[训练入口](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/train.py#L69-L109)、[Lightning 封装](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/models/model.py#L88-L140)及[发布配置](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/configs/TelePiT.yaml)，并非正文逐项披露。公开源码中训练损失是标准化标签上的 `MSE()`，验证/早停的 `val_loss` 也用 MSE，而 `test_step` 才用 `RMSE()`；论文纬度加权物理单位 RMSE 是另一个评测层，不能把训练监控指标直接等同 Table 1。[源码训练/验证/测试](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/models/model.py#L78-L119)

**源码复现阻断**：固定提交的 `train.py` 顶层写 `from predict import reverse_normalize`，仓库文件清单没有 `predict.py`；虽然 `step1_predict_to_npy.py` 另有同名函数，不能不改代码就执行入口。此外 `S2S/models/model.py` 无条件导入 `CirT`、`ClimaX`、三个 TelePiT 变体及四个消融模块，公开的 `S2S/models/` 仅有 `TelePiT.py` 和 `model.py`；因此即使只选 TelePiT 也会在导入阶段失败。README Quick Start 的配置路径/CLI 与 `--run_model TelePiT` 实际接口也不一致。这些是**当前公开提交**的可运行性问题，不能反推作者训练时没有内部完整代码，更不能据此把论文报告数值判为伪造。[训练入口](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/train.py#L1-L29)｜[模块导入](https://github.com/tfeilyu/TelePiT/blob/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/models/model.py#L1-L11)｜[源码目录](https://github.com/tfeilyu/TelePiT/tree/0e6e67285231c6b96cfb040d58f76ce374168528/S2S/models)

## 2. 数值结果与异常核对

| 原文 Table 1，天 15–28 | TelePiT | CirT | 单位/警告 |
|---|---:|---:|---|
| Z500 RMSE | **48.671** | 53.807 | 表标 `gpm`；须核对和 CirT 原表的位势单位转换 |
| T850 RMSE | **1.873** | 2.051 | K，约改善 8.7%（按 CirT 为分母） |
| U10 RMSE | **0.491** | 0.567 | m/s；与数周逐日风场误差不可比 |
| T2m RMSE | **12.057** | 28.526 | 表标 K，数值**异常大**：CirT 原文同类任务 T2m 约 2 K；可能评测尺度/单位/标准化不一致，不能据此宣称 57.7% 的可信温度提升 |
| T2m ACC | 0.996 | 0.977 | 高相关需排查残余季节循环/气候态定义与两周平均协议 |

原文 Table 1 还给第 5–6 周 Z500 47.761、T850 1.894、T2m 12.063。若后一期很多 RMSE 反而略低于前一期，必须检查预测目标、气候态与分母；不能把该表当作独立业务基准的终审结果。原文比较还包括 FourCastNetV2、Pangu-Weather、GraphCast、ClimaX 及 ECMWF/UKMO/NCEP/CMA，部分基线字段缺失，并非所有变量都有同质对照。[原文 Tables 1–3](https://arxiv.org/html/2506.08049)

原文纬度加权 RMSE 使用归一化的 `cos(latitude)`，ACC 对预测和观测分别减去气候态，SpecDiv 对归一化功率谱计算相对熵；附录另给 MS-SSIM、SpecRes。低 SpecDiv 不能替代守恒验证，高 ACC 也未证明极端事件的空间定位可靠。业务数值系统之间的发行频率和成员设置并不一致，表 3 的比较需要进一步核对配对初值与缺失样本处理。[附录 B.1–B.2](https://arxiv.org/html/2506.08049v3)

| 原文 Table 4，2018 第 3–4 周 | Z500 RMSE | T850 RMSE | T2m RMSE | U10 RMSE |
|---|---:|---:|---:|---:|
| 完整 TelePiT | **48.671** | **1.873** | **12.057** | **0.491** |
| 去 SHE | 51.217 | 1.983 | 27.206 | 0.562 |
| 去 WD | 51.949 | 2.018 | 22.901 | 0.542 |
| 去 ODE | 50.582 | 2.002 | 22.780 | 0.542 |
| 去 TA | 51.333 | 2.023 | 23.923 | 0.548 |

在作者流程下，四种消融均降低表现，支持模块设计有作用；但 T2m 的可疑量级同时出现在消融表，不能仅凭巨大的 T2m 消融差距判断物理正确性。附录把 2019 年当额外样本外检查，仍报告 T2m 12.81 K、CirT 28.42 K：重复出现降低了“单格排版错误”的可能，却没有消除单位/标准化/变量协议错误的风险。[Table 4、附录 B.7](https://arxiv.org/html/2506.08049v3)

## 3. 阅读判断与复验

思路上，显式把非局地遥相关注入注意力值得与 CirT 的球面归纳偏置、FuXi-S2S 的多变量延伸比较。但当前最重要的是**表格审计**：先补全缺失模块/气候态文件并锁定依赖；用同一批 ChaosBench 样本核对连续日期和两个两周平均标签；在逆标准化后重算 T2m 与 Z500 单位；统一 2018 初值、缺测和气候态基期；用公开代码的零边界/固定 Euler 与附录设定分别对照；检查经度平均的信息损失；再用多随机种子验证消融。没有这些复核，v3 数值排序只能记录为作者报告，不能作为可靠模型选型依据。正式书目由 HKUST 核验，正式版实验 PDF 未直接核验。[开放 v3 方法及实验](https://arxiv.org/html/2506.08049v3)｜[正式书目](https://researchportal.hkust.edu.hk/en/publications/physics-informed-teleconnection-aware-transformer-for-global-subs/)

[返回首页](../../README.md) · [返回总表](../../气象大模型_中期预报论文追踪.md)
