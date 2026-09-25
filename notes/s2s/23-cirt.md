# CirT：环形纬圈 patch 与频域注意力的全球第 3–6 周预测

> Yang Liu、Zinan Zheng、Jiashun Cheng、Fugee Tsung、Deli Zhao、Yu Rong、Jia Li，*CirT: Global Subseasonal-to-Seasonal Forecasting with Geometry-inspired Transformer*，[arXiv:2502.19750v1](https://arxiv.org/abs/2502.19750)，2025-02-27；[ICLR 2025 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)；[作者代码（固定提交 a5a9b66）](https://github.com/compasszzn/CirT/tree/a5a9b666e553b3c1caa1e167073649d478b79e94)。初读：2026-09-24；正式论文与代码复核：2026-09-25。**以下将论文报告、代码实现和阅读判断分别标注，不能把代码默认值当作论文实验已确认设置。**

## 1. 几何动机与方法

经纬度图像同尺寸 patch 在球面上面积不同，左右边界又应首尾相接。CirT 沿纬度将每条纬圈作为**环形 patch/token**，在注意力模块前后进行 Fourier/逆 Fourier 变换，再由 Transformer 直接输出第 3–4 周和第 5–6 周的双周平均场。这里的关键是**直接预测多周均值**，不是把 6 小时天气模型滚到 42 天；两种任务误差量级不能直接相减。值得细辨：原文式 6 的 DFT 实际作用于每条纬线经过线性投影后的**长度 `D` 潜特征轴**，并非明确对原始 240 点经度序列直接做 FFT；“经向频谱”是作者的设计诠释，严格物理解释仍需验证。[方法 §3，式 5–10](https://arxiv.org/html/2502.19750v1)

### 数据来源、字段与标签

| 项目 | 论文可核实配置 | 阅读提示 |
|---|---|---|
| 再分析 | ERA5，1.5°，121 × 240 全球经纬网格 | 每条纬线在球面上的实际长度不同 |
| 高空 | 位势、比湿、温度、纬向风、经向风、垂直速度，分别取 10、50、100、200、300、500、700、850、925、1000 hPa | 六变量 × 十层 = 60 通道 |
| 地表 | 2 m 气温、10 m `u` 和 `v` 风 | 总共 63 通道 |
| 年份 | 1979–2016 训练、2017 验证、2018 测试 | 按年份隔离；初值频率及标准化系数未完整披露 |
| 监督标签 | 第 15–28 天、第 29–42 天各自的时段平均场 | 不是第 28/42 天某个瞬时场 |

训练输入是初始全球场和位置网格，输出同时包含两组 63 通道平均场。正式论文附录 A.1.2 明确：对高分辨率 ERA5 和部分基线预报，从 0.25° 网格**按重合坐标抽点**到 1.5°，不是区域均值或双线性插值。作者代码的两份下载脚本都读取 WeatherBench 2 ERA5 *daily* Zarr，日期为 1979-01-01 至 2018-12-31，分别筛选 6 个高空变量与 3 个地表变量；`latitude`、`longitude` 下标每隔 6 点选一个，`fillna(0)` 后按日写为 Zarr。因此“抽点”和“缺测置零”有代码证据，而这些处理是否等同于表 1 全部官方基线的生产流程，论文没有逐模型文件级核验。[正式论文附录 A.1.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)；[高空下载脚本](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/data_download/training_data/step1_pressure_level_download.py)；[地表下载脚本](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/data_download/training_data/step2_single_level_download.py)。

代码里的标准化脚本对 `pressure_level_1.5` 和 `single_level_1.5` 目录下**所有**日文件做每通道、跨时间与空间的 `nanmean/nanstd`，保存 `mean/sigma`；数据集读取时将输入和目标都按对应通道 `(x-mean)/sigma`。按下载脚本默认日期，目录会包含 2017、2018 年，因此如果原样依次运行下载与统计脚本，验证/测试年份会参与标准化统计，存在评测期信息泄漏；这是一项**代码路径风险推断**，不能断言论文表 1 确实这样运行。均值/标准差不是按经纬格点逐点计算，也不是仅以训练年份限定；论文未给标准化统计文件或清楚的气候态基期。[统计脚本](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/data_download/training_data/step3_compute_climatology.py)；[数据集实现](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/CIRT/dataset.py)。

作者代码在 `n_step=28, lead_time=15` 下取初始日之后第 15–42 天的日场，前 14 天和后 14 天分别求平均，再形成 `2 × 63 × 121 × 240` 的监督标签。按年选文件时，`__len__` 会丢掉末尾无法凑足 42 天的初值，故并非某年每天都能成为该年样本；实际分母、闰年与初始化频率应按代码生成样本检查。论文附录的经度端点列表同时写“1440 个 0.25° 格点”与包含 `180°` 的闭区间，数量不自洽；实际代码使用 `np.arange(..., step=6)` 下标抽样，复现应以原始数组坐标核对。[数据集实现](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/CIRT/dataset.py)。

```mermaid
flowchart LR
  X[ERA5 初始全球经纬场] --> P[按纬度切成环形 patch]
  P --> E[展平 240 × 63 并投影为 D 维 token]
  E --> FFT[沿潜特征 D 轴作 DFT]
  FFT --> A[实部与虚部拼接后跨纬线注意力]
  A --> IFFT[逆变换回空间]
  IFFT --> H1[第 3–4 周平均场]
  IFFT --> H2[第 5–6 周平均场]
```

上图依据原文 Figure 2 和 Methods 重绘。训练与评价使用 ERA5；原文把 RMSE、ACC、物理结构相似度与多种 ML/业务 S2S 基线相较。模型规模表给 CirT **16M 参数、约 2.2G FLOPs**，明显小于所列 Pangu-Weather 256M 和 GraphCast 37M；但 FLOPs 的计算对象、输入分辨率及一次预测范围必须同口径方能比较真实业务成本。[原文 Table 4](https://arxiv.org/html/2502.19750)

### 结构张量流、训练目标与参数

每条纬圈包含 `W × K = 240 × 63` 个数；先展平后用可学习矩阵映射到宽度 `D=256`，加纬线位置嵌入，得到 `H × D=121 × 256`。每个 Transformer block 对长度 `D` 的纬线 token 做 DFT，拼接复数系数的实部和虚部，再将各纬线作为注意力 token 做多头混合；混合后重新组合复数并逆变换，最后 MLP 形成下一个 block 输入。输出展平 MLP 一次性生成两个未来平均窗。[式 5–10](https://arxiv.org/html/2502.19750v1)

训练目标是两个 14 天平均场的均权网格 MSE，公式以 `2×K×H×W` 归一化。评测则用按 `cos(latitude)` 归一的纬度加权 RMSE，以及相对经验观测气候态异常的 ACC；训练式未显示纬度权重。**主模型**在所列 ERA5 训练集上直接训练，未先以外部基础模型预训练；但正式论文**附录 Table 6 确实另做 CirT 自身的微调实验**，详见下节。FourCastNetV2、Pangu-Weather、GraphCast 是**对照模型**的预训练权重，作者没有针对 S2S 再微调，不能与 CirT 附录微调混淆。[正式论文式 11–12、实验 §4、附录 Table 6](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

| 公开超参数 | 数值 | 复现缺口 |
|---|---:|---|
| batch size | 16 | 未说明单卡还是全局 |
| 隐藏宽度、头数 | 256、16 | 对应直接训练模型的设置 |
| Transformer 层数 | 8 | 模型规模另报 16M 参数 |
| 学习率、训练轮数 | 0.01、20 epoch | 正式论文未说明优化器、调度器、权重衰减和随机种子 |
| 实现与硬件 | PyTorch Lightning；8 张 GeForce RTX 4090 | 正式论文说法；预训练基线在 A800 80GB 推理，硬件不同 |

ClimaX 按同配置重新训练，其余三个 AI 基线通过 ECMWF AI models 获取原训练权重；GraphCast 第 5–6 周推理显存不足。原文只说直接训练基线共享超参数，不能扩展成所有比较系统共享同一训练集和预报定义。[实验 §4](https://arxiv.org/html/2502.19750v1)

**代码现状与论文报告之间的差异。**固定的公开提交中，`CirT.yaml` 实际设 `train_years=1979..2016`、`val_years=[2018]`、`test_years=[2017]`，与论文“2017 验证、2018 测试”**正好相反**。`train.py` 设置 `CUDA_VISIBLE_DEVICES='0,1'`、`Trainer(devices=2)`，也与论文所述 8 卡不同。`model.py` 采用 `AdamW(lr=0.01)`，并使用按 epoch 更新的 `CosineAnnealingLR(T_max=500, eta_min=0.001)`；配置训练只有 20 epoch，所以实际调度不会走满 500 epoch。脚本 `pl.seed_everything(42)`，按验证损失最低保存 checkpoint，然后 `trainer.test(..., ckpt_path='best')`；由于配置年份对调，按默认代码复现时“best”来自 2018，最终测试却是 2017。这些是**公开代码默认路径**，不能据此推定论文数值来自该配置；复现论文必须先更正切分、检查统计期并确认训练设备。[配置](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/CIRT/configs/CirT.yaml)；[入口脚本](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/train.py)；[优化器及 dataloader](https://github.com/compasszzn/CirT/blob/a5a9b666e553b3c1caa1e167073649d478b79e94/CIRT/models/model.py)。

## 2. 原文性能与重要消融

下表完整转录正式论文 Table 1 的七个评测变量。RMSE 越低越好，ACC 越高越好；`—` 表示作者没有报告，不等于零，也不能据此排名。`z` 的单位是位势 `m²/s²`，不是位势高度米。两个预报窗都是双周平均，数字不能与逐日确定性 RMSE 或集合 CRPS 混用。[ICLR 2025 Table 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

| 目标窗 | 变量（RMSE 单位） | FourCastNetV2 | GraphCast | Pangu-Weather | ClimaX | CirT | CirT ACC |
|---|---|---:|---:|---:|---:|---:|---:|
| 周 3–4 | z500（m²/s²） | 615 | 618 | 649 | 602 | **477** | 0.984 |
| 周 3–4 | z850（m²/s²） | 402 | 411 | 416 | 372 | **304** | 0.963 |
| 周 3–4 | t500（K） | 2.093 | 2.176 | 2.271 | 2.186 | **1.687** | 0.988 |
| 周 3–4 | t850（K） | 2.390 | 2.370 | 2.569 | 2.618 | **1.903** | 0.988 |
| 周 3–4 | t2m（K） | — | 2.158 | — | 2.998 | **2.007** | 0.993 |
| 周 3–4 | u10（m/s） | 2.328 | — | 2.431 | 2.334 | **1.806** | 0.896 |
| 周 3–4 | v10（m/s） | 1.896 | — | 1.984 | 1.906 | **1.511** | 0.811 |
| 周 5–6 | z500（m²/s²） | 652 | — | 754 | 619 | **471** | 0.985 |
| 周 5–6 | z850（m²/s²） | 426 | — | 461 | 375 | **301** | 0.964 |
| 周 5–6 | t500（K） | 2.250 | — | 2.829 | 2.254 | **1.672** | 0.988 |
| 周 5–6 | t850（K） | 2.567 | — | 2.998 | 2.741 | **1.933** | 0.989 |
| 周 5–6 | t2m（K） | — | — | — | 3.168 | **2.026** | 0.992 |
| 周 5–6 | u10（m/s） | 2.479 | — | 2.679 | 2.355 | **1.809** | 0.895 |
| 周 5–6 | v10（m/s） | 1.980 | — | 2.104 | 1.939 | **1.512** | 0.812 |

表中优势需要放在实验边界内读：ClimaX 是按作者设定重新直接训练，其余三个 AI 模型用公开预训练权重、滚动到目标窗，并**没有** S2S 目标微调；这同时比较了**目标形式与结构**，不是纯粹架构的等资源消融。FourCastNetV2 的 2018 年 10 月数据遗失，只比较其余 11 个月；GraphCast 因显存只能推到第 4 周，第 5–6 周空缺。ECMWF AI-models 取到的 FourCastNetV2/Pangu 权重未提供表中 t2m 推理；GraphCast 风场亦缺项。正式论文没有逐模型列出完全配对的初值日期和样本量，故部分误差差异可能受配对样本不齐影响。[正式论文 §4、附录 A.1.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

| 原文表 | CirT | 基线/消融 | 解读 |
|---|---:|---:|---|
| Table 1，周 3–4 Z500 RMSE | **477** | ClimaX 602、GraphCast 618、Pangu 649（原文位势单位 m²/s²） | 模型数值优势需按同双周均值目标理解 |
| Table 1，周 3–4 T2m RMSE | **2.007 K** | GraphCast 2.158、ClimaX 2.998 | 仅对已报告此字段的基线比较 |
| Table 1，周 5–6 Z500 RMSE | **471** | ClimaX 619、Pangu 754；GraphCast 此格未报 | 缺项不能填成败给/胜过 GraphCast |
| Table 5，周 3–4 自回归消融 Z500 RMSE | 直接法 **477** | 自回归 **781** | 说明目标设计很重要，不能把全部增益归于环形 patch |
| Table 5，周 5–6 T2m RMSE | 直接法 **2.026** | 自回归 **5.047** | 日均/双周均目标越远，滚动法误差级联越突出 |
| Table 7，欧洲周 3–4 Z500 RMSE | **651** | ClimaX 855、GraphCast 905 | 欧洲区域优势来自原文条件；仍需独立年份验证 |

这些数值直接摘自[原文 Tables 1、5、7](https://arxiv.org/html/2502.19750)。原文对环形 patch 与频域混合做消融，证明二者贡献，但与“直接预测 vs 自回归”的任务重设相比，不能简单用总成绩估算每个结构改动的净贡献。月度/纬度分析还指出高纬区域的几何处理可能更有价值。[Results/Ablation](https://arxiv.org/html/2502.19750)

| Table 2：第 3–4 周结构消融 | Z500 RMSE（m²/s²） | T850 RMSE（K） | T2m RMSE（K） | 解释 |
|---|---:|---:|---:|---|
| 平面网格 patch、无 DFT | 516 | 2.168 | 2.554 | 两个设计均去除 |
| 环形 patch、无 DFT | 502 | 2.077 | **2.000** | T2m 略好于完整模型，故并非所有指标完整模型最优 |
| 平面网格 patch、有 DFT | 497 | 2.050 | 2.583 | 仅用频域模块不能替代球面 patch |
| 环形 patch、有 DFT（CirT） | **477** | **1.903** | 2.007 | Z500、T850 最好 |

这比笼统说“二者均改善一切指标”更准确：Z500 呈现叠加收益，但 T2m 的 FFT 增益并不成立。由于 Fourier 发生在压缩后的潜维，消融可以支持“频域混合有益于若干变量”，不足以直接证明原始经度波数的物理可解释性。原文附表/图还比较南北中高纬，提示球面归纳偏置的效益随纬带变化。[Table 2–3](https://arxiv.org/html/2502.19750v1)

Table 2 的第 5–6 周 z500 四格依次为 `501 / 498 / 494 / 471`，继续支持环形 patch 与频域模块组合有效；但同窗 t2m 为 `2.578 / 2.178 / 2.650 / 2.026 K`，可以看出普通网格加 FFT 反而变差。Table 3 也有值得保留的**反例**：低纬第 3–4 周 z500，FourCastNetV2 为 `200`，CirT 为 `206 m²/s²`，CirT 并非所有纬带所有变量都占优。作者所说“generally”不等于“every case”。[正式论文 Tables 2–3](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

### CirT 本身的微调实验：正式论文附录 Table 6

微调**确实存在**，但只作为附录对照，不是主模型的训练流程。作者先训练一个**自回归 CirT**，再冻结它的 Transformer encoder，替换为重新初始化的输入 embedding 和输出 head，使其对两个双周窗直接给出预测。表中一组更新 embedding 与 decoder，另一组只更新 decoder；“decoder”在此应理解为输出头，而非另一个完整 Transformer 解码器。论文没有交代自回归预训练的完整优化日程、微调时的学习率/epoch/样本量，也没有提供可直接核验该消融的独立配置。因此不能把上节代码默认 AdamW/20 epoch 自动归给微调实验。[正式论文附录 A.2、Table 6](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

| 目标窗 / 训练方式 | z500 | z850 | t500 K | t850 K | t2m K | u10 m/s | v10 m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 周 3–4，embedding + decoder 微调 | 480 | 315 | **1.660** | **1.870** | **1.983** | 1.842 | 1.530 |
| 周 3–4，仅 decoder 微调 | 540 | 346 | 1.885 | 2.327 | 2.715 | 2.013 | 1.619 |
| 周 3–4，直接训练 | **477** | **304** | 1.687 | 1.903 | 2.007 | **1.806** | **1.511** |
| 周 5–6，embedding + decoder 微调 | 485 | 312 | 1.679 | **1.923** | 2.032 | 1.847 | 1.535 |
| 周 5–6，仅 decoder 微调 | 588 | 354 | 2.190 | 2.702 | 3.145 | 2.043 | 1.650 |
| 周 5–6，直接训练 | **471** | **301** | **1.672** | 1.933 | **2.026** | **1.809** | **1.512** |

直接训练在多数变量更好，但 embedding+head 微调在周 3–4 的三个温度量和周 5–6 的 t850 略好；“不需要微调”不能概括为“微调从未做过”或“微调对任何量均无益”。[正式论文 Table 6](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

### 数值模式、区域结果和原图信息

数值系统的来源不是四个同配置神经网络：UKMO GloSea6 为每日 60 天 control，NCEP CFSv2 为每日 45 天 control，CMA BCC-CSM2-HR 为周一/周四 60 天 control，ECMWF IFS CY48R1 为周一/周四 46 天 control。论文评测 ERA5 指向 2018 年，但未清晰交代这些数值模式归档预报的统一初值年份、版本/回报重预报与配对样本；尤其 CY48R1 版本与 2018 年测试年份的关系需独立澄清。故文中“超过业务系统”是**作者在其收集样本上的报告**，尚不足以解读为严格同年份、同初值、同输入与后处理的操作性竞赛。[正式论文附录 A.1.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

Table 7 的第 3–4 周欧洲 z500 为 CirT `651`、ClimaX `855`、GraphCast `905 m²/s²`，北美相应为 `801/879/1014`；Table 8 的 z500 多尺度结构相似度（MS-SSIM）在周 3–4 为 CirT `0.909`、GraphCast `0.872`，周 5–6 CirT 仍为 `0.909`、GraphCast 未报。这与全球 RMSE 互为补充，却不等同于极端事件或概率校准能力。[正式论文 Tables 7–8](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

原图阅读索引：Figure 1 是球面和平面 patch 几何，Figure 2 是本文上方 Mermaid 结构图的依据；Figure 3 比较不同气压层 RMSE，但曲线经过图内 min–max 归一，不应把纵轴当绝对误差；Figure 4 / 附录 Figures 7–13 是不同变量和预报窗的全球 RMSE 地图，揭示高纬与北美等误差热点；Figure 5 / 附录 Figures 14–16 是按月误差，属于单个 2018 年测试段的季节切片；Figure 6 为数值模式与 CirT 的 ACC 层次对比。未从曲线或色块反推“精确数值”，可核实的数字以上列表格为准。[正式论文全文与图注](https://proceedings.iclr.cc/paper_files/paper/2025/file/fc5a1845bee1f5405ef99ba25c2d44e1-Paper-Conference.pdf)

## 3. 阅读判断与复现

CirT 提供一个参数较小、明确针对球面网格的 S2S 直接预报器；它不属于多任务天气基础模型，也不给出完整逐日概率集合。模型结果适合作为第 3–6 周**双周平均**基线，不宜与 FuXi-S2S/TianXing-S2S 的逐日集合 CRPS 在同一列排名。复现应核对 ERA5 年份划分、异常气候态、位势与高度单位、双周平均的对齐方式，并与相同任务重新训练的平面 ViT 及持久性/气候态基线做配对评测。[原文 Methods/Experiments](https://arxiv.org/html/2502.19750)

[返回首页](../../README.md) · [返回总表](../../气象大模型_中期预报论文追踪.md)
