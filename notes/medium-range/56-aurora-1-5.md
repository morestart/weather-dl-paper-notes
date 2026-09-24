# Aurora 1.5：基础模型如何经过六段微调变成中期概率集合预报系统

> Jonathan A. Weyn、Zekun Ni、Amit Misra、Will Fein、Haiyu Dong、Wessel P. Bruinsma、Richard E. Turner、Matt Corey、Kit Thambiratnam、Kevin White、Kenji Takeda、Hongyu Sun，*Aurora 1.5: Fine-Tuning a Foundation Model for Medium-Range Ensemble Weather Prediction*，微软研究院[官方 15 页预印本 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)，稿内日期 **2026-07-09**；[发布页](https://www.microsoft.com/en-us/research/publication/aurora-1-5-fine-tuning-a-foundation-model-for-medium-range-ensemble-weather-prediction/)。阅读日期：2026-09-24。**证据级别：作者机构公开预印本，不冒充已经同行评审的期刊论文。**发布页作者栏缺 Hongyu Sun，PDF 首页列有 12 人，本笔记按 PDF 首页记录。

## 1. 研究问题、模型家族与实际预报时效

初代 [Aurora](01-aurora.md) 是在异构气象/地球系统资料上预训练的 3D Swin-Transformer 基础模型，已有确定性中期天气能力。Aurora 1.5 问的是：能否从其权重出发，用有限的后训练获得更多近地气象量、原生小时级输出和具有模型扰动的概率集合，而不从零训练新基础模型？作者提供两个**不同检查点**：Stage 1b 后的确定性 `Aurora 1.5`，Stage 2b 后的随机 `Aurora 1.5 ENS`。论文主评分是 **1–10 天**，并非 15 天/S2S；虽然代码可继续滚动，本文不证明更长时效的业务技巧。[官方论文 §1–3、§6](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

```mermaid
flowchart LR
  A[初代 Aurora 预训练权重\nPerceiver 式变量编码] --> B[Stage 1: 新单层量 + 随机 0-12h lead\nERA5, MAE]
  B --> C[1a: ERA5 两步自回归 MAE]
  C --> D[1b: IFS HRES 分析两步微调]
  D --> E[确定性 Aurora 1.5]
  E --> F[Stage 2: AdaLN 高斯噪声\n双成员 fair CRPS, ERA5]
  F --> G[2a: ERA5 四步随机滚动\nstop-gradient]
  G --> H[2b: IFS HRES 分析随机微调]
  H --> I[Aurora 1.5 ENS\n初值扰动 + 模型扰动]
```

结构是据原文 §2/§6 重绘的训练谱系，不是原图复制。基础模型本体保持 encoder–processor–decoder：变量嵌入与 Perceiver 式编码器把不同字段投向共用潜空间，3D Swin Transformer 处理时空潜表示，解码器投回原生经纬网格；Fourier lead-time 嵌入使目标时间可变。新增单层量主要增加输入/输出变量参数和权重重用，随机性则由 backbone AdaLN 模块中的噪声嵌入产生，并不是每个集合成员分别训练一套模型。[§2](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

## 2. 数据字段、预处理及边界条件

| 数据与用途 | 原文可核验设置 | 复现时的重要区别 |
|---|---|---|
| Stage 1、1a、2、2a | **0.25° 规则经纬网 ERA5**；Stage 1 训练 **1981–2023**、验证 **1979–1980**；其余 ERA5 阶段沿该字段/网格配置 | 这是从 Aurora 已训练权重继续微调，不是从零预训练；验证年在训练期之前，不可误写为 2024 验证。原文没有明确给出所有后续 ERA5 阶段逐年的切分细节。[§2、§7.1](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf) |
| Stage 1b、2b | **2018–2023 ECMWF 业务 HRES 分析**，字段配置尽量与 ERA5 阶段一致 | 目的是适配实时业务分析初值分布；这两段是论文明确的 operational-analysis fine-tune，不可与 ERA5 主微调混称一种训练资料。[§2、§7.1](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf) |
| 高空变量 | 继承 Aurora 的多气压层大气变量；主概率 scorecard 显示位势、温度、比湿在 850/700/500/200 hPa 的比较 | scorecard 的 88.9% 只针对选定的高空与五个近地变量 × lead time，**不是所有新输出变量或所有垂直层都胜出**。[图 2](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf) |
| 单层变量 | 原有 `2t,10u,10v,msl`；新输入/输出包括 `2d,tcwv,tcc,100u,100v,sp,lcc,mcc,hcc,skt,stl1,swvl1,siconc,sd`；仅解码输出的七项为 `i10fg,blh,uvb,ssrd,ttr,tp,sf` | 附录表 3 共列 **25 个单层输出量**。`uvb/ssrd/ttr` 的辐射及 `tp/sf` 的降水量，目标是**验证时刻之前 1 小时累计量**；不能把它们当作瞬时场，也不能把仅输出字段反馈给编码器。[附录 A 表 3](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf) |
| 静态/外部输入 | 扩充地形方向和坡度、植被覆盖、土壤/植被类型；另以位置和时间预计算大气顶入射太阳辐射加入地表输入 | 太阳辐射是**输入而非预测目标**，每个自回归步重新计算；论文未做其增益消融，不能把长期稳定性改进归因于该字段。[§6.2、§7.1](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf) |

新变量使用对应缩放参数，训练损失沿 Aurora 的逐变量权重叠加至所有变量、垂直层与格点；论文没有公布每个新字段的全部数值缩放/标准化系数，也没有在文中详列所有高空字段的逐通道预处理。因此这里明确区分**可核验的字段/时间累计口径**与需要查代码/检查点才可复验的完整归一化配置。[§2、§6.4、附录 A](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

为了滚动稳定，作者只在**把预测作为下一步输入之前**裁剪若干字段：总柱水汽、第一层土壤水分不得小于 0；总/低/中/高云量和海冰覆盖约束在 `[0,1]`；雪深约束 `[0,10 m]`。该操作不直接改动已输出给用户的预报，也不作用于仅输出字段。Fourier lead-time 嵌入的最短“波长”由 **1 分钟**改成 **6 小时**，避免 1 小时训练间隔在高频正弦/余弦特征中产生不合适的周期性。[§6.2](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

## 3. 六段微调：目标、优化参数与产物

原文称“三阶段路线”，但表 2 实际拆出 **1、1a、1b、2、2a、2b 六段**。下表据论文原[表 2](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)逐项重排；GPU 均为 **NVIDIA A100 80GB**。`steps` 是论文表里的样本/更新量级，不是“有效独立年份数”。

| 阶段 | 资料/损失/输出 | 更新量；GPU；耗时 | LR 与其它训练设置 |
|---|---|---|---|
| 1 | ERA5；MAE；扩单层变量并训练 0–12h 可变 lead（1h 间隔，6/12h 抽到的概率更高） | 3.76M；32；约 2 周 | 主 LR `5e-5`、余弦退火；新增地表编码/解码权重、时间嵌入、层归一化权重特殊 LR `2e-4`；drop-path `0.1`；单步/单成员。 |
| 1a | ERA5；MAE；**两步**自回归，梯度穿过两步 | 376K；8；约 3 天 | LR `3e-6` 固定；drop-path `0.1`；不再随机 lead。 |
| 1b | 2018–2023 HRES 分析；MAE；两步业务分析微调，产出**确定性 Aurora 1.5** | 26K；8；约 18 小时 | LR `3e-6` 固定；drop-path `0`；无噪声。 |
| 2 | ERA5；**双成员 fair CRPS**；开启 AdaLN 噪声与 0–12h 可变 lead | 1.9M；32；约 10.5 天 | 主 LR `5e-5`、余弦；噪声嵌入 MLP/LayerNorm 特殊 LR `2e-4`；drop-path `0`；两次随机前向。 |
| 2a | ERA5；双成员 CRPS；**四步/24h**随机滚动 | 128K；32；约 3 天 | LR `5e-6` 固定；因内存限制用 **stop-gradient**，不把梯度穿过整个四步链；不再随机 lead。 |
| 2b | 2018–2023 HRES 分析；双成员 CRPS；四步业务分析微调，产出 **Aurora 1.5 ENS** | 26K；24；约 1 天 | LR `3e-6` 固定；drop-path `0`；继续噪声注入。 |

上述各阶段均从上一个已训练检查点继续，而不是彼此独立的六次从零训练。Stage 1/2 的新参数有更高 LR，Stage 1a/1b/2a/2b 是明确的 rollout/业务分析再微调。原文没有列出每阶段的批量大小、Adam/AdamW 的 β、权重衰减、全量梯度裁剪等细节；不得照搬旧 Aurora 的参数猜填。确定性基线比较时，作者还把原 Aurora 0.25° 推荐检查点加做两步自回归专门化，且**禁用 LoRA**，并非拿完全原始预训练权重作唯一基线。[§2、§3.1 脚注、§6.1 表 2](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

### 概率目标与模型扰动到底如何产生

Stage 2 在 AdaLN 相关路径加入独立高斯噪声，让同一网络和同一初值的不同前向运行得到不同成员。每个训练更新仅采 **2 个成员**，用 fair CRPS：

`[|x₁−y|+|x₂−y|]/2 − |x₁−x₂|/2`。

第一项惩罚对真值的误差，第二项校正集合成员间的离散度；与只最小化成员均方误差不同，这允许网络学习概率分布的合理 spread。推理时可用同一个初值只靠模型噪声生成集合，也可同时使用 **ECMWF ENS 初值扰动 + 独立模型噪声**。常规滚动的主步长仍是 **6h**；若输出 1h 细分预报，相邻细分步通过 FIFO 最近噪声的归一化和保持连贯，满一个 6h 主步后更换掉旧噪声，使主步间有效扰动重新独立。注意“1h 原生输出”不等于所有业务评测都以 1h 积分。[§6.3–6.4](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

## 4. 图表与性能：不同测试协议绝不可混排

### 4.1 全球中期主评测（图 1–3、图 7）

测试年为 **2024**，每周一/周四 00 UTC 起报，共 **103 次**。确定性图 1 比 Aurora 1.5、专门化原 Aurora 与 IFS HRES，对业务分析场计算 RMSE/MAE；它展示 Z500、T850、2m 温度、2m 露点和总云量等，不能把某几个示例曲线解读为全变量全区域制胜。作者称随机 lead 训练改善长滚动稳定性，但该消融图**未展示**，所以只可记为作者描述。[§3.1、图 1](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

概率图 2 使用同一 103 次起报，但 **Aurora ENS 的两张输入时次为 ECMWF ENS 0h 和 6h 扰动成员**；双方各 **50 成员**。作者报告在选定高空位势/温度/比湿及五个近地量的变量 × 预报步长目标中，Aurora 1.5 ENS 的 CRPS 优于 ECMWF ENS 的比例为 **88.9%**。这不是“比 ENS 降低 88.9% 误差”，也不是 88.9% 的所有天气变量。图 2 蓝色越深是相对 CRPS 越低；200 hPa 位势在多数时效偏红，第 10 天 2m 温度也略差，温/湿/部分近地变量在最初约 18h 可暂时偏差。可能与 ENS 初值输入分布有关，而非论文证明的唯一原因。[§3.2、图 2](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

图 3 给 2026-06-24 18 UTC 起报、48–72h 平均的总云量及向下太阳辐射集合均值/标准差，仅作空间形态展示；并非图 2 同样的 2024 统计样本。图 7 的第 3–7 天秩直方图显示，相比动力 ENS 常见的低离散度，Aurora 1.5 ENS 通常因 CRPS 优化拉大 spread，且有**过度离散**趋势。这种校准缺陷不能被较好 CRPS 一笔带过。[§3.2、图 3/7](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

### 4.2 热带气旋与极端温度（图 4–6、图 8、表 1）

极端事件另用 **2024–2025 四次/日起报**，Aurora ENS 改为 **32 成员，全部取相同 IFS control T0 初值**，仅靠模型噪声分叉。因与全球主 scorecard 的 50 个 ENS 扰动初值设置不同，不能把两个试验的概率成绩直接并排排名。[§4](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

气旋路径以 IBTrACS v04r01 为真值，在匹配的风暴—起报样本上比较第 1–5 天（30/54/78/102/126h）的球面距离。确定性 Aurora 1.5 相对原 Aurora 的路径误差减少 **9–24%**，集合**中位路径**改善 **13–34%**，且中位比集合均值更好。论文用按盆地分层的 **1000 次风暴 ID bootstrap** 判断差异显著性，并对日期变更线做圆周经度平均；远离成员中位数超过 15° 的路径被标记为离群。初始缺少 IC 扰动使气旋路径秩图在第 1 天偏欠离散，到第 3 天趋于更均衡，和全球格点第 3–7 天的“可能过度离散”不可混为一谈。[§4.1、§7.2、图 4/8](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

Helene 的单次示例 2024-09-24 00 UTC：NHC 官方预报、确定性 Aurora 1.5、Aurora 1.5 ENS 均值路径的**个例平均误差**分别为 **110.6 / 58.4 / 26.3 km**；集合路径约快 6h。它不能替代全年、多盆地统计，也不能凭此断言 Aurora 业务全面超过 NHC。[图 5](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

极端温度以 ERA5 的日最高/最低 2m 温度验证 2024–2025 的 **126/132/138/144h（约 5–6 天）**，阈值是每格点 **1991–2020** ERA5 年度气候态分位。下表是原文[表 1](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)的中文选摘，误差/CRPS 单位均为 K：

| 事件类 | 样本数 | 初代 Aurora 确定性 MAE | Aurora 1.5 ENS 中位 MAE | Aurora 1.5 ENS CRPS（95% CI） |
|---|---:|---:|---:|---:|
| 日最高温 > p90 | 7,661,227 | 1.703 | 1.000 | 0.733 [0.730, 0.738] |
| 日最高温 > p95 | 4,592,688 | 1.795 | 1.034 | 0.761 [0.756, 0.764] |
| 日最低温 < p10 | 2,862,642 | 2.229 | 2.226 | 1.675 [1.668, 1.684] |
| 日最低温 < p05 | 1,221,630 | 2.379 | **2.487（退步）** | 1.890 [1.882, 1.898] |

图 6 的阈值化 RMSE 更直接表明热尾 `tmax` 改进明显，异常冷的 `tmin` 则比较困难；对于最冷 p05，ENS **中位数 MAE 比原 Aurora 更差**，但全分布 CRPS 仍比初代确定性 MAE 小。确定性模型的 CRPS 数学上等于 MAE，不过把**概率集合 CRPS**和**确定性 MAE**当作完全公平的同类业务模型比较，作者自己也明确不赞成；应优先看与 ECMWF ENS 同成员数的图 2 或分开报告。[§4.2、图 6、表 1](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

## 5. 关键判断与复现缺口

这篇论文确实回答了“基础模型如何微调成中期集合”：少量变量参数扩展、可变 lead、业务分析适配；然后引入 AdaLN 噪声和双成员 fair CRPS，再以四步自回归及业务分析完成随机微调。其可贵之处是清楚地揭示**确定性版和集合版不是相同训练目标**，同时在 10 天中期范围内用相同 50 成员对 ECMWF ENS 评分。性能并非全面占优：200 hPa 位势、第 10 天 2m 温度、早期若干时效与冷极端点预测是明显例外；全球格点校准和气旋路径校准亦有不同偏差。[官方预印本 §2–5](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

复验清单：确认 PDF 首页作者与版本；取初代 0.25° Aurora 专门化 checkpoint，禁用 LoRA；核对 ERA5/HRES 训练年份及 25 个单层字段的输入/仅输出分组；按表 2 重跑六阶段并确认变量缩放/损失权重在代码中一致；只在递归输入时做物理裁剪；主图用 2024 年 103 起报和 **50× ENS 初值**，极端图用 2024–2025 年四次/日和 **32×同一 control 初值**；分开验全球 CRPS、rank histogram、TC 路径与热/冷尾，并保留不利格点/变量。模型[代码和检查点](https://github.com/microsoft/aurora)公开，但 HRES 分析与 ENS 全量档案需遵循 ECMWF 数据授权；论文没有公开完整逐字段缩放表，也未做与扩散随机化方案的系统对比。[§5–7](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/07/Aurora_1_5_Paper.pdf)

[返回首页](../../README.md) · [返回总表](../../气象大模型_中期预报论文追踪.md)
