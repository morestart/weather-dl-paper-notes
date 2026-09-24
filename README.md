# Weather DL Paper Notes / 气象深度学习文献阅读笔记

2024–2026 年气象深度学习文献的中文阅读笔记，特别关注中期（约 10–15 天）与次季节预报。每篇论文保留独立文件，记录可核验的研究问题、方法、数据、指标、表格结果、局限和复现问题。

## 目录与分类

每篇完成稿只放一个目录，按论文的**主要研究问题和实际评测时效**分类：`notes/nowcasting/` 收录短临及以小时级任务为主的通用模型；`notes/medium-range/` 收录日尺度预报，重点约 8–15 天，少数 6–7 天的过渡方法在正文注明；`notes/s2s/` 为第 2 周之后至季节前段；`notes/assimilation/` 收录以观测同化为主要贡献的论文，即使也报告中期技巧。跨领域论文只按主问题归档，文内注明其他时效。没有笼统的 `other`。

### [中期预报](notes/medium-range/)

- [Aurora：异构地球系统预训练与中期天气](notes/medium-range/01-aurora.md) — 结构重绘、跨任务性能口径表与 10 天时效纠偏。
- [FengWu：长滚动、多模态与扩散集合](notes/medium-range/02-fengwu.md) — 结构重绘、10 天以上技巧证据表和正文/图注口径辨析。
- [FuXi Weather：真实卫星观测到 10 天全球预报](notes/medium-range/08-fuxi-weather.md) — 循环同化结构、Z500 技巧时效与逐变量性能表。
- [Aardvark Weather：观测直达格点和站点](notes/medium-range/09-aardvark-weather.md) — 三模块结构、全球与分区性能表及不利结果。
- [Weather Prediction with Diffusion：引导式扩散预报](notes/medium-range/34-diffusion-guided.md) — 结构重绘、9–14 天证据表与引导模式边界。
- [SOFT：7 天自输出微调](notes/medium-range/31-soft.md) — 中期前段过渡方法，不能误称 10–15 天验证。
- [CoDiCast：6 天条件扩散](notes/medium-range/35-codicast.md) — 中期前段过渡方法，不能误称 10–15 天验证。
- [AIFS：ECMWF 确定性底座](notes/medium-range/41-aifs-base.md) — 2024 研究版的字段、三阶段训练/微调及 10 天评分。
- [AIFS-CRPS / AIFS ENS v1：近公平 CRPS 集合](notes/medium-range/37-aifs-crps.md) — 逐字段归一化、四阶段优化、15 天集合、46 天外推与业务版别边界。
- [hy-IFS-ENS：IFS–AIFS 逐成员谱约束混合集合](notes/medium-range/48-hybrid-ifs-aifs-ens.md) — 2026 独立方法论文；预训练/微调、T21/12h 约束、15 天评分及图 1–9 的负面结果。
- [EPT-2 / EPT-2e：动态 lead 与能源变量集合](notes/medium-range/38-ept-2.md) — Jua 欧洲企业系列，公开主评测到 240h。
- [EPT-1.5：欧洲能源场景与 10 天格点/站点评测](notes/medium-range/45-ept-1-5.md) — 业务规格 20 天与实际公开验证 10 天分开。
- [FuXi-ENS：15 天流依赖集合](notes/medium-range/39-fuxi-ens.md) — 与确定性 FuXi 和 FuXi Weather 分篇。
- [AIFS-DOP：观测直达中期](notes/medium-range/42-aifs-dop.md) — 逐仪器数据期、缺测处理、三天 rollout 微调及 10 天验证。
- [GenCast：15 天条件扩散集合](notes/medium-range/43-gencast.md) — 1° 预训练至 0.25° 微调和概率技巧分开核查。
- [NeuralGCM：可微动力与神经参数化](notes/medium-range/44-neuralgcm.md) — 天气评分与多年气候模拟分别阅读。
- [AIFS Single 1.1：业务训练更新和边界层](notes/medium-range/46-aifs-single-1-1.md) — 公开 ERA5 预训练、IFS rollout 微调、降水消融。
- [GraphDOP：卫星/常规观测直达预报](notes/medium-range/47-graphdop.md) — 主证据约 5 天，不能借后续 AIFS-DOP 的 10 天技巧。
- [10–15 天高温预报的 AI 模型评测](notes/medium-range/17-heat-emulators.md) — 六类系统的回报样本、阈值口径和可比性逐项核对；本研究不训练新模型。
- [ATLAS：概率中期预报框架](notes/medium-range/18-atlas.md) — 拆开模型变体、数据和训练日程。
- [Searth Transformer / YanTian](notes/medium-range/24-searth.md) — 10.3 天时效与预训练/微调细节。
- [Nested-EAGLE：全球–区域嵌套预报](notes/medium-range/26-nested-eagle.md) — 数据、归一化、分阶段训练。
- [WeatherNext 3：观测增强概率预报](notes/medium-range/28-weathernext3.md) — 多阶段训练和 15 天集合。
- [ERDM：扩散式概率天气预报](notes/medium-range/29-erdm.md) — 数据、网络、优化步骤和评分。

### [次季节预报（S2S）](notes/s2s/)

- [TianQuan-S2S：气候态融合与逐层噪声](notes/s2s/04-tianquan-s2s.md) — 模型结构、概率/确定性性能表及 Wind10 的负面结果。
- [Xaurora：谱一致性 S2S 微调](notes/s2s/05-xaurora-egu-abstract.md) — EGU 摘要级解读；不填造性能数字。
- [ESFM S2S 策略：多尾、LoRA 与慢变量注意力](notes/s2s/06-esfm-s2s-egu-abstract.md) — EGU 摘要及海报解读，记录 38 天/96 成员实例。
- [基础模型+MSWEP 的 S2S 降水](notes/s2s/07-s2s-precip-egu-abstract.md) — EGU 摘要级解读，厘清九年比较计划与未披露的结果。
- [PBC：S2S 概率偏差订正](notes/s2s/10-pbc-subseasonal.md) — 按 v3 新题名精读，含双分支概率校准结构图和竞赛口径。
- [FuXi-S2S：42 天日均集合](notes/s2s/40-fuxi-s2s.md) — 2024 正式论文；76 通道、课程训练参数与 MJO 评测口径。
- [TianXing-S2S：45 天扩散式次季节集合](notes/s2s/21-tianxing-s2s.md) — 逐项整理数据、训练与周平均评分。
- [AI 模型类比预测：周 3–4 可解释匹配](notes/s2s/12-ai-model-analogs.md) — 类比库、遮罩优化与训练/验证划分。
- [Swift：一致性模型集合 S2S 预报](notes/s2s/14-swift.md) — 完整训练日程、多步微调和概率评分。
- [Marchuk：潜空间扩散集合](notes/s2s/15-marchuk.md) — 数据时序、VHT 训练与 LoRA 架构辨析。
- [TelePiT：潜空间物理先验与遥相关注意力](notes/s2s/20-telepit.md) — 数值消融、完整训练参数及原文冲突审计。
- [OmniCast：跨中期和 S2S 的潜空间生成](notes/s2s/22-omnicast.md) — 两套数据协议、VAE/生成器训练及图表解读。
- [ReST：站点温度的次季节订正](notes/s2s/11-rest.md) — GEFS/站点资料处理、训练目标与六次独立重复。
- [基础模型 S2S 微调基准：EGU25 摘要](notes/s2s/13-foundation-finetuning-egu.md) — 摘要级证据，列明未公开的训练/微调参数与结果。
- [GAN-W2C：降水集合的天气到气候尺度后处理](notes/s2s/16-gan-w2c.md) — 多源网格对齐、生成成员设置与训练资料缺口。
- [CirT：环形纬圈与频域注意力](notes/s2s/23-cirt.md) — 63 通道、训练超参数、频域操作轴及结构消融。
- [西美国 3D U-Net：S2S 降水后处理](notes/s2s/25-western-us-3d-unet.md) — ECMWF 输入、区域资料、训练与消融设置。
- [天气—气候桥梁 Perspective](notes/s2s/27-weather-climate-bridge-perspective.md) — 观点文章，分清概念框架与不存在的自有模型成绩。
- [欧洲天气型的第 3 周技巧窗口](notes/s2s/30-weather-regime-windows.md) — 前兆特征、神经网络变体与交叉验证信息泄漏风险。
- [SFNO-HENS/NeuralGCM：MJO 遥相关评估](notes/s2s/19-mjo-teleconnections-evaluation.md) — 公开图注逐图解读；全文方法尚不可读，明确标为证据受限稿。

### [短临/小时级与通用模型](notes/nowcasting/)

- [WeatherGFM：视觉提示统一气象任务](notes/nowcasting/32-weathergfm.md) — 主实验含 SEVIR 小时级任务；ERA5 附录最长 7 天。
- [降水临近预报综述](notes/nowcasting/36-nowcasting-survey.md) — 临近预报分类与指标框架。

### [数据同化](notes/assimilation/)

- [XiChen：4DVar 梯度驱动观测同化](notes/assimilation/03-xichen.md) — 观测—分析—预报链重绘，分开呈现不同初值条件的性能。
- [DiffDA：扩散式天气尺度同化](notes/assimilation/33-diffda.md) — 稀疏观测与背景场融合，分开讨论分析和后续预报。

本次公开批次为 **48 篇独立笔记**，其中 **47 篇按可取得原文完成逐篇核验**，会议摘要与 Perspective 按来源等级解读，不冒充完整实验论文；另 **1 篇 #19 仅取得期刊摘要、开放前页与出版方图注**，虽已整理并上传，仍需要全文方法和补充材料才能称为完整精读。2024–2026 全年回溯也未穷尽。因此本仓库仍不能称为“全部完成”。[论文追踪表](气象大模型_中期预报论文追踪.md)保留全部 48 条书目与版本历史。

## 阅读笔记约定

每份详细笔记至少包括：论文与版本、核心问题、模型机制、训练与评测设置、带出处的具体结果、对照实验、证据局限、与中期/S2S 的关系和下一步复现检查。作者报告与阅读判断会明确区分。只有摘要或会议摘要可用时会标注资料不足，不填充推测细节。

## 来源与版权

此仓库只保存原创中文解读、依据论文重绘并标明出处的示意图、经核对的选摘数值表，以及指向原文的链接；不转载论文全文或未经授权的原图。论文与代码各归原作者及其授权方所有。不同数据集、阈值、预报时效或评测协议的数据不合并排名；缺少具体数字的图形不反推伪精确数值。
