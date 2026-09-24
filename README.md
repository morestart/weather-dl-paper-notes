# Weather DL Paper Notes / 气象深度学习文献阅读笔记

2024–2026 年气象深度学习文献的中文阅读笔记，特别关注中期（约 10–15 天）与次季节预报。每篇论文保留独立文件，记录可核验的研究问题、方法、数据、指标、表格结果、局限和复现问题。

## 目录与分类

每篇完成稿只放一个目录，按论文的**主要研究问题和实际评测时效**分类：`notes/nowcasting/` 收录短临、小时级及约 3–4 天内的短期预报与评测；`notes/medium-range/` 收录日尺度预报，重点约 8–15 天，少数 5–7 天的前段/过渡方法在正文注明；`notes/s2s/` 为第 2 周之后至季节前段；`notes/assimilation/` 收录以观测同化为主要贡献的论文，即使也报告中期技巧。跨领域论文只按主问题归档，文内注明其他时效。没有笼统的 `other`。

### [中期预报](notes/medium-range/)

- [Weighted Potential CRPS：AI 与 HRES 的第十天极端信息量](notes/medium-range/84-weighted-pcrps-extremes.md) — 2026 arXiv v1 独立评测；2020 年 702 次起报、1/3/5/7/10 天与 1979–2019 ERA5 极端门槛，完整解读 EasyUQ 样本内最优拟合、表 1 月纪录次数、图 1–5/附图 13–20，并区分“潜在排序信息”与原始纪录误差/真实集合技巧；指出正文样本对数量的乘法冲突及第十天对季节气候态的负结果。
- [破纪录极端天气：AI 与 HRES 的十天比较](notes/medium-range/83-record-breaking-extremes-zhang-2026.md) — 2026 *Science Advances* 正式独立评测；逐地逐月 1979–2017 历史纪录、2020 三类样本数、AI/HRES 不同真值和 2018/2020 GraphCast 训练期差异，按图 1–4 拆解平均优势与纪录劣势，指出业务 GraphCast 微调期覆盖测试年的版别边界。
- [GraphCast 球谐 AMSE：双重惩罚、五段微调与第十天谱技巧](notes/medium-range/82-amse-graphcast-spectral-loss.md) — 2025 ICML 独立损失方法；2016–2021 HRES-fc0/ERA5 降水配对、32,500 batch 五段微调、图 1–14 与表 2 全数值，区分 1250→160 km 的谱阈值、7.5 天 CRPS 和仅图示的十天谱诊断，保留降水与 MAE 对照的负面结果。
- [第十天极端温度与大风：GraphCast、Pangu 与 IFS 的全球分区比较](notes/medium-range/81-olivetti-messori-extremes-2024.md) — 2024 GMD 正式独立评测；2020 年 702 次半业务起报、1/3/5/7/10 天、ERA5 1.5° 的三种尾部定义和图 1–12/附录 A–D，区分 GraphCast 平均优势、长 lead 尾部低估及 FuXi 的另套附录口径。
- [GraphDOP 跨圈层案例：十天海冰亮温、飓风冷尾流与欧洲热浪](notes/medium-range/77-graphdop-coupled-earth-system.md) — 2025 独立 ECMWF 研究；2000–2020 多源观测、4×3h 潜推进/多窗微调、图 1–7 的第十天海冰案例与飓风/热浪负例，区分案例示范和未给出的全球统计评分。
- [Aurora：异构地球系统预训练与中期天气](notes/medium-range/01-aurora.md) — 结构重绘、跨任务性能口径表与 10 天时效纠偏。
- [Aurora 1.5：基础模型六段微调为中期集合](notes/medium-range/56-aurora-1-5.md) — 25 个单层字段、ERA5/IFS 六段训练参数、50/32 成员两套评测、极端温度负例与可靠性边界。
- [Prithvi-Precip：卫星观测增强全球降水预报](notes/medium-range/57-prithvi-precip.md) — 仅到 96h 的前段方法；逐项说明异构卫星编码、数据处理、训练/二次微调、图 5–13 与独立雷达/雨量站负例。
- [Prithvi WxC：掩码/天气联合预训练基础模型](notes/medium-range/58-prithvi-wxc.md) — 2024 原始底座与降水后续分篇；160 字段、两阶段训练/完整公开参数、降尺度与重力波微调、表 1 和图 1–11 的证据边界。
- [ArchesWeather / ArchesWeatherGen：均值加流匹配残差集合](notes/medium-range/59-archesweather-gen.md) — 82 场 ERA5、CLA 结构、确定性/生成五段训练、表 1–2 与图 1–17；区分 15 天生成能力和 10 天定量评分。
- [SPW：冻结确定性模型的推理期权重扰动集合](notes/medium-range/60-spw-deterministic-ensembles.md) — 2026 年 57 页完整论文；四种底座各自 σ/扰动组、112 次 10 成员回报、表 1–12/图 1–17 及格点与区域均值校准反例。
- [WeatherNext 2 / FGN：边际 CRPS 训练的 15 天概率集合](notes/medium-range/61-fgn-weathernext2.md) — 四阶段 ERA5→HRES 微调、32 维共享噪声、56 成员评测、原文图 1–5/补图的正负证据与 ENS 版别边界。
- [FourCastNet 3：球面谱概率集合](notes/medium-range/62-fourcastnet3.md) — 72 字段 ERA5、三段训练与真实微调参数、八通道球面随机过程、50 成员 15 天图表及 60 天稳定性边界。
- [WeatherNext Cyclones：15 天全球台风概率集合](notes/medium-range/63-weathernext-cyclones.md) — *Nature* 正式论文；ERA5/HRES-fc0/IBTrACS 数据与缺测处理、六阶段训练/气旋头冻结微调、图 1–5/补图性能、2025 线上版别与强度校准负例。
- [FuXi-TC：FuXi-2.0/WRF 教师驱动的扩散式台风细化](notes/medium-range/64-fuxi-tc.md) — 独立的 5 天区域前段研究；WRF/ERA5/FuXi 数据链、冻结底座与 U-Net 训练参数、主图 1–7/正式补图 1–15、北大西洋零样本及集合校准证据边界。
- [Pangu–WRF：谱约束与海洋混合层耦合的两周台风预报](notes/medium-range/65-pangu-wrf-2week-tc.md) — 2024 同行评审原文；冻结底座与物理参数、五场跨海盆筛选、六组消融、图 1–7 和 14 天路径/强度成绩的可比性。
- [PuYun：大核注意力卷积与 10 天级联预报](notes/medium-range/66-puyun.md) — 2024 原始预印本；69 字段 ERA5、120k 单步预训练与两次 10k 自回归微调、原创结构图、完整第 10 天性能表与单步消融，明确 2017 验证交叉及未做的 0.1° 微调。
- [FuXi-Extreme：冻结伏羲底座的五天地表极端后处理](notes/medium-range/67-fuxi-extreme.md) — 2024 正式发表、2023 开放全文；六年 ERA5 与条件 DDPM 的真实训练参数、原创流程图、CSI/SEDI 与 RMSE/ACC 反向权衡、29 次台风回报及 IBTrACS/ERA5 强度结论翻转。
- [FengWu：长滚动、多模态与扩散集合](notes/medium-range/02-fengwu.md) — 结构重绘、10 天以上技巧证据表和正文/图注口径辨析。
- [FuXi Weather：真实卫星观测到 10 天全球预报](notes/medium-range/08-fuxi-weather.md) — 循环同化结构、Z500 技巧时效与逐变量性能表。
- [FuXi-2.0：小时级与气海表层联合中期预报](notes/medium-range/51-fuxi-2-0.md) — 88 场、6h/1h 双网络、训练超参、能源与台风真值口径。
- [Aardvark Weather：观测直达格点和站点](notes/medium-range/09-aardvark-weather.md) — 三模块结构、全球与分区性能表及不利结果。
- [Weather Prediction with Diffusion：引导式扩散预报](notes/medium-range/34-diffusion-guided.md) — 结构重绘、9–14 天证据表与引导模式边界。
- [SOFT：7 天自输出微调](notes/medium-range/31-soft.md) — 中期前段过渡方法，不能误称 10–15 天验证。
- [CoDiCast：6 天条件扩散](notes/medium-range/35-codicast.md) — 中期前段过渡方法，不能误称 10–15 天验证。
- [AIFS：ECMWF 确定性底座](notes/medium-range/41-aifs-base.md) — 2024 研究版的字段、三阶段训练/微调及 10 天评分。
- [AIFS-CRPS / AIFS ENS v1：近公平 CRPS 集合](notes/medium-range/37-aifs-crps.md) — 逐字段归一化、四阶段优化、15 天集合、46 天外推与业务版别边界。
- [hy-IFS-ENS：IFS–AIFS 逐成员谱约束混合集合](notes/medium-range/48-hybrid-ifs-aifs-ens.md) — 2026 独立方法论文；预训练/微调、T21/12h 约束、15 天评分及图 1–9 的负面结果。
- [hy-IFS：确定性 IFS–AIFS 模型层谱约束](notes/medium-range/49-hybrid-ifs-aifs-single.md) — 137 层 AI 的三阶段训练、36h 滚动微调、10 天逐图成绩和物理方案消融。
- [GDPS-SN：加拿大 GEM–GraphCast 谱约束](notes/medium-range/50-gdps-graphcast-spectral-nudging.md) — 2024 原始路线；13 层权重、DCT 双截断、两季 10 天验证及强天气尾部。
- [EPT-2 / EPT-2e：动态 lead 与能源变量集合](notes/medium-range/38-ept-2.md) — Jua 欧洲企业系列，公开主评测到 240h。
- [EPT-1.5：欧洲能源场景与 10 天格点/站点评测](notes/medium-range/45-ept-1-5.md) — 业务规格 20 天与实际公开验证 10 天分开。
- [FuXi-ENS：15 天流依赖集合](notes/medium-range/39-fuxi-ens.md) — 与确定性 FuXi 和 FuXi Weather 分篇。
- [AIFS-DOP：观测直达中期](notes/medium-range/42-aifs-dop.md) — 逐仪器数据期、缺测处理、三天 rollout 微调及 10 天验证。
- [GenCast：15 天条件扩散集合](notes/medium-range/43-gencast.md) — 1° 预训练至 0.25° 微调和概率技巧分开核查。
- [NeuralGCM：可微动力与神经参数化](notes/medium-range/44-neuralgcm.md) — 天气评分与多年气候模拟分别阅读。
- [AIFS Single 1.1：业务训练更新和边界层](notes/medium-range/46-aifs-single-1-1.md) — 公开 ERA5 预训练、IFS rollout 微调、降水消融。
- [AIFS Marine：大气、海表、海冰、海浪联合中期预报](notes/medium-range/52-aifs-marine.md) — ORAS6/ecWAM 数据配对、四变体参数、两阶段训练、补表 S.1 的约束/损失与图 1–14 的负面结果。
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
- [AIFS-SUBS：第 2–6 周概率集合](notes/s2s/53-aifs-subs.md) — 24h 两帧、ERA5/业务分析双版训练、五年回报与 29 周竞赛，含 MJO/SSW 及负面成绩。
- [FengWu-W2S：六小时连续的天气—次季节集合](notes/s2s/54-fengwu-w2s.md) — 78 字段海陆气特征交互、50+40 epoch 训练/微调、双扰动与 42 天图表的负技巧区。
- [热带松弛实验：Pangu/NeuralGCM 的第 3–4 周强降水事件](notes/s2s/55-tropical-relaxation-mlwp.md) — 30 成员、两种松弛掩码/变量、逐图 ACC/MAE 和 Rossby 波源负面反例；本文不重新训练模型。
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
- [SFNO-HENS/NeuralGCM：MJO 遥相关评估](notes/s2s/19-mjo-teleconnections-evaluation.md) — 加州大学开放的 43 页全文；447 次回报、58/10/11 成员、VPM 预处理、图 1–13、训练期重叠/不稳定成员及热带初值干预的完整证据。

### [短临、小时级与短期观测预报](notes/nowcasting/)

- [通用扩散概率降尺度：ERA5→CERRA，给多底座生成 5 km 集合](notes/nowcasting/80-universal-diffusion-downscaling.md) — 2026 Jua 独立方法论文；24M 参数/50 epoch/8×H100 的完整训练与 128 步采样、2014–2023 资料处理和 90h 站点评测，明确原稿样本计数/通道摘要矛盾及无 10–15 天证据。
- [欧洲站点极端天气检验：EPT-2.1、AIFS 和物理模式](notes/nowcasting/79-jua-europe-extremes.md) — 2026 Jua 主导的独立评测研究；欧洲 1,871 气象站/955 太阳站/4,005 雨量站、十个月对照与 48h 上限，详细拆解滚动去偏、极端阈值、四变量正负分数和缺失的模型训练信息。
- [WeatherGFM：视觉提示统一气象任务](notes/nowcasting/32-weathergfm.md) — 主实验含 SEVIR 小时级任务；ERA5 附录最长 7 天。
- [降水临近预报综述](notes/nowcasting/36-nowcasting-survey.md) — 临近预报分类与指标框架。
- [DAWP：卫星观测空间同化后做全球 72 小时短期预报](notes/nowcasting/74-dawp.md) — 2025 NeurIPS 主会；四模态卫星数据处理、VAE/AIDA/AIWP 三段训练、原创流程图、0–72h 通道误差和 12h 降水 CSI/FAR，以及强阈值误报反例与复现公平性边界。
- [Transformer-DOP：从多类直接观测做 12 小时预报的原型](notes/nowcasting/75-transformer-dop.md) — 2024 ECMWF 预印本；五类观测数据及角色、掩码预训练到预报微调的结构重绘、图 1–5 证据表与全部未披露的训练/评分参数，不能借后续 GraphDOP/DAWP 的成绩。
- [GraphDOP 表征探针：多传感器云、观测几何与单半球资料](notes/nowcasting/76-graphdop-representations.md) — 2025 ECMWF 独立研究；2013–2023 多类观测与 SEVIRI-only 另训实验、4×3h 潜步进及滚动微调的证据边界、图 1–10 的定量选摘和无观测半球退化。

### [数据同化](notes/assimilation/)

- [GraphDOP 的同化诊断：观测敏感度、FSOI 与有害通道](notes/assimilation/78-graphdop-fsoi.md) — 2025 ECMWF 独立诊断论文；缩小的 2013–2022 观测配置、约 58M 参数、Z-score/自动微分/FSOI 公式、图 1–7 与 ATMS ch4/17 正贡献反例；明确并未运行新同化系统或验证 10–15 天技巧。
- [XiChen：4DVar 梯度驱动观测同化](notes/assimilation/03-xichen.md) — 观测—分析—预报链重绘，分开呈现不同初值条件的性能。
- [DiffDA：扩散式天气尺度同化](notes/assimilation/33-diffda.md) — 稀疏观测与背景场融合，分开讨论分析和后续预报。
- [FuXi-En4DVar：冻结伏羲底座的集合四维变分同化](notes/assimilation/68-fuxi-en4dvar.md) — 200 成员 Perlin 背景集合、6 小时合成观测、L-BFGS 控制量优化及图 1–4；不冒称已验证 15 天预报。
- [FuXi-DA：FY-4B/AGRI 真卫星亮温的深度学习同化](notes/assimilation/69-fuxi-da.md) — 4 km→0.25° super-observation、三分支网络、6000 步训练与冻结 FuXi 的 60h 监督，区分订正收益和前 7 天的额外观测增益。
- [FengWu-4DVar：SHT 球面协方差与一年循环同化](notes/assimilation/70-fengwu-4dvar.md) — 正式 ICML 版的 1.4°/合成观测试验、M1/M6 分工、模型误差项消融、计算成本边界及附录最多 7 天预报。
- [Tensor-Var：学习特征空间中的凸四维变分同化](notes/assimilation/71-tensor-var.md) — 2025 ICML；两阶段编码器/CME 训练、伪卫星观测、数值表、图 10 时效坐标冲突及约 800 次同化后失稳。
- [4DVarFormer：四维变分梯度驱动区域分析场](notes/assimilation/72-4dvarformer.md) — 2024 正式论文及 15 页补充材料；区域 FourCastNet 的 80+40 epoch 训练/微调、同化网络参数、30 天循环与 7 天预报分开解读。
- [EarthNet：九模态卫星观测直接生成全球分析场](notes/assimilation/73-earthnet.md) — 2024 arXiv v1 全文；0.16° 多源卫星/MiRS 数据处理、三阶段 VAE/MMAE 训练、原创建模图、探空分层误差表和原文通道/附表矛盾。仅验证 1h 背景，无 10–15 天预报证据。

本次公开批次为 **84 篇独立笔记**，当前每条均按可取得的**原始全文、会议摘要或正式观点文章**逐篇核验并独立解读；会议摘要与 Perspective 不冒充完整实验论文。此前仅有摘要的 #19 已找到加州大学开放的 43 页同行评审稿，现按全文升级，仍注明其期刊补充材料尚未取得。#67 按开放预印本全文解读并核对期刊书目，正式正文受订阅限制；#68、#69 已按正式开放全文核对，但各自独立补充材料尚未取得；#70 已按 ICML 正式论文及附录核对；#71 依据 ICML 正式书目与 arXiv v3 全文，正式 PMLR PDF 尚未逐页核对；#72 依据出版社开放正文与正式补充材料；#73 依据 arXiv v1 全文/附录；#74 依据 NeurIPS 2025 正式书目与 arXiv v1 全文/附录，作者代码尚未放出；#75 依据 arXiv v1 正文及图 1–5，原稿不含量化性能表或可复现的训练超参数；#76 依据 arXiv v1 正文/附录与图 1–10，分清新数据/变体与原 GraphDOP 的参数边界；#77 依据 arXiv v1 HTML/图 1–7 核验，39 MB PDF 下载超时，文中不把个例当全球统计评测；#78 依据 arXiv v1 HTML/附表与图 1–7，辨明 FSOI 同化诊断与真实 OSE/预报评分的区别；#79 依据 arXiv v1 全文/附表 S1–S19，分清欧洲站点短期评测与独立 EPT 训练论文；#80 依据 arXiv v3 全文/附录，核验通用扩散降尺度的训练参数、采样步骤与 90h 边界；#81 依据 2024 GMD 正式论文/附录 A–D，区分第十天全球极端评分与 FuXi 的另一套评测口径；#82 依据 ICML 2025 书目与 arXiv v2 全文/图 1–14/表 1–2，正式 PMLR PDF 未逐页核对；#83 依据 *Science Advances* 正式全文/主图 1–4 与早期 arXiv 原图，正式独立补图 S1–S21 尚未逐张取得；#84 依据 arXiv v1 正文/表 1/图 1–5 及附图 13–20，指出 EasyUQ 样本内拟合的解释边界和原文样本数乘法冲突。**2024–2026 全年回溯仍未穷尽**，因此本仓库不宣称已收齐这三年的全部气象深度学习论文。[论文追踪表](气象大模型_中期预报论文追踪.md)保留全部 84 条书目与版本历史。

## 阅读笔记约定

每份详细笔记至少包括：论文与版本、核心问题、模型机制、训练与评测设置、带出处的具体结果、对照实验、证据局限、与中期/S2S 的关系和下一步复现检查。作者报告与阅读判断会明确区分。只有摘要或会议摘要可用时会标注资料不足，不填充推测细节。

## 来源与版权

此仓库只保存原创中文解读、依据论文重绘并标明出处的示意图、经核对的选摘数值表，以及指向原文的链接；不转载论文全文或未经授权的原图。论文与代码各归原作者及其授权方所有。不同数据集、阈值、预报时效或评测协议的数据不合并排名；缺少具体数字的图形不反推伪精确数值。
