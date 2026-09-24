# Weather DL Paper Notes / 气象深度学习文献阅读笔记

2024–2026 年气象深度学习文献的中文阅读笔记，特别关注中期（约 10–15 天）与次季节预报。每篇论文保留独立文件，记录可核验的研究问题、方法、数据、指标、表格结果、局限和复现问题。

## 目录与分类

每篇完成稿只放一个目录，按论文的**主要研究问题和实际评测时效**分类：`notes/medium-range/` 为中期预报（重点约 8–15 天），`notes/s2s/` 为第 2 周之后至季节前段的次季节预测；跨越两者的论文按主要实验归档，并在笔记中注明另一时效。仅覆盖较短时效、通用方法、同化或临近预报的论文放在 `notes/other/`，不强行算作中期或 S2S。

### [中期预报](notes/medium-range/)

- [Aurora：异构地球系统预训练与中期天气](notes/medium-range/01-aurora.md) — 结构重绘、跨任务性能口径表与 10 天时效纠偏。
- [FengWu：长滚动、多模态与扩散集合](notes/medium-range/02-fengwu.md) — 结构重绘、10 天以上技巧证据表和正文/图注口径辨析。
- [XiChen：4DVar 梯度驱动观测同化](notes/medium-range/03-xichen.md) — 观测—分析—预报链重绘，分开呈现不同初值条件的性能。
- [FuXi Weather：真实卫星观测到 10 天全球预报](notes/medium-range/08-fuxi-weather.md) — 循环同化结构、Z500 技巧时效与逐变量性能表。
- [Aardvark Weather：观测直达格点和站点](notes/medium-range/09-aardvark-weather.md) — 三模块结构、全球与分区性能表及不利结果。
- [Weather Prediction with Diffusion：引导式扩散预报](notes/medium-range/34-diffusion-guided.md) — 结构重绘、9–14 天证据表与引导模式边界。

### [次季节预报（S2S）](notes/s2s/)

- [TianQuan-S2S：气候态融合与逐层噪声](notes/s2s/04-tianquan-s2s.md) — 模型结构、概率/确定性性能表及 Wind10 的负面结果。
- [Xaurora：谱一致性 S2S 微调](notes/s2s/05-xaurora-egu-abstract.md) — EGU 摘要级解读；不填造性能数字。
- [ESFM S2S 策略：多尾、LoRA 与慢变量注意力](notes/s2s/06-esfm-s2s-egu-abstract.md) — EGU 摘要及海报解读，记录 38 天/96 成员实例。
- [基础模型+MSWEP 的 S2S 降水](notes/s2s/07-s2s-precip-egu-abstract.md) — EGU 摘要级解读，厘清九年比较计划与未披露的结果。
- [PBC：S2S 概率偏差订正](notes/s2s/10-pbc-subseasonal.md) — 按 v3 新题名精读，含双分支概率校准结构图和竞赛口径。

### [其他气象深度学习](notes/other/)

- [SOFT：自输出微调与 7 天自回归天气预报](notes/other/31-soft.md) — 实测上限 7 天，不归入 8–15 天目录。
- [WeatherGFM：用视觉提示统一气象任务](notes/other/32-weathergfm.md) — 通用多任务方法，含 SEVIR 和 ERA5 实验。
- [DiffDA：扩散模型同化全球大气场](notes/other/33-diffda.md) — 重点是同化及短期预报证据。
- [CoDiCast：条件扩散全球预报](notes/other/35-codicast.md) — 实测上限 144 小时（6 天）。
- [降水临近预报综述](notes/other/36-nowcasting-survey.md) — 临近预报分类与指标框架。

其余 20 篇旧版短笔记正在重新核验，**尚未达到详细阅读笔记标准**，暂不放入上述完成稿目录。[论文追踪表](气象大模型_中期预报论文追踪.md)记录已发现的工作；2024–2026 全量回溯仍在进行中。

## 阅读笔记约定

每份详细笔记至少包括：论文与版本、核心问题、模型机制、训练与评测设置、带出处的具体结果、对照实验、证据局限、与中期/S2S 的关系和下一步复现检查。作者报告与阅读判断会明确区分。只有摘要或会议摘要可用时会标注资料不足，不填充推测细节。

## 来源与版权

此仓库只保存原创中文解读、依据论文重绘并标明出处的示意图、经核对的选摘数值表，以及指向原文的链接；不转载论文全文或未经授权的原图。论文与代码各归原作者及其授权方所有。不同数据集、阈值、预报时效或评测协议的数据不合并排名；缺少具体数字的图形不反推伪精确数值。
