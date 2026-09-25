# Xaurora：Aurora 的谱一致性 S2S 微调（EGU 摘要解读）

> Eliot Walt、Wessel Bruinsma、Maurice Schmeits、Efstratios Gavves、Dim Coumou，*Xaurora: Advancing subseasonal-to-seasonal forecasting by fine-tuning foundation weather models with spectral consistency*，[EGU25-9956 官方摘要](https://doi.org/10.5194/egusphere-egu25-9956)，会议 2025-04-27 至 05-02；页面标注 2025-03-15 更新。初读 2026-09-24，2026-09-25 复核 EGU/EXCLAIM 原件和同名后续摘要。**资料级别仅为会议摘要，不是完整研究论文**；以下不会填充摘要未公开的表格或网络超参数。

## 研究问题与最小可核验结论

Aurora 已在中期天气预报表现较强，但把自回归天气基础模型带到两周至三个月 S2S 时，问题由“保留精细瞬时天气”转为“保留仍可预报的低频异常和遥相关”。EGU 摘要提出 Xaurora，意为 extended Aurora：在 Aurora 基座上做 S2S 微调，使潜在表征按目标时效聚焦可预报信号，并通过频域解码器与损失约束空间频谱一致性。摘要称会用消融、确定性/概率评分及遥相关指数评估，并“展示初步结果”，**但未披露任何具体指标数值、测试年、显著性区间或成绩表**。[官方摘要](https://meetingorganizer.copernicus.org/EGU25/EGU25-9956.html)

## 方法流程与能够推断、不能推断之处

公开摘要明确写了两个模块。第一是在潜在嵌入上添加回归 head，以 lead time 为条件保留可预报部分；第二是频域解码器和谱一致性损失，目标是让输出聚焦更可预报的频率。这可作为一种结构假设：远期天气的小尺度高频分量易失去相位技巧，过分追逐像素细节可能损害低频信号。然而**摘要没有公开频域变换选择、具体损失公式、谱段权重、回归头的训练标签或最终预报输出频率**，不能据关键词自行补写 FFT 权重公式或模型参数量。[官方摘要](https://meetingorganizer.copernicus.org/EGU25/EGU25-9956.html)

### 图：只包含摘要明确披露环节的概念流程

本图为本仓库根据[EGU 官方摘要](https://meetingorganizer.copernicus.org/EGU25/EGU25-9956.html)自行绘制的**高层流程**，不是未经公开的网络结构图。

```mermaid
flowchart LR
  A[天气初始状态] --> B[Aurora 基础模型潜在表示]
  C[目标 S2S 时效] --> D[附加回归 head]
  B --> D
  D --> E[按 lead 保留可预报信号]
  E --> F[频域解码器]
  F --> G[S2S 预报]
  H[谱一致性损失] -.微调约束.-> F
```

该图不能说明具体层数、频域变换或是否同时存在独立概率生成器；摘要对此均未给出。更不能把原 Aurora 的 **10 天正式论文天气成绩**算作 Xaurora 的 14–90 天实测成绩。[Aurora 正式论文](https://doi.org/10.1038/s41586-025-09005-y)、[EGU 摘要](https://doi.org/10.5194/egusphere-egu25-9956)

## 性能与实验披露表

| 应核验项目 | 当前官方摘要实际披露 | 对性能判断的影响 |
| --- | --- | --- |
| 目标范围 | S2S，约两周至三个月 | 是研究目标，不等于逐 lead 已证实技巧 |
| 微调基座 | Aurora | 可讨论迁移策略，但需基座版本/权重 |
| 对照 | 提及与原 Aurora 做谱一致性微调消融 | 无具体对照模型设置和数值 |
| 确定性/概率评分 | 表示会提供标准分数 | **未给指标名称、数值、样本量** |
| 遥相关指标 | 表示会考察 | 未给 MJO/ENSO 等具体指数和结果 |
| 数据切分/分辨率/成员数/成本 | 未披露 | 无法复现，也不能画准确性能曲线 |

因此这是一条**值得跟进的研究线索，不是已证实提升多少天的结果**。对照天津 Quan 的气候态融合、Aurora 的长滚动微调可提出研究假说，但没有相同数据和指标，不能放入统一排名。[EGU 摘要](https://doi.org/10.5194/egusphere-egu25-9956)

## 下一步检索与评审清单

### 同名材料的版本辨识（2026-09-25 核验）

| 材料 | 人员、实际方案 | 对本条能提供的新增证据 |
|---|---|---|
| [2025 EGU25-9956](https://meetingorganizer.copernicus.org/EGU25/EGU25-9956.html) | Walt、Bruinsma、Schmeits、Gavves、Coumou；**lead 条件回归头＋频域解码器/谱一致性微调** | 本条的唯一官方方法来源；只有文字摘要，没有实验表或具体超参数。 |
| [2025 EXCLAIM Symposium 日程册](https://ethz.ch/content/dam/ethz/special-interest/projects/exclaim-dam/documents/exclaim-symposium-2025---document-archive/EXCLAIM%20Symposium%20-%20Program%20brochure.pdf) | 同一标题、同一五位作者；列于 2025-06-02 的 15:15–15:30 口头报告 | 证明该工作另有学术报告记录，**日程册没有额外方法/结果**，不构成第二篇完整论文或独立复验。 |
| [2026 EMS2026-650](https://meetingorganizer.copernicus.org/EMS2026/EMS2026-650.html) | Walt、Kofinas、Mücke、Gavves、Coumou；标题改为 *Probabilistic Weather Forecasting with Foundation Models and Stochastic Interpolants*；公开方法为**随机插值框架＋LoRA 微调 Aurora 的漂移预测器** | 同名“Xaurora”但作者组合与方法重点均改变；摘要没有说明它是否继承 2025 年的频域解码器。**不能把 LoRA/随机插值训练细节倒灌到本条，也不能把其“竞争性”定性成绩冒充 2025 S2S 数值。** |

本轮按题名、第一作者、合作作者及 arXiv/官方会议页面检索，**未找到可公开核验的 2025 谱一致性 Xaurora 完整论文或原始性能表**；这里的结论只限本轮可检索公开资料，不等于论文永远不会发表。完整稿若出现，应核对频域损失公式/谱段、可预报信号标签、基座版别、训练/验证/测试年份、2/3/4/6 周逐 lead 技巧、概率成员构成及 CRPS/可靠性、遥相关指数和显著性。当前可给出的是**摘要的完整证据边界**，不是凭空补出“详细训练参数”的论文精读。[EGU 官方摘要](https://meetingorganizer.copernicus.org/EGU25/EGU25-9956.html)；[EMS 2026 官方摘要](https://meetingorganizer.copernicus.org/EMS2026/EMS2026-650.html)。

[返回仓库首页](../../README.md) · [返回总追踪表](../../气象大模型_中期预报论文追踪.md)
