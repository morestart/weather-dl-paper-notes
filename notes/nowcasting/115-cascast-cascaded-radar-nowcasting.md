# CasCast：雷达降水短临的确定性预测＋逐帧条件潜扩散

> Junchao Gong、Lei Bai、Peng Ye、Wanghan Xu、Na Liu、Jianhua Dai、Xiaokang Yang、Wanli Ouyang，*CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling*，**ICML 2024 正式论文**，PMLR 235:15809–15822，[正式书目](https://proceedings.mlr.press/v235/gong24a.html)、[14 页正式 PDF（含附录）](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)。早期预印本为 [arXiv:2402.04290](https://arxiv.org/abs/2402.04290)，首发于 **2024-02-06**。作者[公开代码与 SEVIR 权重](https://github.com/OpenEarthLab/CasCast)，本笔记另核查其 [2026-09-25 所见提交 487a68b](https://github.com/OpenEarthLab/CasCast/tree/487a68b5ade9aa829fe7df2e8f6746b4d9acc233) 的配置和数据加载。阅读于 2026-09-25；下文结构图为原创重绘，性能表转写自正式论文，不转载原图。

## 一、这篇究竟解决什么：只有 60 分钟，不是中期预报

CasCast 的输入和输出都是**雷达回波序列**。作者把短临分成两个问题：EarthFormer/SimVP 等确定性网络在像素空间定位大尺度雨区，CasFormer 在冻结自编码器的潜空间、以粗略雨图为条件生成细雨斑与强峰值。这样做针对单值均方误差常使小尺度强降水变模糊、纯生成式模型又可能把整体雨带放错位置的张力。[正式论文 §1、§3、图 1–3](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

主实验覆盖美国 SEVIR、香港 HKO-7 和法国 MeteoNet；三者**都是约 1 小时的雷达短临**：分别由 13/10/12 帧历史预测 12/10/12 帧未来。没有全球大气多变量预报、卫星直接观测到 10 天、3–15 天回报或 S2S。按用户的四目录分类归入 `notes/nowcasting/`。论文摘要的 **“+91.8%”** 是 SEVIR 上**单一高阈值、16 像素邻域 CSI** 相对 EarthFormer 的提升，非所有指标或全部地区平均提升。[正式表 1–4](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

## 二、数据、预处理与验证协议

| 数据 | 作者报告的源资料与切分 | 输入→输出、预处理和阈值 | 不能跨数据集直接比较的原因 |
| --- | --- | --- | --- |
| **SEVIR VIL** | 美国 2017–2019 的 NEXRAD 雷达 VIL 镶嵌；**35,718 训练 / 9,060 验证 / 12,159 测试**，每样本 **384×384、约 1 km/像素、5 min**。 | **13 帧（65 min）→12 帧（60 min）**。公开代码按 EarthFormer 的事件列表，每个 49 帧事件切出 **0–24、12–36、24–48/49** 三窗，加载 `npy` 时像素除 **255**，张量从 `1×H×W×T` 转 `T×1×H×W`。评分 VIL 原像素阈值 **16/74/133/160/181/219**；最高的 219 按论文附录换算约 **32.23 kg/m² VIL**。 | VIL 为**垂直积分液态水含量**，不是地面一小时降水量；SEVIR CRPS 是归一化雷达值的分数，不能冒称 mm/h。相邻三窗同源、时间重叠，需保持事件级 train/val/test 隔离。 |
| **HKO-7** | 2009–2015 香港地区约 **480×480、雷达观测高度 2 km、6 min**；**8,772/492/1,152**。 | **10 帧（60 min）→10 帧（60 min）**。论文说对应雨强门槛 **0.5/2/5/10/30 mm/h** 的像素门槛为 **84/118/141/158/185**；以最高的 `CSI-185` 报强雨。 | 约 **512×512 km** 区域/2 km 量级采样，不是与 SEVIR 的 1 km 同原始网格；附录 dBZ/像素换算文字应连同原始代码复核，不能直接套用 SEVIR 的 VIL 单位。 |
| **MeteoNet** | 法国 2016–2018 西北/东南雷达资料，本文用**东南区**；**6,978/2,234/994**，**0.01°、5 min**。原阵列 **565×784**，为避低质量/缺测区取左上 **400×400**。 | **12 帧（60 min）→12 帧（60 min）**。按 Marshall–Palmer `Z=200R^1.6` 把 **0.5/2/5/10/30 mm/h** 映射至约 **19/28/35/40/47 dBZ**；最高列为 `CSI-47`。 | 0.01° 是角度，不是各纬度严格 1 km；法国的 dBZ 门槛与香港的编码像素值、SEVIR 的 VIL 含义不同。 |

资料与门槛据[正式表 1、§4.1、附录 A](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)；SEVIR 三窗与 /255 则据[作者 README](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/README.md)、[数据加载器](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/datasets/sevir_used.py)。论文没有在主表列出三套数据的全部具体起止切分日期或 MeteoNet/HKO 所有归一化常数，不能填造“严格年外推”；作者公开列表可用于单独审计每个 SEVIR 事件是否跨集合。

**指标的统计对象**也要区分：CSI=TP/(TP+FP+FN)、HSS 计入 TN；`POOL1` 为原格点二值评分，`POOL4/16` 先对 **4×4/16×16 网格做 max pooling** 再评分，后两者允许小范围位置误差，所以数值通常更高，并不代表把预测分辨率真实提高。CSI-M 是多个雨强门槛的 CSI 均值。作者称所有模型按 **10 成员**算指标；确定性模型重复单值后，CRPS 退化为 MAE，不能把其“概率技能”与真正 10 条不同成员等同。CRPS 越低越好，SSIM/HSS/CSI 越高越好。[§4.1.2、附录 B](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

## 三、模型结构：原创流程图与逐阶段张量路径

~~~mermaid
flowchart LR
  A["历史雷达帧\nSEVIR 13×384×384"] --> B["阶段 1：像素空间确定性骨干\nSEVIR EarthFormer；HKO/MeteoNet SimVP"]
  B --> C["未来 12/10/12 帧粗预测\nMSE 监督，雨区位置相对稳定"]
  A --> D["阶段 2：逐帧自编码器 E\n历史帧压至连续潜变量"]
  C --> E["同一预训练 E\n未来粗预测逐帧压缩"]
  F["真值未来帧"] --> G["E 生成扩散训练目标"]
  E --> H["阶段 3：CasFormer 条件 DiT\n每个未来帧：粗潜变量＋带噪潜变量"]
  D --> H
  G --> H
  H --> I["逐帧 patch/半数 attention 层\n再聚合整段时序＋历史交叉注意力"]
  I --> J["噪声预测 MSE\n1000 步噪声日程"]
  J --> K["推理 DDIM 20 步\n重复采样 10 成员"]
  K --> L["冻结解码器 D\n高分辨率未来雷达回波"]
~~~

该图依据[正式图 1、图 3、式 1–3](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)重绘。**阶段 1**可以换 ConvLSTM、SimVP、EarthFormer；作者在主 SEVIR 实验用 EarthFormer，在 HKO-7/MeteoNet 用 SimVP，不是“一个 EarthFormer 权重零样本迁移三地”。确定性分支以未来真值序列的 MSE 学习均值/位移；它本身不输出随机成员。[§3.2、§4.2、表 4](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

**阶段 2**逐帧自编码器 `E/D` 用像素重建与对抗目标训练，把高分辨率雷达帧压为连续潜空间；论文图 4 对比不同潜维度，空间/通道压得越狠，**强阈值**重建 CSI/HSS 掉得越多，所以“潜扩散计算便宜”有可观察的极端强度损失边界。公开 SEVIR 配置为四层 down/up、通道 **128/256/512/512**、每层 **2** 残差块、潜通道 **4**、latent **48×48×4**；判别器从 step **25,001** 开始，`kl_weight=1e-6`、`disc_weight=0.5`、`perceptual_weight=0`。这组权重是**现行开源配置**，不是正文逐项公布的所有数据集最终超参。[§3.3、图 4](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)、[AE 配置](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/configs/sevir_used/autoencoder_kl_gan.yaml)

**阶段 3**把同一未来时刻的“模糊预测潜量”和“带噪真值潜量”沿通道拼接，分别做 patch embedding 与前半段 diffusion attention；随后 MLP 合并未来时序，使用历史雷达潜量做 cross-attention，再由后半段块联合解码出噪声/下一步潜量。对应噪声训练目标 `E||ε−εθ(z_k,k,[E(粗预测),E(历史)])||²`；新细节由高斯噪声的不同抽样得到，而不是简单在确定性图上随机加像素噪声。论文以 **1 帧一组**的指导获得比 **6 帧/12 帧一组**更好训练曲线与最终 CSI-M。公开配置 `split_num=12` 指整段**12 个未来帧**进入模型，代码逐帧 reshape；不能因为这个配置名字就误判论文实际用了图 7 较差的“12-frame-split”消融。[§3.3、图 7](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)、[CasFormer 实现](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/networks/casformer.py)

## 四、训练流程、公开参数、微调与代码/论文冲突

| 阶段 | 正式论文可核事实 | 现行作者 SEVIR 代码补充；不要与论文最终运行日志混同 |
| --- | --- | --- |
| 1. 确定性骨干 | 按 EarthFormer/SimVP 的既有训练方法，以 MSE 先训模糊未来帧；SEVIR 主版 EarthFormer，另外两地 SimVP。正文未逐项披露这三地所有骨干的最终 LR、batch、epoch、选优规则。 | [EarthFormer YAML](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/configs/sevir_used/EarthFormer.yaml) 为本地 batch **8**、4 GPU 脚本名义全局 batch **32**、AdamW LR **1e−3**、betas **0.9/0.999**、衰减 **1e−5**、余弦/暖启、`max_step=100000`；`max_epoch=1` 与步数共同存在，实际停止优先级须查运行日志。 |
| 2. 自编码器 | 像素重建＋对抗目标，**AdamW 峰值 LR 1e−4、余弦日程**；未给独立正式训练小时数和数据集间迁移精度。 | [AE YAML](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/configs/sevir_used/autoencoder_kl_gan.yaml) 为本地 batch **8**、4 GPU 脚本、`max_step=200000`、AdamW betas **0.9/0.999**、weight decay **1e−5**、FP32；具体重建/对抗/KL 权重见上节。 |
| 3. 缓存潜表示 | 论文图 3 要对历史、粗预测与目标使用同一预训练编码器；**不是在扩散训练期间从头联合更新所有模块**。 | [作者 README](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/README.md) 要先将目标真值和 EarthFormer 粗预测离线压缩，存到 `latent_data/sevir_latent/48x48x4`，再供 DiT 读取；公开权重也分骨干/AE/CasFormer 三件。 |
| 4. 条件扩散 | 线性噪声表 **1000 训练步**、classifier-free guidance、AdamW **LR 5e−4/余弦**，论文报告 **200k 训练 optimizer steps、4×A100、约 18 h、全局 batch 32**；推理 **DDIM 20 去噪步×10 成员**。 | [扩散 YAML](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/configs/sevir_used/cascast_diffusion.yaml) 为本地 batch **8**、4 GPU、β **1e−4→0.02**、`p_uncond=0.1`、`guidance_weight=1`、AdamW betas **0.9/0.95**；CasFormer 宽 **1152**、16 头、逐帧前段/整序列后段各 **12** 层、patch **2**。但该配置却写 `max_step=100000`，且[训练脚本](https://github.com/OpenEarthLab/CasCast/blob/487a68b5ade9aa829fe7df2e8f6746b4d9acc233/scripts/train_diffusion.sh) 加 `--debug`；不能声称现行脚本“一键复现了论文 200k 步正式成绩”。 |

这里没有“SEVIR 预训练→香港/法国小样本微调”的报告；正式论文限制段明说**跨数据集需要重新训练**。三个数据集上的骨干/扩散各按其资料训练，缺乏统一模型的零样本泛化证据。作者公开的 SEVIR 权重为三阶段分别下载，不能把条件引导的 `guidance_weight=1` 说成作者做过精调的最优值。论文未披露 HKO/MeteoNet 完整可复现的逐阶段优化器/日程/种子/硬件明细；本笔记不拿 SEVIR YAML 替代其他地区。[正式限制段](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

## 五、模型性能：正式表 2–4 的完整关键对照

### 5.1 SEVIR：常规/极端/概率三种读法

| 正式表 2 指标 | EarthFormer | PreDiff⋆ | NowcastNet | CasCast (EarthFormer) | 必须保留的限定 |
| --- | ---: | ---: | ---: | ---: | --- |
| **CRPS↓**，归一化 VIL | 0.0251 | **0.0202** | 0.0283 | **0.0202** | CasCast 与 PreDiff **并列**最低，正文“最低”不能读成严格独占。⋆PreDiff/LDM 因原生 384 尺度效果差而在 **128 下采样训练**，可比性不完全相同。 |
| **SSIM↑ / HSS↑** | 0.7756 / 0.5411 | 0.7648 / 0.4914 | 0.5696 / 0.5365 | **0.7797 / 0.5602** | 几何结构和列联总体略有优势，不推导校准。 |
| **CSI-M，POOL1 / POOL4 / POOL16↑** | 0.4310 / 0.4319 / 0.4351 | 0.3875 / 0.3918 / 0.4157 | 0.4152 / 0.4452 / 0.5024 | **0.4401 / 0.4640 / 0.5225** | 强调不同邻域定义；POOL16 最好不等于每格点雨斑最精确。 |
| **CSI-181，POOL1 / POOL16↑** | 0.2622 / 0.2562 | 0.2076 / 0.2264 | 0.2495 / 0.3725 | **0.2879 / 0.3900** | 区域强 VIL 指标，不能转换成 181 mm/h。 |
| **CSI-219，POOL1 / POOL16↑** | 0.1448 / 0.1481 | 0.1032 / 0.1213 | 0.1422 / 0.2700 | **0.1851 / 0.2841** | `0.2841/0.1481−1≈91.83%` 是摘要的最大相对提升；对更强的 NowcastNet 仅约 **5.2%**。 |

数值均转写自[正式表 2](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)。作者还说相对最佳确定性模型的 CSI-M POOL4/16 改善 **7.4%/20.1%**：按表 2 的分母分别取 EarthFormer 的 0.4319、PredRNN 的 0.4623，即 `0.4640/0.4319−1≈7.4%`、`0.5225/0.4623−1≈13.0%`；后一项若取 EarthFormer 0.4351，才约 **20.1%**。故正文“best-performing deterministic model”在 POOL16 的文字与表内 **PredRNN 0.4623 > EarthFormer 0.4351** 不完全一致；阅读时不要用 20.1% 当相对所有确定性基线的最优值。正式表的 † 注明 EarthFormer 的部分区域极端评分使用**官方 checkpoint**，不是一个标准化重训比较。[§4.2、表 2](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

### 5.2 香港与法国：不能只报获胜单元格

| 正式表 3 | SimVP（当地确定性骨干） | NowcastNet | PreDiff⋆ | CasCast（当地 SimVP 条件） | 读法 |
| --- | --- | --- | --- | --- | --- |
| **HKO-7 CRPS↓** | 0.0248 | 0.0296 | 0.0244 | **0.0205** | 归一化雨图而非 mm/h；三地分别训练。 |
| **HKO-7 CSI-M POOL1/16↑** | 0.4236 / 0.4134 | 0.4234 / 0.4724 | 0.3221 / 0.3046 | **0.4267 / 0.4938** | POOL1 优势很小，POOL16 较明显。 |
| **HKO-7 CSI-185 POOL1/16↑** | 0.1881 / 0.2233 | 0.2025 / 0.3601 | 0.0788 / 0.1113 | **0.2158 / 0.3653** | 相对 NowcastNet 的 POOL16 只高 **0.0052**。 |
| **MeteoNet CRPS↓** | 0.0218 | 0.0277 | 0.0197 | **0.0180** | 绝对 CRPS 不能直接与 SEVIR 原 VIL 比。 |
| **MeteoNet CSI-M POOL1/16↑** | 0.3017 / 0.3577 | 0.2955 / 0.3734 | 0.2546 / 0.2935 | **0.3156 / 0.4420** | 强调东南法国裁切 400×400 子区。 |
| **MeteoNet CSI-47 POOL1/16↑** | 0.0997 / 0.1599 | **0.1236** / 0.2115 | 0.0490 / 0.0867 | 0.1204 / **0.2357** | **原格点强雨 CasCast 低于 NowcastNet**，只有邻域强雨更高。 |

数值转写自[正式表 3](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)。论文称“三套数据均展示优势”是整体判断，不抹去 MeteoNet 原格点强阈值的反例。数据集之间的物理量、阈值、区域和骨干不相同，不能把绝对 CRPS 或 `CSI-219` 与 `CSI-47` 排成统一榜单。

### 5.3 消融图表：级联收益与逐帧指导

| SEVIR 表 4，单值骨干→CasCast | CSI-M POOL1 | CSI-M POOL16 | CSI-219 POOL1 | CSI-219 POOL16 |
| --- | ---: | ---: | ---: | ---: |
| 无确定性条件的 CasFormer | 0.3210 | 0.4402 | 0.0656 | 0.1733 |
| ConvLSTM → CasCast(ConvLSTM) | 0.4102 → **0.4145** | 0.4475 → **0.5044** | 0.1322 → **0.1575** | 0.1734 → **0.2593**（+49.54%） |
| SimVP → CasCast(SimVP) | 0.4153 → **0.4206** | 0.4530 → **0.5133** | 0.1338 → **0.1629** | 0.1685 → **0.2657**（+57.69%） |
| EarthFormer → CasCast(EarthFormer) | 0.4310 → **0.4401** | 0.4351 → **0.5225** | 0.1448 → **0.1851** | 0.1481 → **0.2841**（+91.83%） |

这比只看 EarthFormer 一组更能说明条件信息的角色：**无模糊先验的 CasFormer 空间错位更多**，有了不同强度的确定性骨干，区域强阈值均上升；但提升幅度随骨干/阈值变化，不是“级联给任何模型固定 +91.8%”。[表 4](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

[图 5](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf) 六个 10–60 min lead 的展示中，CasCast 这一样例比 ConvLSTM/SimVP 少平滑，且比 PreDiff 保留更强的局地峰值；这是**个例可视化**，不能替代表 2–3 的统计。图 6 的帧级 CSI-M、CSI-219 随 lead 增大普遍下滑，CasCast 曲线仍会衰减；作者没有证明到两小时后甚至 10 天的技巧。图 7 比较 1/6/12 帧分组，图例的终值 CSI-M 为 **44.01/43.54/43.21**（百分数口径，相当于 0.4401/0.4354/0.4321），1 帧组也有较低训练 loss。原图没有给跨 seed 方差，不足以声称微小差值具有统计显著性。[图 5–7](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gong24a/gong24a.pdf)

## 六、复现与迁移时最容易误读的边界

1. **不要把短临套给中期**：三套输入是过去雷达回波，预报仅 60 分钟，不包含 NWP 大尺度强迫；与 AIFS/GenCast/FuXi 的 10–15 天全球天气技能无法同列。
2. **不要把原图视觉信息当正式数表**：本笔记表中四位小数来自 PMLR 正式表 2–4；图 5 是案例，图 6–7 只引用曲线方向及图例数字。论文没有发布逐事件置信区间或所有模型原始评分数组。
3. **数据与评分口径**：SEVIR VIL 219 对应 kg/m² 积分液态水，不是雨强；HKO 像素 185 与 MeteoNet dBZ 47 也不等值；max pool 邻域评分比格点评分宽容。CRPS 与表中确定性模型的 MAE 等价，单一合成图与 10 成员随机集合不能混称。
4. **训练版本**：正式正文明确 CasFormer 200k steps/18 h，现行 SEVIR YAML 是 100k 且脚本有 `--debug`。至少要确认真正运行日志、checkpoint step、DDIM/CFG 推理设置与代码提交版本，才能主张复现了正式表。公开配置未覆盖 HKO/MeteoNet 的所有训练超参；不得以 SEVIR 配置推断它们。
5. **跨地区微调**：作者明说目前需对各数据集**重新训练**，并未报告冻结 SEVIR 编码器后只微调香港/法国的完整协议。目标是未来研究，不是已完成的零样本或参数高效迁移。

**阅读判断**：CasCast 的可靠贡献是把**空间定位较稳的单值雨带**变成生成器的条件，再以逐帧潜扩散补局地尾部；SEVIR 表 4 和三地区表 2–3 支持其约一小时短临收益。最不应省略的反面是 CRPS 与 PreDiff 并列、法国原格点强阈值未赢 NowcastNet、POOLED 指标更宽容、跨区要重训，以及现行公开训练上限与论文不一致。它与五天 He 等降水后处理在“确定性＋生成式”思想上相近，但**观测输入、尺度、训练资料和时效完全不同**，不能合并评分。
