# 项目：目标、背景和已有结论

永久文档，上限 150 行。只在目标或结论变化时改写。

## 目标

在 ImageNet 256×256 类条件生成上复现 REPA：SiT + REPA 训练 400K iter 后，无 CFG 的 FID-50K 达到论文报告值附近，并明显好于同步数的 vanilla SiT。

- 论文：Yu et al., *Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think*，ICLR 2025 Oral，arXiv:2410.06940
- 官方代码：https://github.com/sihyun-yu/REPA（README 自述"可能不能精确复现论文结果"）
- 复现规模：SiT-B/2 + REPA 训 400K，每 50K 存一个 EMA checkpoint，对照论文 100K / 200K / 400K 的 49.5 / 33.2 / 24.4。vanilla SiT-B/2 基线最后再训。
- 论文之外的参照：其他论文自己复跑 REPA-B/2 400K 为 22.0–22.8，vanilla B/2 400K 为 35.5–35.8；没有论文按 REPA 协议报告 vanilla B/2 在 400K 以下的数。

## 背景

扩散 Transformer 训练慢，论文认为瓶颈之一是模型要从头学到好的视觉表示。REPA 用冻结的自监督编码器（DINOv2）特征去对齐 SiT 中间层的表示，作为一个额外的正则项，宣称把 SiT-XL/2 达到同等 FID 的训练步数缩短 17.5 倍以上。

## 方法概要

- 主干：SiT（linear 插值路径，v 预测，uniform 权重），在 SD-VAE 的 32×32×4 latent 上训练。
- 对齐：取 SiT 第 `encoder-depth` 层（B/2 用 4，L/2、XL/2 用 8）的 patch token，经一个 3 层 SiLU MLP 投影器，和干净图像的 DINOv2-B patch 特征算负余弦相似度。
- 损失：L = L_velocity + λ·L_REPA，λ = 0.5（`--proj-coeff`）。投影器只在训练时使用。
- 训练：batch 256，AdamW，lr 1e-4 恒定，无 weight decay，fp16 混合精度 + 梯度裁剪。
- 采样与评测：SDE Euler–Maruyama，250 步；FID-50K，用 ADM（openai/guided-diffusion）评测套件和参考 batch。

## 论文报告值（复现目标，未在本仓库复现）

ImageNet 256×256，无 CFG，SDE 250 步，FID-50K。列：FID / sFID / IS / Prec / Rec。

| 模型 | 步数 | 结果 |
|---|---|---|
| SiT-B/2 | 400K | 33.0 / 6.46 / 43.7 / 0.53 / 0.63 |
| SiT-B/2 + REPA | 400K | 24.4 / 6.40 / 59.9 / 0.59 / 0.65 |
| SiT-L/2 | 400K | 18.8 / 5.29 / 72.0 / 0.64 / 0.64 |
| SiT-L/2 + REPA | 400K | 9.7（Table 2，DINOv2-B，depth 8）；Table 8 写 10.0，待核对 |
| SiT-XL/2 + REPA | 400K | 7.9 / 5.06 / 122.6 / 0.70 / 0.65 |
| SiT-XL/2 | 7M | 8.3 / 6.32 / 131.7 / 0.68 / 0.67 |
| SiT-XL/2 + REPA | 4M | 5.9 / 5.73 / 157.8 / 0.70 / 0.69 |
| SiT-XL/2 + REPA，CFG 1.8，区间 [0, 0.7] | 4M | 1.42 / 4.70 / 305.7 / 0.80 / 0.64 |

除 SiT-L/2 两行和 7.9、1.42 外，其余数字来自论文 HTML 的摘要提取，引用前对照论文原表再核对一次。

## 已有结论

每条一行，写数字，并标明证据强度：`[已验证]` 表示在当前评测入口上复现过，`[历史]` 表示来自旧项目且未在本仓库复现，`[观察]` 表示定性印象。

- （暂无）

## 已排除的方向

- （暂无）
