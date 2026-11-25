# 2026-11-25-eqvae-ft-1ep

类型：ablation　状态：queued　机器：cluster（1×A100，集群空闲时）

## 问题
从 kl-f8 起，用 EQ-VAE 损失在 ImageNet train 上微调 1 个 epoch（等效 batch 160，8,007 步）后，等变误差 S(s) 和 R(θ) 能否降到接近 `zelaki/eq-vae`，同时 rFID 和 PSNR 不变差？

## 影响的决定
- S(s) 明显低于对照组（noeq），且接近发布版：问用户是否续训到 5 epoch，再把模型加进可视化。
- 否则：先对照官方 `autoencoder.py` 逐项排查实现，不增加训练步数。

## 预测
- S(s) 降到 trunk 中 SD-VAE 测值的 0.6 倍以下，R(θ) 也下降。
- rFID ≤1.0，PSNR 比 SD-VAE 低不超过 0.5 dB。

## 推翻条件
S(s) 比 SD-VAE 降低不到 30%，就判定为实现有问题或 1 个 epoch 不够。不调超参去凑数。

## 相对主干的改动
configs/exp/2026-11-25-eqvae-ft-1ep.yaml：在主干 SD-VAE 上做微调，`train` 段全部用主干默认值（p_eq=0.5）。

## 评测
同 PIPELINE.md。导出的 diffusers VAE 经 bme_xfer 拉回 PC，用同一入口评测 50k。

## 计算和存储预算
- 先 sbatch 跑 200 步测速和测显存。
- 正式运行时，按实测外推，超过 12 GPU 小时就先报告；超过外推时间的 1.5 倍就停。
- 出现 NaN 或损失发散就停。
- 存储上限 5 GB。

## 要保存的产物
result.json；checkpoints/last.pt（含判别器和优化器，用于续训）；vae/（diffusers 格式，用于评测）；train_log.jsonl。

## 结果
【运行后填写】
