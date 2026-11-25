# 决策记录

上限 200 行。只记会改变后续工作方向的决策：更换主干、更换评测集或主指标、修改数据契约、放弃一个方向。作业参数、学习率、运行过程不属于决策，不记在这里。

每条不超过 5 行，编号顺延，不重复使用。被推翻的决策不删除，在后面加一行"被 D-xxx 取代"。

## D-001 微调数据用 ImageNet train 代替 OpenImages（2026-11-25）
- 决定：EQ 微调和对照微调都在集群已有的 ImageNet train 上做，先跑 1 epoch（等效 batch 160，约 8,007 步）。
- 理由：OpenImages 不在任何机器上，体积 500 GB 以上；作者发布的 `eq-vae-ema` 也是在 ImageNet 上训的。
- 影响：和论文的 `zelaki/eq-vae`（OpenImages 5 epoch）比时，数据和步数都不同，只能看方向和量级。

## D-002 评测协议照官方 eval.py，评测集为 ImageNet val 50k，评测在 PC 上做（2026-11-25）
- 决定：唯一入口是 `python -m eqvae.evaluate`。rFID 用 pytorch-fid 的 InceptionV3（2048 维），LPIPS 用 VGG，SSIM 用 torchmetrics，PSNR 在 uint8 上逐张计算后取平均。
- 理由：这样才能和 Table 1、Table 12 直接对比；集群用于训练，评测集中在一台机器上，环境版本才一致。
- 影响：集群上微调得到的 VAE 要导出成 diffusers 格式，经 bme_xfer 拉回 PC 评测。

## D-003 等变误差用 posterior 均值（2026-11-25）
- 决定：附录 C.3 的公式中，E(·) 取 posterior 均值，不采样。
- 理由：论文没写；均值是确定的，不引入采样噪声。重建指标仍按官方用 `.sample()`。
- 影响：如果 SD-VAE 测得的 R/S 与 0.89/0.69（或 0.93/0.80）差得多，先换成 `.sample()` 核对，再下结论。

## D-004 自写单卡训练循环，不搬官方 Lightning 代码（2026-11-25）
- 决定：用 diffusers `AutoencoderKL`、lpips VGG、taming NLayerDiscriminator 加 hinge 损失和自适应权重。超参照官方 `eqvae_config.yaml`。
- 和官方的一处差异：生成器和判别器在同一步看同一对变换后的样本。官方因为 Lightning 对两个优化器各调用一次 `training_step`，两者可能抽到不同的变换。
- 理由：官方环境是 torch 1.7 + Lightning 旧版，在集群上难以复用。
