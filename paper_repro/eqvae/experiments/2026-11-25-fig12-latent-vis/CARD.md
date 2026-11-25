# 2026-11-25-fig12-latent-vis

类型：analysis　状态：running（已跑完，待用户核对）　机器：pc

## 问题
在同样 8 张 val 图上，能否重现论文 Fig 1 和 Fig 2 的观察？
- Fig 1：SD-VAE 的 latent 主成分图有高频噪点，EQ-VAE 的更平滑。
- Fig 2：SD-VAE 的 D(τ∘E(x)) 明显退化，而 D(E(τ∘x)) 正常；EQ-VAE 两者都正常。

## 影响的决定
- 观察成立：正文结论写进 PROJECT.md；微调后把我们的模型加进同一组图，直接对比。
- 观察不成立：说明可视化或变换实现与论文不同。先对照官方 `vis_latent.py` 和 `autoencoder.py` 排查，再看 trunk 的等变误差。

## 预测
- Fig 1：三个主成分映射成 RGB 后，SD-VAE 有明显的格点状高频纹理；eq-vae 和 eq-vae-ema 的色块更连贯，能看出物体轮廓。
- Fig 2（s=0.5）：SD-VAE 的 D(τ∘E(x)) 模糊、有块状伪影，EQ 两版接近 D(E(τ∘x))。旋转 90° 时差异更小，但方向相同。

## 推翻条件
8 张图中有一半以上看不出 SD-VAE 与 eq-vae 的差别；或者 eq-vae 的 D(τ∘E(x)) 与 τ∘x 之间的逐图 PSNR，不比 SD-VAE 高至少 2 dB。

## 相对主干的改动
无训练，也不重新生成主干的评测缓存。只读三个模型的权重和 8 张固定的 val 图。名单为排序后第 0、6250、…、43750 张，与 DATA.md 一致。

## 评测
定性看图；定量辅助是这 8 张图上 D(τ∘E(x))、D(E(τ∘x)) 各自与 τ∘x 的逐图 PSNR，写进 result.json，不替代 trunk 的指标。

## 计算和存储预算
PC 上几分钟，显存 ≤4 GB；存储上限 50 MB。

## 要保存的产物
result.json；fig1_latent_pca.png（输入、SD-VAE、eq-vae、eq-vae-ema）；fig2_scale05.png 和 fig2_rot90.png（每个模型两列 D(E(τ∘x))、D(τ∘E(x))）。git 里的图缩到 1100 px 并量化为 256 色；原图在 runs/ 下的 figures_full/。

## 结果
run 2026-11-25-fig12-latent-vis-20261125-044911-pc，commit 5cb6832，72 秒。用户要求先出图，所以在 trunk 锁定前运行。
- 定性：两个观察都清楚复现。SD-VAE 的 PCA 图是格点噪声，EQ 两版能看出物体轮廓（ema 更平滑）。s=0.5 时 SD-VAE 的 D(τ∘E(x)) 糊成色块，EQ 两版接近 D(E(τ∘x))；rot90 时 SD-VAE 出现条带和错位伪影。
- D(τ∘E(x)) 与 τ∘x 的 PSNR（8 张均值，dB）：s=0.5 时 SD 20.31、eq 21.06、ema 21.68；rot90 时 20.01、22.26、23.95。D(E(τ∘x)) 三者都约 23.6（s=0.5）和 26.1（rot90）。
- 按事先写的推翻条件：s=0.5 时 eq-vae 只比 SD-VAE 高 0.75 dB，不到 2 dB，这一条成立（预测被推翻）；rot90 时高 2.25 dB，未触发。"旋转时差异更小"的预测也错了：旋转的差距更大。
- 解释（不改结论）：s=0.5 时 SD-VAE 的输出是低频正确的模糊图，PSNR 对模糊惩罚小，所以 PSNR 低估了图中的差距。定量结论以 trunk 的等变误差为准。
