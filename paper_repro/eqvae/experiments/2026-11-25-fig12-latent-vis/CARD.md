# 2026-11-25-fig12-latent-vis

类型：analysis　状态：queued　机器：pc

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
- 定性：看图。
- 定量辅助：在这 8 张图上，算 D(τ∘E(x)) 与 τ∘x 之间、以及 D(E(τ∘x)) 与 τ∘x 之间的逐图 PSNR，写进 result.json。不替代 trunk 的指标。

## 计算和存储预算
PC 上几分钟，显存 ≤4 GB；存储上限 50 MB。

## 要保存的产物
- result.json
- fig1_latent_pca.png（列为：输入、SD-VAE、eq-vae、eq-vae-ema）
- fig2_scale05.png（每个模型三列：τ∘x、D(E(τ∘x))、D(τ∘E(x))）
- fig2_rot90.png（版式同上）

## 结果
【运行后填写】
