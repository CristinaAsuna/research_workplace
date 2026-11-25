# 2026-11-25-released-vae-eval

类型：trunk　状态：queued　机器：pc

## 问题
在 ImageNet val 50k 上，用官方协议评测原版 SD-VAE（kl-f8）、`zelaki/eq-vae` 和 `zelaki/eq-vae-ema`，rFID、PSNR、LPIPS、SSIM 和 R(θ)、S(s) 能否对上论文 Table 1 和 Table 12？

## 影响的决定
- 对上了：锁定主干 trunk-v1（SD-VAE 加这套评测入口）；开始可视化卡片，等集群空闲后做微调卡片。
- 对不上：先查协议，包括 kl-f8 是否等于 SD-v1.5 的 vae、resize 方式、sample 还是 mean、PSNR 的定义；不进入微调。

## 预测
- SD-VAE：rFID 0.90±0.10，PSNR 25.82±0.3；eq-vae：rFID 0.82±0.10，PSNR 25.95±0.3。
- 等变误差：eq-vae 的 S(s) 比 SD-VAE 低至少 30%，R(θ) 也更低。

## 推翻条件
- 任一模型的 rFID 偏离超过 ±0.10，或 PSNR 偏离超过 ±0.3：协议不一致，先排查，不锁定。
- eq-vae 的 S(s) 不比 SD-VAE 低：要么等变误差的实现有错，要么论文的观察不成立。先按 D-003 换成 `.sample()` 核对，再下结论。

## 相对主干的改动
无。这张卡片建立主干：configs/trunk.yaml 的 `eval` 段加 `eqvae.evaluate`。

## 评测
同 PIPELINE.md 第 4 节。三个模型各跑一次，seed 0。

## 计算和存储预算
- PC 显存 ≤5 GB（运行前用 nvidia-smi 确认空闲显存）。
- 先用 500 张做 smoke 测吞吐；墙钟预计 2–4 小时，超过 6 小时就停下来报告。
- 存储上限 1 GB，只存 Inception 参考统计和指标。
- 准备工作：新建环境，下载 eq-vae、eq-vae-ema 和 kl-f8 权重，并从集群拉 val 6.4 GB；每一项执行前单独确认。

## 要保存的产物
result.json（三个模型的全部指标、论文值、差值）。参考统计放在 runs/_cache/fid-ref/（cache_dir），不进 git。

## 结果
【运行后填写】
