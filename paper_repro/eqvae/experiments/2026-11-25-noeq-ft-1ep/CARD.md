# 2026-11-25-noeq-ft-1ep

类型：ablation　状态：queued　机器：cluster（eqvae-ft-1ep 跑完后再提交）

## 问题
用同样的配方、同样的步数，但不加等变损失（p_eq=0）微调，等变误差是否基本不变？用它区分 EQ 组的改善来自等变损失，还是来自"多训一会儿"（对应论文 Table 6 的 SD-VAE†）。

## 影响的决定
- 对照组的 S(s) 接近 SD-VAE，而 EQ 组明显更低：结论是等变误差的下降来自等变损失，写进 PROJECT.md。
- 对照组的 S(s) 也大幅下降：说明是微调本身（比如数据从 OpenImages 换成 ImageNet）在起作用，回头重新审视 eqvae-ft-1ep 的结论。

## 预测
- S(s) 和 R(θ) 与 SD-VAE 相差不到 15%。
- rFID 与 EQ 组接近，论文 SD-VAE† 为 0.81。

## 推翻条件
对照组的 S(s) 比 SD-VAE 降低超过 30%。

## 相对主干的改动
configs/exp/2026-11-25-noeq-ft-1ep.yaml：在主干上微调，只设 `p_eq: 0`（不走等变分支），其余与 eqvae-ft-1ep 相同。低分辨率先验分支照常保留。

## 评测
同 PIPELINE.md，与 eqvae-ft-1ep 一样拉回 PC 评测。

## 计算和存储预算
与 eqvae-ft-1ep 相同：超过它实测时间的 1.5 倍就停；存储上限 5 GB。

## 要保存的产物
result.json；checkpoints/last.pt；vae/；train_log.jsonl。

## 结果
【运行后填写】
