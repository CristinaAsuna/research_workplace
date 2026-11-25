# 实验登记表

每个实验一行，由 `tools/project.py new` 追加。状态：queued → running → done，另有 stopped（达到停止门槛）和 stale（主干已变，结果不再可比）。一句话结论不超过 40 字，写数字。从旧项目搬来的结论也记在这里，"卡片"一栏写旧项目里的路径。

| id | 类型 | 问题 | 影响的决定 | 状态 | 一句话结论 | 卡片 |
|---|---|---|---|---|---|---|
| 2026-11-25-released-vae-eval | trunk | SD-VAE 与发布的 EQ-VAE 能否对上 Table 1/12 | 对上则锁定主干；对不上先查协议 | queued | — | [卡片](experiments/2026-11-25-released-vae-eval/CARD.md) |
| 2026-11-25-fig12-latent-vis | analysis | 能否重现 Fig 1 平滑 latent 与 Fig 2 latent 变换退化 | 成立则写入结论；否则查变换实现 | queued | — | [卡片](experiments/2026-11-25-fig12-latent-vis/CARD.md) |
| 2026-11-25-eqvae-ft-1ep | ablation | EQ 微调 1 epoch 能否降低 S(s) 且不伤重建 | 成立则问是否续训 5 epoch；否则查实现 | queued | — | [卡片](experiments/2026-11-25-eqvae-ft-1ep/CARD.md) |
| 2026-11-25-noeq-ft-1ep | ablation | 不加 EQ 的同配方微调，等变误差是否不变 | 不变则归因于等变损失；否则重审 EQ 结论 | queued | — | [卡片](experiments/2026-11-25-noeq-ft-1ep/CARD.md) |
