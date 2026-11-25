# 当前状态

更新：2026-11-25

上限 40 行。每次会话结束时改写，不追加历史。

## 当前主干

主干未锁定。第一件事是用 trunk 卡片在 PC 上锁定 SD-VAE 的评测数。

- tag：【待填，锁定后为 paper_repro/eqvae/trunk-v1】
- commit：【待填】
- 配置：configs/trunk.yaml
- 评测集：ImageNet val 50k，官方 eval.py 协议（DATA.md 契约）
- 主指标：rFID 与等变误差 R(θ)、S(s)。基线是论文中的 SD-VAE（0.90 / 0.89 / 0.69）和 eq-vae（0.82 / 0.56 / 0.43）
- checkpoint：sd-vae（kl-f8），无训练

## 当前唯一问题

在 val 50k 上，原版 SD-VAE 和发布的 EQ-VAE 能否对上 Table 1 和 Table 12？→ 卡片：experiments/2026-11-25-released-vae-eval/CARD.md（2026-11-25 已批准）

正在跑：PC tmux `eqvae-trunk-eval`（run 2026-11-25-released-vae-eval-20261125-045211-pc），3 个模型依次，预计约 6.5 小时。Fig 1/2 可视化已先跑完（用户要求），待核对。翻译模型已停，评测完再用 Start-Translation.ps1 重启。

## 排队（最多 5 项）

1. ablation：EQ 微调 1 epoch，集群空闲时 → experiments/2026-11-25-eqvae-ft-1ep/CARD.md
2. ablation：不加 EQ 的对照微调，排在第 1 项之后 → experiments/2026-11-25-noeq-ft-1ep/CARD.md

## 搁置的想法

- analysis：Fig 3（τ∘x 与 D(τ∘E(x)) 之间的 rFID，s=0.25/0.5/0.75 和旋转），论文值 107/83/52/78 → 31/7/2/12。
- analysis：latent 的频谱（高频能量占比），解释为什么 EQ-VAE 的 latent 更平滑。
- ablation：续训到 5 epoch，看等变误差是否接近 eq-vae-ema。
