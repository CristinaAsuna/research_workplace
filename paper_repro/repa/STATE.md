# 当前状态

更新：2026-11-22

上限 40 行。每次会话结束时改写，不追加历史。

## 当前主干

主干未锁定。第一件事是用一张 trunk 卡片跑通端到端流程，产出并锁定一个数。

- tag：【待填，锁定后为 paper_repro/repa/trunk-v1】
- commit：【待填】
- 配置：configs/trunk.yaml
- 评测集：ADM ImageNet 256 参考 batch vs 50K 样本，SDE 250 步，无 CFG
- 主指标：FID-50K（越低越好）；对比同规模同步数 vanilla SiT（论文值见 docs/PROJECT.md）
- checkpoint：【待填，机器:路径】

## 当前唯一问题

SiT-B/2 + REPA 训 400K 的 FID-50K 能否落在论文 24.4 附近（≤26.5）→ 卡片：experiments/2026-11-22-sitb2-repa-400k/CARD.md（2026-11-22 已批准）

准备：两套离线环境和共享资产已就绪，32 图 smoke 通过（4359826）。已提交全量裁剪（4359840）和 D-003 校验（4359855）。裁剪核对后再做 VAE 编码。训练提交前先 `git pull`：要用上 `full_checkpoint_steps`（100K、200K 存完整状态）。

## 排队（最多 5 项）

1. vanilla SiT-B/2 400K 基线（同一流程，用来检验 REPA 的加速），放在最后

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。主干锁定后再开卡片。

- analysis：8 个 EMA checkpoint 上测 SiT 第 4 层与 DINOv2 的 CKNNA、空间结构指标、线性探测，和 FID 曲线对照（iREPA、Platonic）。
- ablation：iREPA（卷积投影器 + 编码器特征空间归一化），100K 对照 Table 8 的 49.5。
- ablation：HASTE 式从 full_200k.pt 起训到 400K、λ=0，和主干 400K 比。
- analysis：REPA-G，采样时用主干投影器加 DINOv2 特征引导，不训练。
