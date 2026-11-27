# 当前状态

更新：2026-11-27

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

准备：全量裁剪 1,281,162 张已核对（4359840），D-003 FID-50K 5.9584 通过（4359855），命名 checkpoint 续跑 smoke 通过（4360748）。全量 VAE 编码已核对（4362133）。24 worker 单卡测速（4363106）：batch 128 跑 1,000 步，稳态 0.3518 秒/步、363.9 张/秒，低于 0.45 门槛；batch 256 跑 200 步，0.6776 秒/步、峰值 58.6 GiB。双卡作业 4363187 因分布式初始化连接端口 0 超时，在第 0 步退出。修复启动端口后，CPU 双进程 Gloo 归约通过（两端均为 3，耗时 102.5 秒）。待重提双卡训练，前 1,000 步按 0.45 秒/步门槛检查，W&B 离线。

## 排队（最多 5 项）

1. vanilla SiT-B/2 400K 基线（同一流程，用来检验 REPA 的加速），放在最后

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。主干锁定后再开卡片。

- analysis：8 个 EMA checkpoint 上测 SiT 第 4 层与 DINOv2 的 CKNNA、空间结构指标、线性探测，和 FID 曲线对照（iREPA、Platonic）。
- ablation：iREPA（卷积投影器 + 编码器特征空间归一化），100K 对照 Table 8 的 49.5。
- ablation：HASTE 式从 full_200k.pt 起训到 400K、λ=0，和主干 400K 比。
- analysis：REPA-G，采样时用主干投影器加 DINOv2 特征引导，不训练。
