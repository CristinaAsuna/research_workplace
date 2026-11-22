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

准备：两套离线环境和共享资产已就绪；32 图转换、VAE 编码、3 步训练及 ADM GPU 预热通过（作业 4359826）。待全量预处理和 D-003 校验。

## 排队（最多 5 项）

1. vanilla SiT-B/2 400K 基线（同一流程，用来检验 REPA 的加速），放在最后

## 搁置的想法

做当前卡片时想到的其他实验写在这里，一行一条，不执行。

（暂无）
