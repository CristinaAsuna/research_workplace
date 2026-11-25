# 实验登记表

每个实验一行，由 `tools/project.py new` 追加。状态：queued → running → done，另有 stopped（达到停止门槛）和 stale（主干已变，结果不再可比）。一句话结论不超过 40 字，写数字。从旧项目搬来的结论也记在这里，"卡片"一栏写旧项目里的路径。

| id | 类型 | 问题 | 影响的决定 | 状态 | 一句话结论 | 卡片 |
|---|---|---|---|---|---|---|
| 2026-11-22-sitb2-repa-400k | trunk | SiT-B/2+REPA 400K 的 FID-50K 能否到 24.4 附近 | 是否锁定 trunk-v1 | running | 单卡 615 步超时，稳态 1.83 秒/步 | [卡片](experiments/2026-11-22-sitb2-repa-400k/CARD.md) |
