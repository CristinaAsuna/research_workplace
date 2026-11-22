# 2026-11-22-sitb2-repa-400k

类型：trunk　状态：queued

## 问题
本项目的流程下，SiT-B/2 + REPA 训 400K，FID-50K（无 CFG）能否落在论文的 24.4 附近？中途对照论文 100K 的 49.5、200K 的 33.2。

## 影响的决定
- 400K ≤ 26.5：锁定 trunk-v1，下一张卡片从排队取（vanilla SiT-B/2 基线放在最后）。
- 400K > 26.5，或 100K > 60：停，不开消融；先用官方 SiT-XL/2 4M checkpoint 校验评测流程，再查训练实现。

## 预测
- agent：400K 22–26，与其他论文自己复跑的 22.0–22.8 同量级；100K 43–52。
- 用户：【可选】

## 推翻条件
400K FID-50K > 26.5，agent 的预测被推翻。< 20 时先查评测流程，不直接采信。阈值依据：别人复跑 REPA-B/2 400K 落在 22.0–24.4，单 seed 间 σ≈0.45。

## 相对主干的改动
主干未锁定，本卡片产生 configs/trunk.yaml 第一版：SiT-B/2，linear 路径，v 预测，uniform 权重；DINOv2-B 对齐第 4 层（待对照论文 Table 7 核对），3 层 SiLU MLP 投影器，负余弦，λ=0.5；batch 256，AdamW lr 1e-4 恒定，fp16，seed 0，400K 步。

## 评测
同 PIPELINE.md（ADM 评测，SDE 250 步，无 CFG）。100K、200K、400K 用 FID-50K；50K、150K、250K、300K、350K 只用 FID-10K 看趋势，不进结论。

## 计算和存储预算
- 机器：集群 `bme_gpu`，2×A100 80G PCIe（付费）。训练按 FLOPs 估墙钟 22–33 h（44–66 GPU 小时）；评测约 4 GPU 小时；先跑 1000 步实测速度再定。
- 训练墙钟超过 50 h，或任一 FID-50K 检查点触发上面的止损线，就停下问用户。
- 存储上限 20 GB（runs/ 下全部产物）；生成的样本 .npz 评测后删除。预处理后的 ImageNet 放登记的共享数据目录，不计入。

## 要保存的产物
result.json；checkpoints/last.pt（完整状态，用于续训）；checkpoints/ema_050k.pt … ema_400k.pt（每 50K 一个，只存 EMA 权重，共 8 个；需给 artifacts.py 加一个按名字保存的入口）；figures/fid_curve.png。

## 结果
【运行后填写：数字、与论文的差、结论、这个实验不能说明什么（不能说明 REPA 相对 vanilla 的加速，需要基线）】
