# 2026-11-22-sitb2-repa-400k

类型：trunk　状态：queued

## 问题
本项目的流程下，SiT-B/2 + REPA 训 400K，FID-50K（无 CFG）能否落在论文的 24.4 附近？中途对照论文 Table 8：100K 49.5、200K 33.2。

## 影响的决定
- 400K ≤ 26.5：锁定 trunk-v1（锁定的是流程和这个数，不是 REPA 相对 vanilla 的效应），下一张卡片从排队取。
- 100K > 60、200K > 40 或 400K > 26.5：停，不开消融；评测已在第 0 步校验过，先查训练实现与官方代码的差异。

## 预测
- agent：400K 22–26，与其他论文自己复跑的 22.0–22.8 同量级；100K 43–52。
- 用户：【可选】

## 推翻条件
400K FID-50K > 26.5，agent 的预测被推翻。< 20 时先查评测流程，不直接采信。阈值依据：别人复跑 REPA-B/2 400K 落在 22.0–24.4，单 seed 间 σ≈0.45。

## 相对主干的改动
主干未锁定，本卡片产生 configs/trunk.yaml 第一版：搬入官方 REPA（commit 67f7145）的训练和采样代码；SiT-B/2，linear 路径，v 预测，uniform 权重；DINOv2-B 对齐第 4 层（Table 7、issue #5 已核实），3 层 SiLU MLP 投影器（2048 维），负余弦，λ=0.5；`--legacy`（对齐论文数字，见 D-002）；batch 256，AdamW lr 1e-4 恒定，fp16，梯度裁剪 1.0，EMA 0.9999，seed 0，400K 步。

## 评测
同 PIPELINE.md（ADM 评测，SDE 250 步，无 CFG）。100K、200K、400K 用 FID-50K；50K、150K、250K、300K、350K 只用 FID-10K 看趋势，不进结论。

## 计算和存储预算
- 第 0 步（与数据准备并行）：用官方 SiT-XL/2 4M checkpoint 生成 50K 样本评 FID，`bme_a10080g` 单卡，约 4–8 GPU 小时。与论文 5.9 相差超过 0.5 就停，不开始训练。
- 数据准备：ImageNet 已在集群（`Datasets/Data/imagenet/train/`，1,281,162 张），不用下载；裁剪成 256 PNG 约 100–200 GB（bme_cpu，数小时）、latent 约 40 GB（约 1–2 GPU 小时），放 `Datasets/Data/imagenet/repa_in256/`，不计入下面的上限。
- 训练：集群 `bme_gpu` 2×A100 80G PCIe（付费），FLOPs 估墙钟 22–33 h（44–66 GPU 小时），先跑 1000 步实测速度。墙钟超过 50 h 或出现 NaN 就停下问用户，不自行改精度。中间 FID 在 `bme_a10080g` 单卡上与训练并行，合计约 8 GPU 小时。
- 存储上限 30 GB（runs/ 下全部产物）；生成样本的 PNG 和 .npz 评测后删除。

## 要保存的产物
result.json；checkpoints/last.pt（完整状态，用于续训）；checkpoints/ema_050k.pt … ema_400k.pt（每 50K 一个，只存 EMA 权重，共 8 个；官方默认存完整状态，需改成经 artifacts.py 按名字保存）；figures/fid_curve.png。

## 结果
【运行后填写：数字、与论文的差、结论、这个实验不能说明什么（不能说明 REPA 相对 vanilla 的加速，需要基线）】
