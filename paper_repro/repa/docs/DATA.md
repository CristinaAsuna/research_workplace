# 数据：来源、契约和划分

永久文档，上限 150 行。数据契约只在这里定义，其他文件引用本文件，不重复写。

## 数据集

数据不随仓库分发。精确路径登记在研究工作区的 MACHINES.md 资产表里，这里只写标识和用途。

| 标识 | 内容和规模 | 用途 | 不能混淆的边界 |
|---|---|---|---|
| imagenet-1k-train | ILSVRC2012 train，约 128 万张，1000 类 | 训练 | 只用 train，不用 val 调参 |
| in256-images | 上面的 train 经 center-crop-dhariwal 到 256×256 | 训练时给 DINOv2 的干净图像 | 和 latent 一一对应，同一次预处理产生 |
| in256-vae-sd | 上面的图像经 SD-VAE 编码的 32×32×4 latent | SiT 训练输入 | VAE 版本必须和解码用的一致 |
| adm-ref-in256 | ADM 评测套件的 ImageNet 256 参考 batch（.npz） | FID/sFID/IS/P/R 的参考统计 | 只用于评测 |
| 机器与路径 | 集群上预处理和存放，具体共享目录【待定】，登记进 MACHINES.md | | |

## 契约

- 分辨率 256×256，裁剪方式 `center-crop-dhariwal`（ADM 预处理）。
- latent 形状 4×32×32，来自 Stable Diffusion VAE；解码用 `sd-vae-ft-ema`。latent 的缩放系数和均值/方差以官方 `dataset_tools.py encode` 的输出为准，【待核对：训练代码里是否另做归一化】。
- DINOv2 输入：干净图像（非加噪），按编码器自己的预处理缩放和归一化。
- 标签：ImageNet 1000 类整数 id，顺序以预处理脚本生成的 dataset.json 为准。

## 划分

- 训练：ImageNet train 全量，不留验证集（论文做法）。
- 评测：不从数据集划分，而是生成 50K 张样本，和 ADM 参考 batch 比较，见 docs/PIPELINE.md。
- 冻结的测试集：ADM 参考 batch，不用于调参或挑 checkpoint 之外的任何选择。

## 预处理

基于 edm2 的 `dataset_tools.py`（官方仓库 preprocessing/）：

```bash
python dataset_tools.py convert --source=<IMAGENET>/ILSVRC/Data/CLS-LOC/train --dest=<TARGET>/images --resolution=256x256 --transform=center-crop-dhariwal
python dataset_tools.py encode --source=<TARGET>/images --dest=<TARGET>/vae-sd
```

【待定：用官方代码还是自己实现；预计磁盘占用和耗时在运行卡片里写】
