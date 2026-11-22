# 数据：来源、契约和划分

永久文档，上限 150 行。数据契约只在这里定义，其他文件引用本文件，不重复写。

## 数据集

数据不随仓库分发。精确路径登记在研究工作区的 MACHINES.md 资产表里，这里只写标识和用途。

| 标识 | 内容和规模 | 用途 | 不能混淆的边界 |
|---|---|---|---|
| imagenet-1k-train | ILSVRC2012 train（Kaggle 版），1000 类，可读 1,281,162 张（原 1,281,167，5 张无法解码已隔离）；原始数据 157 GB | 训练 | 只用 train，不用 val 调参 |
| in256-images | 上面的 train 经 center-crop-dhariwal 到 256×256 的 PNG，约 100–200 GB（待实测） | 训练时给 DINOv2 的干净图像 | 和 latent 一一对应，同一次预处理产生 |
| in256-vae-sd | 上面的图像经 sd-vae-ft-mse 编码的 latent 分布参数，约 40 GB（待实测） | SiT 训练输入 | 编码用 ft-mse，解码用 ft-ema，两者都已在集群 |
| adm-ref-in256 | ADM 评测套件的 ImageNet 256 参考 batch（.npz） | FID/sFID/IS/P/R 的参考统计 | 只用于评测 |

位置一律写成 `$LAB_DATA_DIR/<名称>/`（集群上 `LAB_DATA_DIR=/public_bme/home/maxquan/Datasets/Data`，在 `~/.bashrc` 里设置）。只在集群上，PC 和 Mac 不存。
- `$LAB_DATA_DIR/imagenet/`（已有，2026-08 下载并校验）：
  - `raw/ILSVRC/`：原始数据；
  - `train/`、`val/`：ImageFolder 布局，符号链接，已去掉坏图；
  - `invalid_images_4062340.tsv`：坏图清单，坏图隔离在 `quarantine_invalid_links/`。
- `$LAB_DATA_DIR/imagenet-repa-in256/`（待生成）：`images/`（in256-images）和 `vae-sd/`（in256-vae-sd），作为训练的 `--data-dir`。写之前现场查空间和配额，生成后登记进 MACHINES.md。

## 契约

- 分辨率 256×256，裁剪方式 `center-crop-dhariwal`（ADM 预处理）。
- 目录结构：`<data-dir>/images/`（PNG）和 `<data-dir>/vae-sd/`（`img-mean-std-XXXXXXXX.npy` 加 `dataset.json` 标签），两个排序后的文件列表按索引配对。不能单独增删其中一边。
- latent 文件存 `cat([mean, std])`，形状 8×32×32，VAE 输入范围 [-1, 1]。训练时每步从 mean + std·ε 重新采样，再乘 0.18215，没有其他归一化。
- 解码：用 `sd-vae-ft-ema`，先除以 0.18215。
- DINOv2 输入：干净图像，顺序是 /255、ImageNet mean/std 归一化，再 bicubic 缩放到 224；对齐目标是 `x_norm_patchtokens`（256 个 token）。
- 标签：ImageNet 1000 类整数 id，以 `vae-sd/dataset.json` 为准。

## 划分

- 训练：ImageNet train 全量，不留验证集（论文做法）。
- 评测：不从数据集划分，而是生成 50K 张样本，和 ADM 参考 batch 比较，见 docs/PIPELINE.md。
- 冻结的测试集：ADM 参考 batch，只用于评测，不用于调参。

## 预处理

基于 edm2 的 `dataset_tools.py`（官方仓库 preprocessing/）。源目录用去掉坏图的 ImageFolder `train/`，不用 `raw/`（raw 里仍有 5 张无法解码的文件）。裁剪在 `bme_cpu`，编码在 GPU 队列：

```bash
python dataset_tools.py convert --source=$LAB_DATA_DIR/imagenet/train --dest=$LAB_DATA_DIR/imagenet-repa-in256/images --resolution=256x256 --transform=center-crop-dhariwal
python dataset_tools.py encode --source=$LAB_DATA_DIR/imagenet-repa-in256/images --dest=$LAB_DATA_DIR/imagenet-repa-in256/vae-sd
```
