# 数据：来源、契约和划分

永久文档，上限 150 行。数据契约只在这里定义，其他文件引用本文件，不重复写。

## 数据集

精确路径登记在总台 MACHINES.md 的资产表里，这里只写标识和用途。

| 标识 | 内容和规模 | 用途 | 位置 |
|---|---|---|---|
| imagenet-val | ILSVRC2012 val，1000 类 × 50 张 = 50,000 张 JPEG，约 6.4 GB | 唯一评测集（重建指标、等变误差）；可视化取其中 8 张 | cluster:`$LAB_DATA_DIR/imagenet/val/`（软链接视图）；pc:`~/lab/assets/datasets/imagenet-val/`（待从集群 rsync -L 拉取，拉取后登记） |
| imagenet-train | ILSVRC2012 train，可读 1,281,162 张（5 张坏图已隔离） | 微调（代替论文用的 OpenImages，见 D-001） | cluster:`$LAB_DATA_DIR/imagenet/train/`，只在集群 |

OpenImages 不在任何一台机器上，不下载（约 500 GB 以上）。

## 权重

| 标识 | 内容 | 位置 |
|---|---|---|
| sd-vae（kl-f8） | CompVis LDM `kl-f8`，论文的 SD-VAE 基座 | 待下载：`https://ommer-lab.com/files/latent-diffusion/kl-f8.zip` → `$LAB_MODELS_DIR/kl-f8/`。若与 SD-v1.5 的 `vae/` 逐张量相等，PC 上直接用已有的 HF 缓存 |
| eq-vae | `zelaki/eq-vae`，diffusers 格式，约 335 MB | 待下载：hf-mirror → `HF_HOME` |
| eq-vae-ema | `zelaki/eq-vae-ema`，diffusers 格式，约 335 MB | 同上 |

## 契约

- **评测输入**：与官方 `eval.py` 一致。先 `Resize(256)`（短边，bilinear），再 `CenterCrop(256)`，然后转到 [-1, 1]。
- **重建**：`z = posterior.sample()`，用固定 seed；`x̂ = D(z)`。PSNR、SSIM、LPIPS 以及 rFID 的 Inception 特征都在先量化到 uint8 的 x̂ 上计算。
- **变换 τ**（训练、评测、可视化共用 `eqvae.transforms`）：
  - 缩放：`F.interpolate(·, scale_factor=s, mode="bilinear", align_corners=False)`；旋转：`torch.rot90(·, k, dims=[-1, -2])`；先缩放后旋转。
  - 256 像素的图在 s=0.25/0.5/0.75 时缩到 64/128/192 像素，对应 latent 为 8/16/24，两边尺寸一致。
- **等变误差**：在 latent 空间、用 posterior 均值计算（D-003），公式见 PROJECT.md。
- **微调输入**：与官方 OpenImages 加载器一致。随机方形裁剪，边长为短边 × U(0.5, 1)，再用 INTER_AREA 缩到 256；不翻转，范围 [-1, 1]。

## 划分

- 微调只用 imagenet-train，评测只用 imagenet-val 全量 50k，不用 val 调参。
- 可视化用 val 里固定的 8 张（按排序后的文件名，每隔 6,250 张取一张），名单写进卡片。
