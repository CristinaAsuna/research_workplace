# 流程：从数据到评测

永久文档，上限 120 行。只在命令或接口变化时改写。每一步写一条可以直接运行的命令。

## 环境

- **Mac**：`/Users/max/codebase/.ml/.venv/bin/python`，只跑 `pytest tests/` 和 ≤5 分钟的 CPU smoke。
- **PC**：`~/lab/envs/eqvae`，用 uv 新建，不改 `~/lab/envs/main`。torch 与 main 同版本（2.11+cu128），另装 `diffusers lpips pytorch-fid torchmetrics scipy pyyaml opencv-python-headless`。安装时带上 `UV_CACHE_DIR=~/lab/assets/cache/uv`。
- **集群**：`/public_bme2/bme-cuizhm/maxquan/research_iter/envs/eqvae`，依赖与 PC 相同。计算节点不能联网，所以 VGG16、LPIPS、kl-f8 权重要先在登录节点下载到 `$LAB_MODELS_DIR`。
- 每次运行前：

```bash
source tools/env.sh && export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 1. 准备数据

PC 从集群拉 val（只拉一次，拉完登记进 MACHINES.md）：

```bash
rsync -aHL --partial bme_xfer:/public_bme/home/maxquan/Datasets/Data/imagenet/val/ ~/lab/assets/datasets/imagenet-val/
```

## 2. 训练（集群，微调）

```bash
python -m eqvae.train --config configs/trunk.yaml --exp configs/exp/<id>.yaml
```

- checkpoint 存在 `runs/<id>/checkpoints/{last,best}.pt`。
- 训练结束时导出 diffusers 格式的 VAE：`runs/<id>/vae/`（`config.json` 和 `diffusion_pytorch_model.safetensors`）。

## 3. 推理

本项目没有单独的推理步骤。重建在评测入口里完成，结果不落盘；只有 val 的参考 Inception 统计用 `cache_dir("fid-ref", …)` 缓存一份。

## 4. 评测（唯一入口，PC）

```bash
python -m eqvae.evaluate --vae <hf-id 或 本地目录> --data ~/lab/assets/datasets/imagenet-val --n 50000 --seed 0 --out experiments/<id>/result.json
```

- 评测集：ImageNet val 全量 50k，协议见 DATA.md 的契约一节。
- 主指标：rFID（越低越好），以及等变误差 R(θ)、S(s)（越低越好）。同时报告 PSNR、LPIPS、SSIM。
- 必须对比的基线：同一入口下测得的原版 SD-VAE（论文值 rFID 0.90、R 0.89、S 0.69），以及 `zelaki/eq-vae`（0.82、0.56、0.43）。
- 输出：`experiments/<id>/result.json`。
- `--n 500` 只用于 smoke，结果写到 scratch/。

## 5. 可视化（Fig 1、Fig 2）

```bash
python -m eqvae.visualize --vaes sd-vae,zelaki/eq-vae,zelaki/eq-vae-ema --data ~/lab/assets/datasets/imagenet-val --out experiments/<id>/
```

所有实验都用这一条命令评测，结果才能互相比较。不另写评测脚本。
