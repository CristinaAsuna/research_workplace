# 流程：从数据到评测

永久文档，上限 120 行。只在命令或接口变化时改写。

## 环境

官方 REPA 源码固定为 `67f714503e3892f993844aab088ffc5791c92613`。集群用 uv 建两套独立环境，登录节点安装，计算节点直接调用解释器。

```bash
E=/public_bme2/bme-cuizhm/maxquan/research_iter/envs/repa
export UV_CACHE_DIR="$E/uv-cache" UV_PYTHON_INSTALL_DIR="$E/python"
uv venv --python /public_bme/home/maxquan/envs/uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 "$E/train"
uv python install 3.11
uv venv --python 3.11 "$E/adm"
uv pip sync --python "$E/train/bin/python" --build-constraint configs/requirements-train-build.txt configs/requirements-train.lock
uv pip sync --python "$E/adm/bin/python" configs/requirements-adm.lock
```

每次运行从项目目录执行以下命令。W&B 在训练入口设为 offline，记录落在对应运行目录。

```bash
source tools/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
E=/public_bme2/bme-cuizhm/maxquan/research_iter/envs/repa
```

- DINOv2-B：源码用 `$TORCH_HOME/hub/facebookresearch_dinov2_85a24602099d397264d5b30461ad7f3bfd726ca1/`，权重用同一 Hub 目录下的 `checkpoints/dinov2_vitb14_pretrain.pth`。
- VAE：`$LAB_MODELS_DIR/sd-vae-ft-mse/` 和 `$LAB_MODELS_DIR/sd-vae-ft-ema/`。
- ADM：官方 `openai/guided-diffusion@22e0df8` 的 `evaluator.py` 和本地 `classify_image_graph_def.pb`。参考 batch 位置见数据契约。D-003 另需官方 SiT-XL/2 4M checkpoint。

## 1. 准备数据

见 [DATA.md](DATA.md) 的“预处理”。

## 2. 训练

```bash
bash tools/launch-dual-gpu.sh "$E/train/bin/python" --module repa.train --exp-name 2026-11-22-sitb2-repa-400k
```

在分配到两张 GPU 的同一计算节点运行这条命令。脚本为 Accelerate 1.2.1 选择本地非零端口。默认读取 `configs/trunk.yaml`。用 `--exp configs/exp/<id>.yaml` 覆盖对应配置，显式 CLI 参数优先。batch 是全局 batch。续训用同一 `--exp-name` 加 `--resume`，读取完整的 `checkpoints/last.pt`。卡片约定的 50K 间隔另存 EMA 权重。

## 3. 采样

```bash
"$E/train/bin/python" -m torch.distributed.run --standalone --nproc_per_node="${N:-1}" --module repa.generate --ckpt "$CKPT"
```

默认 SiT-B/2、50K 样本、SDE 250 步、无 CFG。读取完整 checkpoint 的 EMA 或纯 EMA 文件。输出路径由程序打印，缓存中 `samples.npz` 为 ADM 输入。D-003 用本地官方 SiT-XL/2 4M checkpoint，显式设置 `--model SiT-XL/2 --encoder-depth 8`。

## 4. 评测（唯一入口）

在计算节点把 `SAMPLES` 设为采样输出的绝对路径。把 ADM 环境的 CUDA 库路径加到动态链接器搜索路径。

```bash
ADM_LIBS=$(find "$E/adm/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort | paste -sd :)
(cd "$LAB_MODELS_DIR/adm" && LD_LIBRARY_PATH="$ADM_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$E/adm/bin/python" "$E/adm/adm-evaluator/evaluator.py" "$LAB_DATA_DIR/adm-ref-in256/VIRTUAL_imagenet256_labeled.npz" "$SAMPLES")
```

- 评测集：ADM ImageNet 256 参考 batch vs 50K 生成样本，SDE 250 步，无 CFG。
- 主指标：FID-50K（越低越好），同时记录 sFID、IS、Precision、Recall。基线为同规模、同步数的 vanilla SiT，论文值见 [PROJECT.md](PROJECT.md)。
- 评测校验：官方 SiT-XL/2 4M checkpoint 的无 CFG FID-50K 应为 5.9 ± 0.5（D-003）。
- FID-10K 只用于中途监控。正式结果写入 `experiments/<id>/result.json`。
