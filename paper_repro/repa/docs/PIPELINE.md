# 流程：从数据到评测

永久文档，上限 120 行。只在命令或接口变化时改写。每一步写一条可以直接运行的命令。

下面的命令是官方仓库的写法，作为起点；接入 `src/repa/` 和 `configs/trunk.yaml` 后改写成本项目的命令。

## 环境

集群（训练、采样、评测）的环境【待定】；PC 只做 smoke。官方：`conda create -n repa python=3.9 -y && pip install -r requirements.txt`。每次运行前 `source tools/env.sh`。

## 1. 准备数据

见 docs/DATA.md 的"预处理"。

## 2. 训练

```bash
# 官方命令（SiT-XL/2）；实验用 --exp configs/exp/<id>.yaml 覆盖 configs/trunk.yaml
accelerate launch train.py --allow-tf32 --mixed-precision="fp16" --seed=0 \
  --path-type="linear" --prediction="v" --weighting="uniform" \
  --model="SiT-XL/2" --enc-type="dinov2-vit-b" --proj-coeff=0.5 --encoder-depth=8 \
  --output-dir="runs/<id>" --exp-name="<id>" --data-dir=<TARGET>
```

## 3. 推理

```bash
# 生成 50K 样本，输出 .npz 写到 runs/<id>/samples/，同一 checkpoint + 采样配置只生成一次
torchrun --nnodes=1 --nproc_per_node=<N> generate.py --model SiT-XL/2 --num-fid-samples 50000 \
  --ckpt <CKPT> --path-type=linear --encoder-depth=8 --projector-embed-dims=768 \
  --per-proc-batch-size=64 --mode=sde --num-steps=250 --cfg-scale=1.0
```

## 4. 评测（唯一入口）

```bash
# ADM 评测套件（openai/guided-diffusion/evaluations）
python evaluator.py <ADM_REF_BATCH_IN256.npz> runs/<id>/samples/<samples>.npz
```

- 评测集：ADM ImageNet 256 参考 batch vs 50K 生成样本，SDE 250 步，无 CFG（另报 CFG 时单独标注）
- 主指标：FID-50K，越低越好；同时记录 sFID、IS、Precision、Recall
- 必须对比的基线：同规模、同步数的 vanilla SiT（论文值见 docs/PROJECT.md；能自己训就以自训数为准）
- 训练中途的监控：FID-10K，只看趋势，不进结论；进结论的检查点一律 FID-50K
- 输出：`experiments/<id>/result.json`

所有实验都用这一条命令评测，结果才能互相比较。不另写评测脚本。
