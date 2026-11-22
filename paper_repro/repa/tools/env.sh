# 每次运行前先执行：source tools/env.sh（作业脚本放在第一行）。
# 工具缓存放进临时目录，作业结束即可丢弃，不进 runs/ 和 experiments/；
# 模型权重放进 MACHINES.md 登记的共享目录：每台机器在 ~/.bashrc 里设置 LAB_MODELS_DIR。

_cache="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER:-user}-repa-cache"
mkdir -p "$_cache"
export MPLCONFIGDIR="$_cache/matplotlib"
export TRITON_CACHE_DIR="$_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$_cache/inductor"
export CUDA_CACHE_PATH="$_cache/cuda"
export XDG_CACHE_HOME="$_cache/xdg"
export PIP_CACHE_DIR="$_cache/pip"
export PYTHONDONTWRITEBYTECODE=1

if [ -n "${LAB_MODELS_DIR:-}" ]; then
  export HF_HOME="$LAB_MODELS_DIR/huggingface"
  export TORCH_HOME="$LAB_MODELS_DIR/torch"
else
  echo "WARN tools/env.sh：没有设置 LAB_MODELS_DIR，模型权重会落到默认缓存。先在 ~/.bashrc 里设成 MACHINES.md 登记的共享权重目录。" >&2
fi
unset _cache
