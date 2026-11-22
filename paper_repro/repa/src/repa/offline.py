"""Resolve staged VAE and DINOv2 files without consulting a model registry."""
import os
from pathlib import Path

DINO_REVISION = "85a24602099d397264d5b30461ad7f3bfd726ca1"


def load_vae(name):
    from diffusers.models import AutoencoderKL

    path = Path(name)
    if str(name) in {"stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"}:
        models = os.environ.get("LAB_MODELS_DIR")
        if not models:
            raise RuntimeError("Set LAB_MODELS_DIR to the shared model directory")
        path = Path(models) / path.name
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Local VAE config not found: {path / 'config.json'}")
    return AutoencoderKL.from_pretrained(str(path), local_files_only=True)


def load_dinov2(model_name):
    import torch

    if not os.environ.get("TORCH_HOME"):
        raise RuntimeError("Set TORCH_HOME to the shared Torch cache with tools/env.sh")
    hub = Path(torch.hub.get_dir())
    repo = hub / f"facebookresearch_dinov2_{DINO_REVISION}"
    weights = hub / "checkpoints" / f"{model_name}_pretrain.pth"
    for path in (repo / "hubconf.py", weights):
        if not path.is_file():
            raise FileNotFoundError(f"Local DINOv2 asset not found: {path}")
    model = torch.hub.load(str(repo), model_name, source="local", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model
