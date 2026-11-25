"""Load the VAEs by alias, and convert the CompVis LDM kl-f8 checkpoint to diffusers format.

CLI (run once per machine after downloading kl-f8.zip):
  python -m eqvae.models convert --ckpt $LAB_MODELS_DIR/kl-f8/model.ckpt \
      --out $LAB_MODELS_DIR/kl-f8/diffusers [--compare <diffusers vae dir or hf id>]
"""
import argparse
import os
from pathlib import Path

import torch

# SD-VAE (LDM kl-f8) architecture in diffusers terms.
SD_VAE_CONFIG = dict(
    in_channels=3, out_channels=3,
    down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
    block_out_channels=[128, 256, 512, 512], layers_per_block=2, act_fn="silu",
    latent_channels=4, norm_num_groups=32, sample_size=256, scaling_factor=0.18215,
)

ALIASES = {"eq-vae": "zelaki/eq-vae", "eq-vae-ema": "zelaki/eq-vae-ema"}


def sd_vae_dir():
    if os.environ.get("EQVAE_SD_VAE"):
        return Path(os.environ["EQVAE_SD_VAE"])
    if not os.environ.get("LAB_MODELS_DIR"):
        raise RuntimeError("set LAB_MODELS_DIR (or EQVAE_SD_VAE) to locate the converted kl-f8 VAE")
    return Path(os.environ["LAB_MODELS_DIR"]) / "kl-f8" / "diffusers"


def resolve(name):
    """Alias -> local dir or HF id. 'sd-vae' is the converted kl-f8; anything else passes through."""
    if name == "sd-vae":
        return str(sd_vae_dir())
    return ALIASES.get(name, name)


def load_vae(name, device="cpu", dtype=torch.float32):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(resolve(name), torch_dtype=dtype).to(device).eval()


def load_ldm_state(ckpt):
    """State dict of an LDM Lightning checkpoint. Falls back to full unpickling only for the
    official kl-f8 file, which may carry non-tensor metadata."""
    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - old Lightning pickles
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    return obj.get("state_dict", obj)


def discriminator_state(ckpt):
    """loss.discriminator.* weights from the LDM checkpoint ({} if it has none)."""
    prefix = "loss.discriminator."
    return {k[len(prefix):]: v for k, v in load_ldm_state(ckpt).items() if k.startswith(prefix)}


def convert_kl_f8(ckpt, out=None):
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    state = {k: v for k, v in load_ldm_state(ckpt).items() if not k.startswith("loss.")}
    vae = AutoencoderKL(**SD_VAE_CONFIG)
    vae.load_state_dict(convert_ldm_vae_checkpoint(state, SD_VAE_CONFIG), strict=True)
    if out is not None:
        vae.save_pretrained(out)
    return vae


def compare_state(a, b):
    """Max |a-b| over all tensors, or None when the key sets or shapes differ."""
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys() or any(sa[k].shape != sb[k].shape for k in sa):
        return None
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    conv = sub.add_parser("convert")
    conv.add_argument("--ckpt", required=True)
    conv.add_argument("--out", required=True)
    conv.add_argument("--compare", help="diffusers VAE dir or HF id (e.g. SD-v1.5 with --subfolder vae)")
    conv.add_argument("--subfolder", default=None)
    args = parser.parse_args(argv)
    vae = convert_kl_f8(args.ckpt, args.out)
    print("saved", args.out, "| discriminator tensors in ckpt:", len(discriminator_state(args.ckpt)))
    if args.compare:
        from diffusers import AutoencoderKL
        other = AutoencoderKL.from_pretrained(args.compare, subfolder=args.subfolder)
        print("max |kl-f8 - compare| =", compare_state(vae, other))


if __name__ == "__main__":
    main()
