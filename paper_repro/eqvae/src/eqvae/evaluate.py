"""The single evaluation entry (docs/PIPELINE.md §4).

python -m eqvae.evaluate --vae sd-vae --data ~/lab/assets/datasets/imagenet-val --n 50000 \
    --exp 2026-11-25-released-vae-eval --out experiments/2026-11-25-released-vae-eval/result.json

Each call evaluates one VAE and merges its metrics into --out under metrics[<vae>], so one
result.json can hold several models. Reconstructions are not written to disk; the Inception
statistics of the reference images are cached once per (file list, protocol).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import artifacts
from .config import TRUNK, git_commit, load_config
from .data import EvalImages, even_subset, list_images
from .metrics import (InceptionFeatures, Lpips, equivariance_errors, fid, gaussian_stats,
                      psnr_uint8, ssim_per_image, to_uint8)

PROTOCOL = "resize256-centercrop256-u8trunc-pytorchfid2048-v1"

# Paper numbers (arXiv 2502.09509v3 Table 1 / Table 12; README for eq-vae-ema). See docs/PROJECT.md.
PAPER = {
    "sd-vae": {"rfid": 0.90, "psnr": 25.82, "lpips": 0.146, "ssim": 0.71, "equiv_rot": 0.89, "equiv_scale": 0.69,
               "equiv_rot_v1v2": 0.93, "equiv_scale_v1v2": 0.80},
    "eq-vae": {"rfid": 0.82, "psnr": 25.95, "lpips": 0.141, "ssim": 0.72, "equiv_rot": 0.56, "equiv_scale": 0.43,
               "equiv_rot_v1v2": 0.49, "equiv_scale_v1v2": 0.15},
    "eq-vae-ema": {"rfid": 0.55, "psnr": 26.15, "lpips": 0.133, "ssim": 0.72},
}


def _reference_stats(files, size):
    names = "\n".join(str(p.name) for p in files).encode()
    key = dict(files_sha=hashlib.sha256(names).hexdigest()[:16], n=len(files), size=size, protocol=PROTOCOL)
    path = artifacts.cache_dir("fid-ref", **key)
    if artifacts.is_complete(path):
        data = np.load(path / "stats.npz")
        return path, (data["mu"], data["sigma"])
    return path, None


@torch.no_grad()
def evaluate(vae_name, data, n, seed=0, batch_size=16, workers=8, device="cuda", cfg=None, log_every=50,
             equiv_n=None):
    from .models import load_vae
    cfg = cfg or load_config()["eval"]
    files = even_subset(list_images(data), n)
    equiv_n = min(equiv_n or cfg.get("equiv_n") or n, len(files))
    equiv_idx = torch.zeros(len(files), dtype=torch.bool)
    equiv_idx[torch.from_numpy(np.linspace(0, len(files) - 1, equiv_n).round().astype(np.int64))] = True
    if len(files) < n:
        raise RuntimeError("only {} images under {}, asked for {}".format(len(files), data, n))
    loader = DataLoader(EvalImages(files, cfg["size"]), batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.startswith("cuda"))
    ref_path, ref_stats = _reference_stats(files, cfg["size"])

    vae = load_vae(vae_name, device)
    lpips_net, inception = Lpips(device), InceptionFeatures(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    psnr, ssim, lp, rec_feats, ref_feats = [], [], [], [], []
    equiv = {}
    start = time.time()
    for it, (x, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        post = vae.encode(x).latent_dist
        rec = vae.decode(post.sample(generator=gen).float()).sample
        rec_u8, x_u8 = to_uint8(rec), to_uint8(x)
        psnr.append(psnr_uint8(rec_u8, x_u8).cpu())
        ssim.append(ssim_per_image(rec, x).cpu())
        lp.append(lpips_net(rec, x).cpu())
        rec_feats.append(inception(rec_u8))
        if ref_stats is None:
            ref_feats.append(inception(x_u8))
        sel = equiv_idx[idx].to(x.device)
        if sel.any():
            errs = equivariance_errors(vae, x[sel], post.mean.float()[sel], cfg["rot_k"], cfg["scales"])
            for key, val in errs.items():
                equiv.setdefault(key, []).append(val)
        if log_every and it % log_every == 0:
            done = min((it + 1) * batch_size, len(files))
            print("[{}] {}/{} images, {:.1f} img/s".format(vae_name, done, len(files),
                                                          done / (time.time() - start)), flush=True)

    if ref_stats is None:
        ref_stats = gaussian_stats(torch.cat(ref_feats).numpy())
        np.savez(ref_path / "stats.npz", mu=ref_stats[0], sigma=ref_stats[1])
        artifacts.mark_complete(ref_path)
    rec_stats = gaussian_stats(torch.cat(rec_feats).numpy())

    equiv = {k: float(torch.cat(v).mean()) for k, v in equiv.items()}
    rot_keys = ["rot{}".format(k) for k in cfg["rot_k"]]
    scale_keys = ["scale{}".format(s) for s in cfg["scales"]]
    runtime = time.time() - start
    return {
        "n": len(files),
        "equiv_n": int(equiv_idx.sum()),
        "rfid": fid(*ref_stats, *rec_stats),
        "psnr": float(torch.cat(psnr).mean()),
        "lpips": float(torch.cat(lp).mean()),
        "ssim": float(torch.cat(ssim).mean()),
        "equiv_rot": float(np.mean([equiv[k] for k in rot_keys])),
        "equiv_scale": float(np.mean([equiv[k] for k in scale_keys])),
        "equiv_detail": equiv,
        "img_per_sec": len(files) / runtime,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None,
        "runtime_sec": runtime,
    }


def merge_result(out, exp, vae_name, metrics, data, n, seed):
    out = Path(out)
    result = json.loads(out.read_text()) if out.exists() else {}
    result.update({
        "exp": exp,
        "commit": git_commit(),
        "config": str(TRUNK.relative_to(artifacts.ROOT)),
        "eval_set": "ImageNet val, {} images ({}), equivariance error on an even {} subset, protocol {}".format(
            n, data, metrics.get("equiv_n"), PROTOCOL),
        "seed": seed,
    })
    result.setdefault("metrics", {})[vae_name] = {k: v for k, v in metrics.items() if k != "runtime_sec"}
    result.setdefault("runtime_sec", {})[vae_name] = metrics["runtime_sec"]
    result["baseline"] = {"paper": PAPER}
    result.setdefault("outputs", {})["fid_ref_cache"] = "runs/_cache/fid-ref/ on the evaluating machine"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", required=True, help="sd-vae | eq-vae | eq-vae-ema | local dir | HF id")
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--equiv-n", type=int, default=None, help="images for the equivariance error (default eval.equiv_n)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--name", default=None, help="key in result.json (default: --vae)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg = load_config()["eval"]
    if args.n < cfg["n"] and not str(Path(args.out).resolve()).startswith(str(artifacts.ROOT / "scratch")):
        parser.error("runs with --n < {} are smoke runs: write --out under scratch/".format(cfg["n"]))
    metrics = evaluate(args.vae, args.data, args.n, args.seed, args.batch_size or cfg["batch_size"],
                       args.workers, args.device, cfg, equiv_n=args.equiv_n)
    merge_result(args.out, args.exp, args.name or args.vae, metrics, args.data, args.n, args.seed)
    print(json.dumps({k: v for k, v in metrics.items() if k != "equiv_detail"}, indent=2))


if __name__ == "__main__":
    main()
