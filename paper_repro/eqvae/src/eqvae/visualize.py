"""Fig 1 (latent PCA) and Fig 2 (decoding transformed latents) of the EQ-VAE paper.

python -m eqvae.visualize --vaes sd-vae,eq-vae,eq-vae-ema --data ~/lab/assets/datasets/imagenet-val \
    --exp 2026-11-25-fig12-latent-vis --out experiments/2026-11-25-fig12-latent-vis

Writes fig1_latent_pca.png, fig2_scale05.png, fig2_rot90.png and result.json (per-image PSNR of
D(tau∘E(x)) and D(E(tau∘x)) against tau∘x). Fig 1 follows evaluation/vis_latent.py of the official
repo (per-image PCA, top-3 components, joint min-max); Fig 2 uses the posterior mean.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import git_commit
from .data import EvalImages, list_images, vis_subset
from .metrics import psnr_uint8, to_uint8
from .transforms import tau


def pca_rgb(latent):
    """(C, H, W) latent -> (3, H, W) in [0, 1]: top-3 principal components, joint min-max."""
    c, h, w = latent.shape
    flat = latent.permute(1, 2, 0).reshape(-1, c).double()
    flat = flat - flat.mean(0, keepdim=True)
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    _, vecs = torch.linalg.eigh(cov)
    rgb = flat @ vecs[:, -3:]  # eigh is ascending, so R,G,B = PC3, PC2, PC1 as in vis_latent.py
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.reshape(h, w, 3).permute(2, 0, 1).float()


def _img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def _grid(rows, col_titles, path, cell=1.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows, n_cols = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell * n_cols, cell * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            ax = axes[r][c]
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run(vae_names, data, out, n_images=8, seed=0, device="cuda", transforms=(("scale05", 0.5, 0), ("rot90", 1.0, 1))):
    from .models import load_vae
    files = vis_subset(list_images(data), n_images)
    ds = EvalImages(files)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    pca_cols = {}
    fig2 = {name: {} for name, _, _ in transforms}
    stats = {}
    for vname in vae_names:
        vae = load_vae(vname, device)
        gen = torch.Generator(device=device).manual_seed(seed)
        post = vae.encode(x).latent_dist
        sf = getattr(vae.config, "scaling_factor", 1.0) or 1.0
        z_sample = post.sample(generator=gen) * sf
        pca_cols[vname] = [F.interpolate(pca_rgb(z.cpu())[None], size=256, mode="bilinear",
                                         align_corners=False)[0].permute(1, 2, 0).numpy() for z in z_sample]
        z = post.mean.float()
        for tname, s, k in transforms:
            tx = tau(x, s, k)
            dec_in = vae.decode(vae.encode(tx).latent_dist.mean).sample      # D(E(tau∘x))
            dec_lat = vae.decode(tau(z, s, k)).sample                          # D(tau∘E(x))
            fig2[tname][vname] = (dec_in, dec_lat)
            p_in = psnr_uint8(to_uint8(dec_in), to_uint8(tx)).cpu().tolist()
            p_lat = psnr_uint8(to_uint8(dec_lat), to_uint8(tx)).cpu().tolist()
            stats.setdefault(vname, {})[tname] = {
                "psnr_D_E_tau_x": p_in, "psnr_D_tau_E_x": p_lat,
                "mean_psnr_D_E_tau_x": float(np.mean(p_in)), "mean_psnr_D_tau_E_x": float(np.mean(p_lat))}
        del vae
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    _grid([[_img(x[i])] + [pca_cols[v][i] for v in vae_names] for i in range(len(files))],
          ["input x"] + ["{} PCA".format(v) for v in vae_names], out / "fig1_latent_pca.png")
    for tname, s, k in transforms:
        label = "s={}".format(s) if k == 0 else "rot {}°".format(90 * k) if s == 1 else "s={}, rot {}°".format(s, 90 * k)
        rows = []
        for i in range(len(files)):
            row = [_img(tau(x[i:i + 1], s, k)[0])]
            for v in vae_names:
                dec_in, dec_lat = fig2[tname][v]
                row += [_img(dec_in[i]), _img(dec_lat[i])]
            rows.append(row)
        titles = ["tau∘x ({})".format(label)]
        for v in vae_names:
            titles += ["{}\nD(E(tau∘x))".format(v), "{}\nD(tau∘E(x))".format(v)]
        _grid(rows, titles, out / "fig2_{}.png".format(tname))
    return files, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vaes", required=True, help="comma-separated aliases, dirs or HF ids")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    names = [v for v in args.vaes.split(",") if v]
    files, stats = run(names, args.data, args.out, args.n_images, args.seed, args.device)
    result = {
        "exp": args.exp, "commit": git_commit(), "config": "configs/trunk.yaml",
        "eval_set": "ImageNet val, {} fixed images (sorted index i*N/{})".format(len(files), len(files)),
        "images": [str(Path(f).relative_to(args.data)) for f in files],
        "metrics": stats,
        "baseline": "qualitative: paper Fig 1 and Fig 2 (s=0.5)",
        "outputs": {"figures": ["fig1_latent_pca.png", "fig2_scale05.png", "fig2_rot90.png"]},
    }
    (Path(args.out) / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({v: {t: (d["mean_psnr_D_E_tau_x"], d["mean_psnr_D_tau_E_x"]) for t, d in s.items()}
                      for v, s in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
