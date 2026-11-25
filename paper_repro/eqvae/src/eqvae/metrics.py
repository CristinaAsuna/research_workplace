"""Reconstruction metrics as in the official eval.py, plus the equivariance error of appendix C.3.

Heavy dependencies (lpips, torchmetrics, pytorch_fid) are imported lazily so tests run without them.
"""
import numpy as np
import torch

from .transforms import rotate, scale


def to_uint8(x):
    """[-1, 1] -> uint8 exactly as eval.py: clamp(127.5·x + 128, 0, 255) then truncate (≈ round)."""
    return torch.clamp(127.5 * x + 128.0, 0, 255).to(torch.uint8)


def psnr_uint8(a, b):
    """Per-image PSNR on uint8 values (MAX 255), as eval.py computes from the saved PNGs."""
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 20 * torch.log10(255.0 / mse.sqrt())


def ssim_per_image(pred, target):
    """torchmetrics SSIM with default args, called one image at a time (eval.py uses batch 1),
    so data_range is inferred per image from the raw [-1, 1] float tensors."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    return torch.stack([ssim(pred[i:i + 1].float(), target[i:i + 1].float()) for i in range(len(pred))])


class Lpips:
    """LPIPS-VGG on raw [-1, 1] tensors (eval.py uses a VGG LPIPS with the same v0.1 linear weights)."""

    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred, target):
        return self.net(pred.float(), target.float()).flatten()


class InceptionFeatures:
    """pytorch-fid InceptionV3 pool3 (2048-d). Input uint8 images, same as reading the PNGs."""

    def __init__(self, device):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def __call__(self, u8):
        return self.net(u8.float() / 255.0)[0].flatten(1).double().cpu()


def gaussian_stats(feats):
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(0), np.cov(feats, rowvar=False)


def fid(mu1, sigma1, mu2, sigma2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def relative_error(t, e):
    """||t - e||² / ||e||² per sample."""
    return ((t - e) ** 2).flatten(1).sum(1) / (e ** 2).flatten(1).sum(1)


@torch.no_grad()
def equivariance_errors(vae, x, z, rot_k=(1, 2, 3), scales=(0.25, 0.5, 0.75)):
    """Appendix C.3: ||tau∘E(x) - E(tau∘x)||² / ||E(tau∘x)||², E = posterior mean (D-003).

    z is E(x) for the batch x. Returns {"rot<k>": (B,), "scale<s>": (B,)}.
    """
    def enc(img):
        return vae.encode(img).latent_dist.mean.float()

    out = {}
    for k in rot_k:
        out["rot{}".format(k)] = relative_error(rotate(z, k), enc(rotate(x, k))).cpu()
    for s in scales:
        out["scale{}".format(s)] = relative_error(scale(z, s), enc(scale(x, s))).cpu()
    return out
