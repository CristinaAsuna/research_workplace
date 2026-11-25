"""Spatial transforms tau shared by training, evaluation and visualization.

Matches the official EQ-VAE code (train_eqvae/ldm/models/autoencoder.py): bilinear scaling with
align_corners=False first, then torch.rot90(k, dims=[-1, -2]). The same op is applied to images and
latents, so for a 256 px image and its 32 px latent, s = m/32 gives 8m px and m latent cells exactly.
"""
import torch
import torch.nn.functional as F


def scale(t, s):
    if s == 1:
        return t
    return F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False)


def rotate(t, k):
    k %= 4
    return t if k == 0 else torch.rot90(t, k=k, dims=[-1, -2])


def tau(t, s=1.0, k=0):
    """tau∘t: scale by s, then rotate by k·90°."""
    return rotate(scale(t, s), k)


def scale_choices(lo=8, hi=31):
    """Official isotropic scale set {lo/32, ..., hi/32} (0.25 .. 0.96875)."""
    return [m / 32 for m in range(lo, hi + 1)]


def sample_tau(rng, lo=8, hi=31, rot_choices=(1, 2, 3)):
    """Draw (s, k) as the official training_step does. rng is a random.Random."""
    return rng.choice(scale_choices(lo, hi)), rng.choice(list(rot_choices))
