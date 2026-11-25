"""CPU tests: transforms, equivariance error, PCA, and one training step with a tiny VAE."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eqvae.config import load_config  # noqa: E402
from eqvae.losses import NLayerDiscriminator, gaussian_kl  # noqa: E402
from eqvae.metrics import equivariance_errors, psnr_uint8, relative_error, to_uint8  # noqa: E402
from eqvae.transforms import rotate, sample_tau, scale_choices, tau  # noqa: E402
from eqvae.visualize import pca_rgb  # noqa: E402


def tiny_vae():
    from diffusers import AutoencoderKL
    torch.manual_seed(0)
    return AutoencoderKL(down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
                         block_out_channels=[8, 8, 16, 16], layers_per_block=1, latent_channels=4,
                         norm_num_groups=4, sample_size=64)


def test_rot90_four_times_is_identity():
    x = torch.randn(2, 4, 32, 32)
    assert torch.equal(rotate(rotate(rotate(rotate(x, 1), 1), 1), 1), x)
    assert torch.equal(rotate(x, 4), x)


def test_image_and_latent_sizes_match_for_all_scales():
    img, lat = torch.randn(1, 3, 256, 256), torch.randn(1, 4, 32, 32)
    for s in scale_choices():
        for k in (0, 1, 2, 3):
            ti, tl = tau(img, s, k), tau(lat, s, k)
            assert ti.shape[-1] == 8 * tl.shape[-1] and ti.shape[-2] == 8 * tl.shape[-2]


def test_sample_tau_ranges():
    rng = random.Random(0)
    draws = [sample_tau(rng) for _ in range(500)]
    assert min(s for s, _ in draws) == 0.25 and max(s for s, _ in draws) == 31 / 32
    assert {k for _, k in draws} == {1, 2, 3}


def test_relative_error_and_psnr():
    e = torch.randn(3, 4, 8, 8)
    assert torch.allclose(relative_error(e, e), torch.zeros(3))
    assert torch.allclose(relative_error(torch.zeros_like(e), e), torch.ones(3))
    x = torch.rand(2, 3, 16, 16) * 2 - 1
    assert torch.isfinite(psnr_uint8(to_uint8(x), to_uint8(x))).all()


def test_fid_identical_is_zero_and_shift_adds_squared_norm():
    import numpy as np
    from eqvae.metrics import fid, gaussian_stats
    feats = np.random.default_rng(0).normal(size=(400, 8))
    mu, sigma = gaussian_stats(feats)
    assert abs(fid(mu, sigma, mu, sigma)) < 1e-6
    assert fid(mu, sigma, mu + 1.0, sigma) == pytest.approx(8.0, rel=1e-6)


def test_to_uint8_matches_eval_py():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    assert to_uint8(x).tolist() == [0, 128, 255, 255]


def test_equivariance_errors_shapes():
    vae = tiny_vae().eval()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    out = equivariance_errors(vae, x, z, rot_k=(1, 2), scales=(0.5,))
    assert set(out) == {"rot1", "rot2", "scale0.5"}
    assert all(v.shape == (2,) and torch.isfinite(v).all() for v in out.values())


def test_pca_rgb_range():
    rgb = pca_rgb(torch.randn(4, 8, 8))
    assert rgb.shape == (3, 8, 8) and rgb.min() >= 0 and rgb.max() <= 1


def test_kl_zero_for_standard_normal():
    assert torch.allclose(gaussian_kl(torch.zeros(2, 4, 3, 3), torch.zeros(2, 4, 3, 3)), torch.zeros(2))


def test_discriminator_handles_smallest_training_image():
    d = NLayerDiscriminator()
    assert d(torch.randn(2, 3, 64, 64)).shape[-1] > 0  # s=0.25 of 256 px


@pytest.mark.parametrize("branch", [("eq", 0.5, 1), ("lowres", 0.5, 0), ("full", 1.0, 0)])
def test_one_train_step(branch):
    from eqvae.train import Trainer
    tcfg = load_config()["train"]
    tcfg = dict(tcfg, micro_batch=2, grad_accum=2)

    def perceptual(a, b):  # stand-in for LPIPS-VGG: (B,1,1,1)
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)

    trainer = Trainer(tiny_vae(), perceptual, tcfg, "cpu")
    assert trainer.lr == pytest.approx(4.5e-6 * 4)
    before = trainer.vae.decoder.conv_out.weight.detach().clone()
    x = [torch.rand(2, 3, 256, 256) * 2 - 1 for _ in range(2)]
    logs = trainer.step(x, branches=[branch, branch])
    assert logs["step"] == 1 and logs["branches"][branch[0]] == 2
    assert all(v == v for v in logs.values() if isinstance(v, float))
    assert not torch.equal(before, trainer.vae.decoder.conv_out.weight)


def test_noeq_config_disables_eq_branch():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(exp=root / "configs/exp/2026-11-25-noeq-ft-1ep.yaml")
    assert cfg["train"]["p_eq"] == 0.0 and cfg["train"]["p_prior_s"] == 0.25
    assert cfg["train"]["micro_batch"] * cfg["train"]["grad_accum"] == 160
