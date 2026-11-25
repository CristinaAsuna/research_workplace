"""LDM LPIPSWithDiscriminator (contperceptual.py) and the taming PatchGAN discriminator.

Generator loss = sum|x - x̂| + w_p·LPIPS (summed over pixels, averaged over the batch; logvar stays 0
because LDM does not optimise it) + kl_weight·KL + d_weight·disc_factor·(-mean D(x̂)),
with d_weight = ||∇nll|| / (||∇g|| + 1e-4), clamped to [0, 1e4], times disc_weight, taken w.r.t. the
decoder's last conv. Discriminator loss = hinge on (x, x̂.detach()).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """taming.modules.discriminator.model.NLayerDiscriminator (BatchNorm, no actnorm)."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 2, padw, bias=False),
                    nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 1, padw, bias=False),
                nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, kw, 1, padw)]
        self.main = nn.Sequential(*seq)

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def gaussian_kl(mean, logvar):
    """KL(N(mean, var) || N(0, 1)) summed over C,H,W -> (B,), as LDM DiagonalGaussianDistribution.kl()."""
    mean, logvar = mean.float(), logvar.float()
    return 0.5 * (mean ** 2 + logvar.exp() - 1.0 - logvar).flatten(1).sum(1)


class VAEGANLoss(nn.Module):
    def __init__(self, perceptual, kl_weight=1e-6, disc_weight=0.5, disc_start=0, perceptual_weight=1.0,
                 disc_factor=1.0):
        super().__init__()
        self.perceptual = perceptual  # callable (x, x̂) -> (B,1,1,1); frozen
        self.kl_weight, self.disc_weight, self.disc_start = kl_weight, disc_weight, disc_start
        self.perceptual_weight, self.disc_factor = perceptual_weight, disc_factor
        self.discriminator = NLayerDiscriminator()
        self.discriminator.apply(weights_init)

    def _factor(self, step):
        return self.disc_factor if step >= self.disc_start else 0.0

    def adaptive_weight(self, nll, g, last_layer):
        nll_g = torch.autograd.grad(nll, last_layer, retain_graph=True)[0]
        g_g = torch.autograd.grad(g, last_layer, retain_graph=True)[0]
        w = torch.norm(nll_g) / (torch.norm(g_g) + 1e-4)
        return torch.clamp(w, 0.0, 1e4).detach() * self.disc_weight

    def generator_loss(self, x, rec, mean, logvar, step, last_layer):
        x, rec = x.float(), rec.float()
        rec_l = (x - rec).abs()
        if self.perceptual_weight > 0:
            rec_l = rec_l + self.perceptual_weight * self.perceptual(x, rec).float()
        nll = rec_l.sum() / rec_l.shape[0]
        kl = gaussian_kl(mean, logvar).mean()
        g = -self.discriminator(rec).float().mean()
        factor = self._factor(step)
        d_w = self.adaptive_weight(nll, g, last_layer) if factor > 0 else torch.zeros((), device=x.device)
        loss = nll + self.kl_weight * kl + d_w * factor * g
        logs = {"nll": float(nll.detach()), "kl": float(kl.detach()), "g": float(g.detach()), "d_weight": float(d_w),
                "l1": float((x - rec).detach().abs().mean())}
        return loss, logs

    def discriminator_loss(self, x, rec, step):
        real = self.discriminator(x.float().detach())
        fake = self.discriminator(rec.float().detach())
        loss = self._factor(step) * hinge_d_loss(real, fake)
        return loss, {"d_loss": float(loss.detach()), "logits_real": float(real.detach().mean()), "logits_fake": float(fake.detach().mean())}
