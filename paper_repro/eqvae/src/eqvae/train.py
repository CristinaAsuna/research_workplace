"""EQ-VAE fine-tuning of SD-VAE, single GPU with gradient accumulation (docs/PIPELINE.md §2, D-004).

python -m eqvae.train --exp configs/exp/2026-11-25-eqvae-ft-1ep.yaml --data $LAB_DATA_DIR/imagenet/train \
    --init-ckpt $LAB_MODELS_DIR/kl-f8/model.ckpt [--smoke --max-steps 200]

Per micro-batch one branch is drawn as in the official training_step:
  with prob p_eq     : tau = scale s∈{8/32..31/32} then rot90 k∈{1,2,3}; loss on (tau∘x, D(tau∘E(x)))
  otherwise          : with prob p_prior_s reconstruct x downscaled by s, else x at full resolution.
Generator and discriminator see the same pair (official Lightning code may redraw; see D-004).
Outputs in runs/<exp>/ (or scratch/smoke/<exp>/): checkpoints/last.pt, train_log.jsonl, vae/.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from . import artifacts
from .config import TRUNK, git_commit, load_config
from .data import TrainImages, list_images, worker_init
from .losses import VAEGANLoss
from .transforms import sample_tau, scale, scale_choices, tau


class Trainer:
    def __init__(self, vae, perceptual, tcfg, device, seed=0):
        self.vae, self.device, self.cfg = vae.to(device).train(), device, tcfg
        self.loss = VAEGANLoss(perceptual, tcfg["kl_weight"], tcfg["disc_weight"], tcfg["disc_start"],
                               tcfg["perceptual_weight"]).to(device)
        eff_batch = tcfg["micro_batch"] * tcfg["grad_accum"]
        self.lr = tcfg["base_lr"] * eff_batch  # LDM main.py scale_lr: accumulate * ngpu * bs * base_lr
        betas = tuple(tcfg["betas"])
        self.opt_ae = torch.optim.Adam(self.vae.parameters(), lr=self.lr, betas=betas)
        self.opt_d = torch.optim.Adam(self.loss.discriminator.parameters(), lr=self.lr, betas=betas)
        self.rng = random.Random(seed)
        self.amp = tcfg.get("bf16", False) and str(device).startswith("cuda")
        self.step_idx = 0

    def draw_branch(self):
        c = self.cfg
        lo, hi = c["scale_choices_32"]
        if self.rng.random() < c["p_eq"]:
            s, k = sample_tau(self.rng, lo, hi, c["rot_choices"])
            return "eq", s, k
        if self.rng.random() < c["p_prior_s"]:
            return "lowres", self.rng.choice(scale_choices(lo, hi)), 0
        return "full", 1.0, 0

    def _micro(self, x, branch, accum):
        mode, s, k = branch
        if mode == "lowres":
            x = scale(x, s)
        disc = self.loss.discriminator
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            post = self.vae.encode(x).latent_dist
            z = post.sample()
            if mode == "eq":
                z, target = tau(z, s, k), tau(x, s, k)
            else:
                target = x
            rec = self.vae.decode(z).sample
            disc.requires_grad_(False)
            g_loss, logs = self.loss.generator_loss(target, rec, post.mean, post.logvar, self.step_idx,
                                                    self.vae.decoder.conv_out.weight)
        (g_loss / accum).backward()
        disc.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            d_loss, dlogs = self.loss.discriminator_loss(target, rec.detach(), self.step_idx)
        (d_loss / accum).backward()
        logs.update(dlogs)
        logs["g_total"] = float(g_loss.detach())
        return logs

    def step(self, micro_batches, branches=None):
        """One optimizer step over len(micro_batches) accumulated micro-batches."""
        accum = len(micro_batches)
        self.opt_ae.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)
        logs, counts = [], {"eq": 0, "lowres": 0, "full": 0}
        for i, x in enumerate(micro_batches):
            branch = branches[i] if branches else self.draw_branch()
            counts[branch[0]] += 1
            logs.append(self._micro(x.to(self.device, non_blocking=True), branch, accum))
        self.opt_ae.step()
        self.opt_d.step()
        self.step_idx += 1
        out = {k: sum(l[k] for l in logs) / len(logs) for k in logs[0]}
        out.update({"step": self.step_idx, "branches": counts})
        return out

    def state(self):
        return {"vae": self.vae.state_dict(), "disc": self.loss.discriminator.state_dict(),
                "opt_ae": self.opt_ae.state_dict(), "opt_d": self.opt_d.state_dict(),
                "step": self.step_idx, "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state()}

    def load_state(self, st):
        self.vae.load_state_dict(st["vae"])
        self.loss.discriminator.load_state_dict(st["disc"])
        self.opt_ae.load_state_dict(st["opt_ae"])
        self.opt_d.load_state_dict(st["opt_d"])
        self.step_idx = st["step"]
        self.rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])


def _infinite(loader):
    while True:
        for x in loader:
            yield x


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="trunk config (default configs/trunk.yaml)")
    parser.add_argument("--exp", required=True, help="configs/exp/<id>.yaml")
    parser.add_argument("--data", required=True, help="ImageFolder train root")
    parser.add_argument("--init-ckpt", default=None, help="LDM kl-f8 model.ckpt, for its discriminator weights")
    parser.add_argument("--smoke", action="store_true", help="write to scratch/smoke/<exp>/")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--grad-ckpt", action="store_true", help="enable gradient checkpointing in the VAE")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    cfg = load_config(args.config or TRUNK, exp=args.exp)
    exp_id, tcfg = cfg["exp"], cfg["train"]
    if args.micro_batch:
        tcfg["micro_batch"] = args.micro_batch
    if args.grad_accum:
        tcfg["grad_accum"] = args.grad_accum
    steps = args.max_steps or tcfg["steps"]
    run = artifacts.run_dir(exp_id, smoke=args.smoke)
    torch.manual_seed(tcfg["seed"])

    import lpips
    from .models import discriminator_state, load_vae
    vae = load_vae(tcfg["init"], "cpu")
    if args.grad_ckpt:
        vae.enable_gradient_checkpointing()
    perceptual = lpips.LPIPS(net="vgg", verbose=False).to(args.device).eval().requires_grad_(False)
    trainer = Trainer(vae, perceptual, tcfg, args.device, seed=tcfg["seed"])

    last = run / "checkpoints" / "last.pt"
    if last.exists():
        trainer.load_state(artifacts.load_checkpoint(run, "last"))
        print("resumed from", last, "at step", trainer.step_idx, flush=True)
    elif args.init_ckpt:
        disc = discriminator_state(args.init_ckpt)
        if disc:
            trainer.loss.discriminator.load_state_dict(disc)
        print("discriminator from kl-f8:", bool(disc), flush=True)

    files = list_images(args.data)
    print("train images:", len(files), "| lr", trainer.lr, "| eff batch",
          tcfg["micro_batch"] * tcfg["grad_accum"], flush=True)
    gen = torch.Generator().manual_seed(tcfg["seed"] + trainer.step_idx)
    loader = DataLoader(TrainImages(files), batch_size=tcfg["micro_batch"], num_workers=tcfg["num_workers"],
                        sampler=RandomSampler(files, generator=gen), drop_last=True, pin_memory=True,
                        worker_init_fn=worker_init, persistent_workers=tcfg["num_workers"] > 0)
    it = _infinite(loader)
    log_path = run / "train_log.jsonl"
    meta = {"exp": exp_id, "commit": git_commit(), "lr": trainer.lr, "train": tcfg, "steps": steps}
    with open(log_path, "a") as log:
        log.write(json.dumps({"meta": meta}) + "\n")
        t0 = time.time()
        while trainer.step_idx < steps:
            logs = trainer.step([next(it) for _ in range(tcfg["grad_accum"])])
            if trainer.step_idx % tcfg["log_every"] == 0 or trainer.step_idx == 1:
                logs["sec_per_step"] = (time.time() - t0) / tcfg["log_every"] if trainer.step_idx > 1 else None
                if str(args.device).startswith("cuda"):
                    logs["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
                if any(v != v for v in logs.values() if isinstance(v, float)):
                    raise FloatingPointError("NaN in losses at step {}: {}".format(trainer.step_idx, logs))
                log.write(json.dumps(logs) + "\n")
                log.flush()
                print(json.dumps(logs), flush=True)
                t0 = time.time()
            if trainer.step_idx % tcfg["save_every"] == 0 or trainer.step_idx == steps:
                artifacts.save_checkpoint(run, trainer.state())
    trainer.vae.eval().save_pretrained(run / "vae")
    print("done:", run / "vae", flush=True)


if __name__ == "__main__":
    main()
