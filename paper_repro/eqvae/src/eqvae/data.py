"""Image lists and datasets. The contract is in docs/DATA.md."""
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpeg", ".jpg", ".png"}


def list_images(root):
    """All images under an ImageFolder root (class dirs may be symlinks), sorted by relative path."""
    root = Path(root)
    out = []
    for dirpath, _, names in os.walk(root, followlinks=True):
        for name in names:
            if os.path.splitext(name)[1].lower() in IMG_EXT:
                out.append(os.path.join(dirpath, name))
    out.sort(key=lambda p: os.path.relpath(p, root))
    return [Path(p) for p in out]


def even_subset(files, n):
    """n files spread evenly over the sorted list (all of them when n >= len)."""
    if n >= len(files):
        return list(files)
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]


def vis_subset(files, n=8):
    """Fixed visualization images: sorted index 0, N/n, 2N/n, ... (DATA.md)."""
    step = len(files) // n
    return [files[i * step] for i in range(n)]


class EvalImages(Dataset):
    """Official eval.py protocol: Resize(256) -> CenterCrop(256) -> [-1, 1]."""

    def __init__(self, files, size=256):
        from torchvision import transforms as T
        self.files = list(files)
        self.tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                             T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            return self.tf(im.convert("RGB")), i


class TrainImages(Dataset):
    """Official OpenImages loader applied to ImageNet: random square crop of side
    min_side·U(0.5, 1), resized to size with cv2.INTER_AREA, no flip, range [-1, 1]."""

    def __init__(self, files, size=256):
        self.files = list(files)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import cv2
        rng = random  # reseeded per worker by worker_init
        with Image.open(self.files[i]) as im:
            img = np.asarray(im.convert("RGB"))
        h, w = img.shape[:2]
        side = max(1, int(min(h, w) * rng.uniform(0.5, 1.0)))
        top, left = rng.randint(0, h - side), rng.randint(0, w - side)
        crop = img[top:top + side, left:left + side]
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 127.5 - 1.0
        return x


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
