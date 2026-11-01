"""Fashion-MNIST loading and the fixed 5 common / 5 rare imbalanced real set.

Images are stored as uint8 tensors of shape (N, 1, 32, 32): the 28x28 originals
padded by 2 pixels of black. `to_float` maps them to [-1, 1].
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Interleaved split: each group mixes garments and footwear, and the class the
# classifier confuses most (Shirt, 6) is common, so rare-class shares stay clean.
COMMON = [0, 2, 4, 6, 8]
RARE = [1, 3, 5, 7, 9]
N_COMMON = 1000
N_RARE = 200
DATA_SEED = 0  # the real set is the same for every lineage seed


def to_float(x_u8):
    return x_u8.float().div(127.5).sub(1.0)


def to_uint8(x_float):
    return ((x_float.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def _load_split(root, train):
    from torchvision.datasets import FashionMNIST
    ds = FashionMNIST(root, train=train, download=True)
    x = ds.data.unsqueeze(1)                      # (N, 1, 28, 28) uint8
    x = F.pad(x, (2, 2, 2, 2), value=0)           # (N, 1, 32, 32)
    y = ds.targets.clone().long()
    return x, y


def load_fashion_mnist(root):
    """Full balanced train (60k) and test (10k) sets, uint8 32x32."""
    xtr, ytr = _load_split(root, True)
    xte, yte = _load_split(root, False)
    return xtr, ytr, xte, yte


def build_real_set(xtr, ytr, meta_path, class_counts=None):
    """Pick the real set R by class index and save real_meta.json.

    Default is the imbalanced 1000/200 set. `class_counts` (10 ints) overrides it, for example
    600 each for the balanced control; the common/rare labels then only name the same five classes.
    """
    rng = np.random.RandomState(DATA_SEED)
    counts_wanted = class_counts or [N_COMMON if c in COMMON else N_RARE for c in range(10)]
    idx = []
    for c in range(10):
        pool = np.where(ytr.numpy() == c)[0]
        idx.extend(rng.choice(pool, counts_wanted[c], replace=False).tolist())
    idx = np.array(sorted(idx))
    x, y = xtr[idx], ytr[idx]
    counts = np.bincount(y.numpy(), minlength=10)
    dist = counts / counts.sum()
    meta = {
        "dataset": "Fashion-MNIST",
        "image_size": 32,
        "classes": CLASS_NAMES,
        "common": COMMON,
        "rare": RARE,
        "common_names": [CLASS_NAMES[c] for c in COMMON],
        "rare_names": [CLASS_NAMES[c] for c in RARE],
        "counts": counts.tolist(),
        "distribution": dist.tolist(),
        "n_total": int(counts.sum()),
        "data_seed": DATA_SEED,
        "balanced": len(set(counts.tolist())) == 1,
        "train_indices": idx.tolist(),
    }
    if meta_path:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    return x, y, meta
