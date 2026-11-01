"""Top-1 and top-5 accuracy of the saved classifier weights on the official Fashion-MNIST test set (CPU).

    python tools/clf_topk.py [--data <dir for torchvision download>]

Loads results/<run>/out/classifier/clf.pt for the two classifier copies (the DCGAN kernel's and the
VAE/DDPM kernel's; the balanced kernels reuse them) and writes results/summary/classifier/topk.json.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mc.classifier import SmallCNN, predict  # noqa: E402
from mc.data import _load_split  # noqa: E402

RUNS = ["mc-fmnist-dcgan-r1", "mc-fmnist-vae-ddpm-r1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "FashionMNIST_data"))
    a = ap.parse_args()
    xte, yte = _load_split(a.data, train=False)
    out = {}
    for run in RUNS:
        path = os.path.join(ROOT, "results", run, "out", "classifier", "clf.pt")
        net = SmallCNN()
        sd = torch.load(path, map_location="cpu")
        net.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd.state_dict())
        p, _ = predict(net, xte, "cpu")
        top5 = p.topk(5, dim=1).indices
        out[run] = {"n_test": int(yte.shape[0]),
                    "top1": (top5[:, 0] == yte).float().mean().item(),
                    "top5": (top5 == yte[:, None]).any(1).float().mean().item(),
                    "weights": f"results/{run}/out/classifier/clf.pt"}
        print(run, out[run], flush=True)
    dst = os.path.join(ROOT, "results", "summary", "classifier")
    os.makedirs(dst, exist_ok=True)
    json.dump(out, open(os.path.join(dst, "topk.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
