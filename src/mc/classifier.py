"""Small CNN classifier for 1x32x32 Fashion-MNIST, trained on the full balanced 60k set.

It is the measuring instrument for class drift. `features` returns the 256-d
penultimate layer, used for the nearest-neighbour agreement check.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import to_float


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.MaxPool2d(2))

        self.body = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))  # 4x4
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128 * 16, 256), nn.ReLU(True))
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n_classes))

    def features(self, x):
        return self.head(self.body(x))

    def forward(self, x):
        return self.out(self.features(x))


def _augment(x):
    # random horizontal flip and +-2 pixel shift (the padding makes room for it)
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    dx, dy = torch.randint(-2, 3, (2,)).tolist()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def train_classifier(xtr_u8, ytr, xte_u8, yte, device, epochs=15, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = SmallCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xtr, ytr = xtr_u8.to(device), ytr.to(device)
    n = xtr.shape[0]
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=steps)
    log = []
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(net(_augment(to_float(xtr[j]))), ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate(net, xte_u8, yte, device)
        log.append({"epoch": ep, "test_acc": acc})
        print(f"[clf] epoch {ep} test_acc {acc:.4f}", flush=True)
    return net, log


@torch.no_grad()
def predict(net, x_u8, device, batch=1000):
    """Returns (probs, features) on CPU."""
    net.eval()
    probs, feats = [], []
    for i in range(0, x_u8.shape[0], batch):
        x = to_float(x_u8[i:i + batch].to(device))
        f = net.features(x)
        probs.append(F.softmax(net.out(f), dim=1).cpu())
        feats.append(f.cpu())
    return torch.cat(probs), torch.cat(feats)


@torch.no_grad()
def evaluate(net, x_u8, y, device):
    p, _ = predict(net, x_u8, device)
    return (p.argmax(1) == y.cpu()).float().mean().item()
