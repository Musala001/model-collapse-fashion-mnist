"""Generators for 1x32x32 images. Each wrapper exposes:

    fit(data_u8, epochs, probe=None, probe_epochs=())  # train from scratch
    sample(n) -> uint8 tensor (n, 1, 32, 32) on CPU
    state_dict() / load_state_dict()

`data_u8` is a uint8 tensor already on the training device. The whole set is
kept on the GPU and batches are indexed directly (no DataLoader), which matters
for the small nets used here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import to_float, to_uint8


def _report(model, rec, epochs):
    """Print the epoch record about ten times per fit, so long trainings show progress."""
    every = max(1, epochs // 10)
    if rec["epoch"] % every == 0 or rec["epoch"] == epochs:
        vals = " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "epoch" and isinstance(v, float))
        print(f"  [{getattr(model, 'tag', 'fit')}] epoch {rec['epoch']}/{epochs} {vals}", flush=True)


def _init_dcgan(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),       # 4x4
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 16x16
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),            # 32x32
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),                     # 16x16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),               # 8x8
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),           # 4x4
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),                 # 1x1 logit
        )

    def forward(self, x):
        return self.net(x).view(-1)


class DCGAN:
    """Adam 2e-4, betas (0.5, 0.999), batch 64, one-sided label smoothing (0.9)."""
    nz = 100

    def __init__(self, device, batch_size=64, lr=2e-4, real_label=0.9):
        self.device = device
        self.bs = batch_size
        self.lr = lr
        self.real_label = real_label
        self.G = DCGANGenerator(self.nz).to(device)
        self.D = DCGANDiscriminator().to(device)
        self.G.apply(_init_dcgan)
        self.D.apply(_init_dcgan)

    def fit(self, data_u8, epochs, probe=None, probe_epochs=()):
        G, D, dev = self.G, self.D, self.device
        optG = torch.optim.Adam(G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        n = data_u8.shape[0]
        log = []
        G.train(); D.train()
        for ep in range(1, epochs + 1):
            perm = torch.randperm(n, device=dev)
            dsum = gsum = 0.0
            nb = 0
            for i in range(0, n - self.bs + 1, self.bs):   # drop last partial batch
                real = to_float(data_u8[perm[i:i + self.bs]])
                b = real.shape[0]
                z = torch.randn(b, self.nz, 1, 1, device=dev)
                fake = G(z)
                optD.zero_grad(set_to_none=True)
                lossD = bce(D(real), torch.full((b,), self.real_label, device=dev)) + \
                    bce(D(fake.detach()), torch.zeros(b, device=dev))
                lossD.backward()
                optD.step()
                optG.zero_grad(set_to_none=True)
                lossG = bce(D(fake), torch.ones(b, device=dev))
                lossG.backward()
                optG.step()
                dsum += lossD.item(); gsum += lossG.item(); nb += 1
            rec = {"epoch": ep, "lossD": dsum / max(nb, 1), "lossG": gsum / max(nb, 1)}
            if probe is not None and ep in probe_epochs:
                rec.update(probe(self))
                G.train()
            _report(self, rec, epochs)
            log.append(rec)
        return log

    @torch.no_grad()
    def sample(self, n, batch=500):
        # Sample in train mode (batch statistics), as during training.
        self.G.train()
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, self.nz, 1, 1, device=self.device)
            out.append(to_uint8(self.G(z)).cpu())
        return torch.cat(out)

    def state_dict(self):
        return {"G": self.G.state_dict(), "D": self.D.state_dict()}

    def load_state_dict(self, sd):
        self.G.load_state_dict(sd["G"])
        self.D.load_state_dict(sd["D"])


class ConvVAE(nn.Module):
    """Four-layer conv encoder, 128-d latent, mirrored decoder with tanh output."""

    def __init__(self, nz=128):
        super().__init__()
        self.nz = nz
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),     # 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),    # 8
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 4
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), # 2
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4, nz)
        self.fc_lv = nn.Linear(256 * 4, nz)
        self.fc_dec = nn.Linear(nz, 256 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),  # 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),    # 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),     # 16
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),                              # 32
        )

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 256, 2, 2))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv


class VAE:
    """Adam 1e-3, batch 64, loss = summed MSE reconstruction + KL regulariser."""

    def __init__(self, device, batch_size=64, lr=1e-3):
        self.device = device
        self.bs = batch_size
        self.lr = lr
        self.net = ConvVAE().to(device)

    def fit(self, data_u8, epochs, probe=None, probe_epochs=()):
        net, dev = self.net, self.device
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        n = data_u8.shape[0]
        log = []
        net.train()
        for ep in range(1, epochs + 1):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            nb = 0
            for i in range(0, n - self.bs + 1, self.bs):
                x = to_float(data_u8[perm[i:i + self.bs]])
                xr, mu, lv = net(x)
                rec = F.mse_loss(xr, x, reduction="sum") / x.shape[0]
                kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.shape[0]
                loss = rec + kld
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
            r = {"epoch": ep, "loss": tot / max(nb, 1)}
            if probe is not None and ep in probe_epochs:
                r.update(probe(self))
                net.train()
            _report(self, r, epochs)
            log.append(r)
        return log

    @torch.no_grad()
    def sample(self, n, batch=500):
        self.net.eval()
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, self.net.nz, device=self.device)
            out.append(to_uint8(self.net.decode(z)).cpu())
        return torch.cat(out)

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["net"])


def _time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, c, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.proj(a.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Compact UNet for 1x32x32: 64-128-128 channels, two res blocks per level, attention at 8x8."""

    def __init__(self, ch=64, mults=(1, 2, 2), attn_res=(8,)):
        super().__init__()
        tdim = ch * 4
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.down, chans, c, res = nn.ModuleList(), [ch], ch, 32
        for i, m in enumerate(mults):
            for _ in range(2):
                self.down.append(nn.ModuleList([ResBlock(c, ch * m, tdim),
                                                 Attention(ch * m) if res in attn_res else nn.Identity()]))
                c = ch * m
                chans.append(c)
            if i < len(mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                chans.append(c)
                res //= 2
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, tdim), Attention(c), ResBlock(c, c, tdim)
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(3):
                self.up.append(nn.ModuleList([ResBlock(c + chans.pop(), ch * m, tdim),
                                              Attention(ch * m) if res in attn_res else nn.Identity()]))
                c = ch * m
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(c, c, 3, padding=1)))
                res *= 2
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(_time_embedding(t, self.ch))
        h = self.inc(x)
        hs = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](h, temb))
            else:
                h = layer(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, hs.pop()], 1), temb))
            else:
                h = layer(h)
        return self.out(h)


class DDPM:
    """Epsilon prediction, T=1000 cosine schedule, Adam 2e-4, batch 128, EMA 0.999,
    fp16 autocast, 50-step deterministic DDIM sampling from the EMA weights."""
    T = 1000

    def __init__(self, device, batch_size=128, lr=2e-4, ema=0.999, ddim_steps=50):
        self.device = device
        self.bs = batch_size
        self.lr = lr
        self.ema_decay = ema
        self.ddim_steps = ddim_steps
        self.net = UNet().to(device)
        self.ema = UNet().to(device)
        self.ema.load_state_dict(self.net.state_dict())
        self.ema.requires_grad_(False)
        s = 0.008
        t = torch.linspace(0, self.T, self.T + 1, dtype=torch.float64)
        f = torch.cos((t / self.T + s) / (1 + s) * math.pi / 2) ** 2
        abar = (f / f[0]).clamp(min=1e-5)
        self.abar = abar[1:].float().to(device)  # abar[t] for t = 0..T-1

    def fit(self, data_u8, epochs, probe=None, probe_epochs=()):
        net, dev = self.net, self.device
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        use_amp = dev == "cuda" or str(dev).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        n = data_u8.shape[0]
        log = []
        net.train()
        for ep in range(1, epochs + 1):
            perm = torch.randperm(n, device=dev)
            tot, nb = 0.0, 0
            for i in range(0, n - self.bs + 1, self.bs):
                x0 = to_float(data_u8[perm[i:i + self.bs]])
                t = torch.randint(0, self.T, (x0.shape[0],), device=dev)
                eps = torch.randn_like(x0)
                ab = self.abar[t][:, None, None, None]
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = F.mse_loss(net(xt, t).float(), eps)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    for pe, p in zip(self.ema.parameters(), net.parameters()):
                        pe.lerp_(p, 1 - self.ema_decay)
                tot += loss.item(); nb += 1
            r = {"epoch": ep, "loss": tot / max(nb, 1)}
            if probe is not None and ep in probe_epochs:
                r.update(probe(self))
                net.train()
            _report(self, r, epochs)
            log.append(r)
        return log

    @torch.no_grad()
    def sample(self, n, batch=500):
        self.ema.eval()
        use_amp = str(self.device).startswith("cuda")
        ts = torch.linspace(self.T - 1, 0, self.ddim_steps, device=self.device).round().long()
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            x = torch.randn(b, 1, 32, 32, device=self.device)
            for j, t in enumerate(ts):
                ab = self.abar[t]
                ab_prev = self.abar[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0, device=self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    eps = self.ema(x, t.repeat(b)).float()
                x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            out.append(to_uint8(x).cpu())
        return torch.cat(out)

    def state_dict(self):
        return {"net": self.net.state_dict(), "ema": self.ema.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["net"])
        self.ema.load_state_dict(sd["ema"])


ARCHS = {"dcgan": DCGAN, "vae": VAE, "ddpm": DDPM}
