"""Recursive training loops with per-generation checkpointing and resume.

D_n for n >= 1:
    replace:               S_{n-1}
    mixture:               S_{n-1} + R_fix (same real slice every generation)
    accumulate:            S_0 + ... + S_{n-1}
    accumulate_sizematch:  accumulate pool, uniformly subsampled to |S| each generation

Generation 0 is trained on R once per (arch, seed) and shared by every loop.
Every generation trains a fresh model from random initialisation for the same
number of epochs. A generation counts as done when its samples are on disk and
its metrics are in the lineage JSON, so a rerun skips it.
"""
import json
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image

from .data import to_float
from .models import ARCHS

LOOPS = ["replace", "mixture", "accumulate", "accumulate_sizematch"]


def lineage_name(arch, loop, seed):
    if loop == "accumulate_sizematch":
        return f"{arch}_accumulate_seed{seed}_sizematch"
    return f"{arch}_{loop}_seed{seed}"


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s % (2 ** 32))


def _fmt(m):
    def g(k):
        v = m.get(k, 0)
        return 0.0 if v is None else v
    return (f"fid {g('fid'):.2f} recall {g('recall'):.3f} lpips {g('lpips'):.3f} "
            f"kl {g('kl'):.3f} conf {g('clf_mean_conf'):.3f} reject {g('reject_frac'):.3f} "
            f"nn_agree {g('clf_nn_agreement'):.3f} rare {g('rare_share'):.3f}")


class Runner:
    def __init__(self, cfg, out, device, evaluator, real_u8, mix_idx, timer):
        self.cfg = cfg
        self.out = out
        self.device = device
        self.ev = evaluator
        self.real_u8 = real_u8
        self.mix_idx = mix_idx
        self.timer = timer
        for d in ["gen0", "lineages", "metrics", "grids"]:
            os.makedirs(os.path.join(out, d), exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def _grid(self, x_u8, name, gen):
        if gen in self.cfg.get("grid_gens", [0, 5, 10]):
            save_image(to_float(x_u8[:8]).add(1).div(2),
                       os.path.join(self.out, "grids", f"{name}_gen{gen}.png"), nrow=8, padding=2)

    def _train_and_sample(self, arch, data_u8, seed_val, probe=False, tag=""):
        _seed_all(seed_val)
        model = ARCHS[arch](self.device)
        model.tag = tag
        epochs = self.cfg["epochs"][arch]
        t0 = time.time()
        probe_fn, probe_eps = None, ()
        pe = self.cfg.get("probe_epochs") or ()
        if isinstance(pe, dict):  # per-arch probe epochs
            pe = pe.get(arch, ())
        if probe and pe:
            n_probe = self.cfg.get("probe_n", 2000)
            probe_fn = lambda m: {"probe_fid": self.ev.quick_fid(m.sample(n_probe))}
            probe_eps = tuple(pe)
        log = model.fit(data_u8.to(self.device), epochs, probe=probe_fn, probe_epochs=probe_eps)
        t_train = time.time() - t0
        t0 = time.time()
        samples = model.sample(self.cfg["n_samples"])
        t_sample = time.time() - t0
        return model, samples, log, t_train, t_sample

    def _evaluate(self, samples, seed_val):
        t0 = time.time()
        m = self.ev(samples, seed=seed_val)
        return m, time.time() - t0

    # ---------------------------------------------------------------- gen 0
    def gen0(self, arch, seed):
        d = os.path.join(self.out, "gen0", f"{arch}_seed{seed}")
        os.makedirs(d, exist_ok=True)
        spath, mpath = os.path.join(d, "samples.pt"), os.path.join(d, "metrics.json")
        if os.path.exists(spath) and os.path.exists(mpath):
            print(f"[gen0] {arch} seed{seed}: found, skipping", flush=True)
            return torch.load(spath), _load_json(mpath)
        print(f"[gen0] {arch} seed{seed}: training on {self.real_u8.shape[0]} real images", flush=True)
        sv = seed * 1000
        model, samples, log, t_tr, t_sa = self._train_and_sample(arch, self.real_u8, sv, probe=True, tag=f"{arch}_gen0_seed{seed}")
        m, t_ev = self._evaluate(samples, sv)
        torch.save(model.state_dict(), os.path.join(d, "model.pt"))
        torch.save(samples, spath)
        _save_json(log, os.path.join(d, "train_log.json"))
        _save_json(m, mpath)
        self.timer(dict(lineage=f"{arch}_gen0_seed{seed}", gen=0, n_train=int(self.real_u8.shape[0]),
                        train_s=t_tr, sample_s=t_sa, metrics_s=t_ev, total_s=t_tr + t_sa + t_ev))
        print(f"[gen0] {arch} seed{seed}: {_fmt(m)} | train {t_tr:.0f}s eval {t_ev:.0f}s", flush=True)
        return samples, m

    # ---------------------------------------------------------------- lineage
    def _train_set(self, loop, n, seed, lin_dir, s0):
        def s(k):
            return s0 if k == 0 else torch.load(os.path.join(lin_dir, f"gen{k}", "samples.pt"))
        if loop == "replace":
            return s(n - 1)
        if loop == "mixture":
            return torch.cat([s(n - 1), self.real_u8[self.mix_idx]])
        pool = torch.cat([s(k) for k in range(n)])
        if loop == "accumulate":
            return pool
        if loop == "accumulate_sizematch":
            g = torch.Generator().manual_seed(seed * 1000 + n + 7)
            idx = torch.randperm(pool.shape[0], generator=g)[: self.cfg["n_samples"]]
            return pool[idx]
        raise ValueError(loop)

    def lineage(self, arch, loop, seed, s0, m0, deadline):
        name = lineage_name(arch, loop, seed)
        lin_dir = os.path.join(self.out, "lineages", name)
        mpath = os.path.join(self.out, "metrics", f"{name}.json")
        metrics = _load_json(mpath)
        metrics["0"] = m0
        _save_json(metrics, mpath)
        self._grid(s0, name, 0)
        for n in range(1, self.cfg["generations"] + 1):
            gdir = os.path.join(lin_dir, f"gen{n}")
            if str(n) in metrics and os.path.exists(os.path.join(gdir, "samples.pt")):
                continue
            if time.time() > deadline:
                print(f"[stop] time budget reached before {name} gen{n}; rerun to resume", flush=True)
                return False
            data = self._train_set(loop, n, seed, lin_dir, s0)
            sv = seed * 1000 + n
            model, samples, log, t_tr, t_sa = self._train_and_sample(arch, data, sv, tag=f"{name} gen{n}")
            m, t_ev = self._evaluate(samples, sv)
            m["n_train"] = int(data.shape[0])
            os.makedirs(gdir, exist_ok=True)
            torch.save(samples, os.path.join(gdir, "samples.pt"))
            _save_json(log, os.path.join(gdir, "train_log.json"))
            # only the latest model is kept: resume needs samples, not weights
            torch.save(model.state_dict(), os.path.join(lin_dir, "model_latest.pt"))
            metrics[str(n)] = m
            _save_json(metrics, mpath)
            self._grid(samples, name, n)
            self.timer(dict(lineage=name, gen=n, n_train=int(data.shape[0]), train_s=t_tr,
                            sample_s=t_sa, metrics_s=t_ev, total_s=t_tr + t_sa + t_ev))
            print(f"[{name}] gen{n} (train n={data.shape[0]}): {_fmt(m)} | "
                  f"train {t_tr:.0f}s eval {t_ev:.0f}s", flush=True)
        return True
