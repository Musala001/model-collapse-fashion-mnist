"""Per-class share lost under a loop, and whether it tracks rarity or difficulty.

    .venv/bin/python tools/class_loss.py results/mc-fmnist-dcgan-r1 [--arch dcgan] [--loop replace] [--gen 10]
        [--compare results/<balanced run>] [--out results/summary]

For each class c and seed: ratio = share at gen G / real share, loss = 1 - ratio (relative to real),
and loss_vs_gen0 = 1 - share_G / share_0 (relative to the same lineage's gen 0, which removes the
generator's own starting bias). Means over seeds are correlated across the 10 classes (Pearson and
Spearman) with the real-data share and with the classifier's per-class test accuracy. n = 10 classes,
so the coefficients are descriptive. With --compare, the per-class ratio of a second run (for example
the balanced control) is shown side by side.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return dict(pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(np.corrcoef(_rank(a), _rank(b))[0, 1]))


def per_class(run_dir, arch, loop, gen):
    meta = json.load(open(os.path.join(run_dir, "out", "data", "real_meta.json")))
    real = np.array(meta["distribution"])
    ratio, loss0, seeds = [], [], []
    pat = re.compile(rf"^{arch}_{loop}_seed(\d+)\.json$")
    for p in sorted(glob.glob(os.path.join(run_dir, "out", "metrics", "*.json"))):
        m = pat.match(os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        if str(gen) not in d:
            continue
        g0, gG = np.array(d["0"]["class_dist"]), np.array(d[str(gen)]["class_dist"])
        ratio.append(gG / real)
        loss0.append(1 - gG / np.maximum(g0, 1e-9))
        seeds.append(int(m[1]))
    clf = json.load(open(os.path.join(run_dir, "out", "classifier", "classifier.json")))
    acc = np.array([clf["per_class_acc"][c] for c in meta["classes"]])
    return meta, real, np.array(ratio), np.array(loss0), acc, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arch", default="dcgan")
    ap.add_argument("--loop", default="replace")
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--compare")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()

    meta, real, ratio, loss0, acc, seeds = per_class(a.run, a.arch, a.loop, a.gen)
    names = meta["classes"]
    rare = set(meta.get("rare", []))
    loss = 1 - ratio.mean(0)
    l0 = loss0.mean(0)
    res = dict(arch=a.arch, loop=a.loop, gen=a.gen, seeds=seeds, classes=names,
               ratio_mean=ratio.mean(0).round(4).tolist(),
               ratio_se=(ratio.std(0, ddof=1) / np.sqrt(len(ratio))).round(4).tolist() if len(ratio) > 1 else None,
               loss_vs_real=loss.round(4).tolist(), loss_vs_gen0=l0.round(4).tolist(),
               real_share=real.round(4).tolist(), clf_acc=acc.round(4).tolist(),
               corr={"loss_vs_real~real_share": corr(loss, real),
                     "loss_vs_real~clf_acc": corr(loss, acc),
                     "loss_vs_gen0~real_share": corr(l0, real),
                     "loss_vs_gen0~clf_acc": corr(l0, acc),
                     "real_share~clf_acc": corr(real, acc)})

    lines = [f"# Per-class loss: {a.arch} {a.loop}, gen {a.gen} (seeds {seeds})\n",
             "ratio = generated share / real share (mean ± SE over seeds). "
             "loss vs real = 1 - ratio. loss vs gen 0 = 1 - share at gen G / share at gen 0.\n"]
    cmp = None
    if a.compare:
        cm, creal, cratio, closs0, _, cseeds = per_class(a.compare, a.arch, a.loop, a.gen)
        cmp = dict(run=a.compare, seeds=cseeds, real_share=creal.round(4).tolist(),
                   ratio_mean=cratio.mean(0).round(4).tolist(),
                   loss_vs_real=(1 - cratio.mean(0)).round(4).tolist(),
                   ratio_se=(cratio.std(0, ddof=1) / np.sqrt(len(cratio))).round(4).tolist() if len(cratio) > 1 else None,
                   loss_vs_gen0=closs0.mean(0).round(4).tolist())
        res["compare"] = cmp
    hdr = "| class | real share | clf acc | ratio | loss vs real | loss vs gen 0 |"
    sep = "|---|---|---|---|---|---|"
    if cmp:
        hdr += " real share (compare) | ratio (compare) | loss vs real (compare) | loss vs gen 0 (compare) |"
        sep += "---|---|---|---|"
    lines += [hdr, sep]
    for i in meta["common"] + meta["rare"]:
        se = f" ± {res['ratio_se'][i]:.2f}" if res["ratio_se"] else ""
        row = (f"| {names[i]}{' (rare)' if i in rare else ''} | {real[i]:.3f} | {acc[i]:.3f} | "
               f"{res['ratio_mean'][i]:.2f}{se} | {loss[i]:+.2f} | {l0[i]:+.2f} |")
        if cmp:
            cse = f" ± {cmp['ratio_se'][i]:.2f}" if cmp["ratio_se"] else ""
            row += (f" {cmp['real_share'][i]:.3f} | {cmp['ratio_mean'][i]:.2f}{cse} | "
                    f"{cmp['loss_vs_real'][i]:+.2f} | {cmp['loss_vs_gen0'][i]:+.2f} |")
        lines.append(row)
    lines.append("\nCorrelations across the 10 classes (descriptive, n = 10):\n")
    lines.append("| pair | Pearson | Spearman |\n|---|---|---|")
    for k, v in res["corr"].items():
        lines.append(f"| {k} | {v['pearson']:+.2f} | {v['spearman']:+.2f} |")
    md = "\n".join(lines) + "\n"

    d = os.path.join(a.out, a.arch)
    os.makedirs(d, exist_ok=True)
    stem = f"class_loss_{a.loop}_gen{a.gen}" + ("_compare" if cmp else "")
    json.dump(res, open(os.path.join(d, stem + ".json"), "w"), indent=1)
    open(os.path.join(d, stem + ".md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
