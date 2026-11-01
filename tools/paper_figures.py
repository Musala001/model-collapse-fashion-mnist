"""Figures for the final report, drawn at their printed size (IEEE column 3.5 in, full width 7.16 in).

    .venv/bin/python tools/paper_figures.py [--out report/figures]

Reads only raw outputs: results/<run>/out/metrics/*.json, results/<run>/out/grids/*.png,
results/<run>/out/metrics/real_reference.json, and the Stanford Dogs pilot metrics in _output_vae.
Curves are the mean over seeds with a band of one standard error. Colour marks the loop, and
line style and marker repeat it so the figures still read in greyscale.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RUN = {"dcgan": "mc-fmnist-dcgan-r1", "vae": "mc-fmnist-vae-ddpm-r1", "ddpm": "mc-fmnist-vae-ddpm-r1",
       "dcgan_bal": "mc-fmnist-dcgan-balanced-r1", "vae_bal": "mc-fmnist-vae-balanced-r1"}
LOOPS = ["replace", "mixture", "accumulate", "accumulate_sizematch"]
LABEL = {"replace": "Replace", "mixture": "Mixture", "accumulate": "Accumulate",
         "accumulate_sizematch": "Size-matched accumulate"}
# categorical slots 1-4 (validated: adjacent CVD dE >= 9.1 on white); style and marker are the second cue
STYLE = {"replace": ("#2a78d6", "-", "o"), "mixture": ("#eb6834", "--", "s"),
         "accumulate": ("#1baf7a", "-.", "^"), "accumulate_sizematch": ("#eda100", ":", "D")}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
RARE = [1, 3, 5, 7, 9]
GENS = list(range(11))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 3,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def lineages(run, arch, loop):
    pat = f"{arch}_accumulate_seed*_sizematch.json" if loop == "accumulate_sizematch" else f"{arch}_{loop}_seed?.json"
    out = []
    for p in sorted(glob.glob(os.path.join(RES, run, "out", "metrics", pat))):
        if loop == "accumulate" and p.endswith("_sizematch.json"):
            continue
        out.append({int(k): v for k, v in json.load(open(p)).items()})
    return out


def mean_se(L, key, gens=GENS):
    a = np.array([[d[g][key] for g in gens] for d in L], float)
    m = a.mean(0)
    se = a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros_like(m)
    return m, se


def draw_loop(ax, L, key, loop, lw=1.1, ms=2.6):
    m, se = mean_se(L, key)
    col, ls, mk = STYLE[loop]
    if len(L) > 1:
        ax.fill_between(GENS, m - se, m + se, color=col, alpha=0.18, linewidth=0)
    ax.plot(GENS, m, color=col, ls=ls, lw=lw, marker=mk, ms=ms, markeredgewidth=0.4,
            markeredgecolor="white", label=LABEL[loop], zorder=3)


def gen_axis(ax, label=True):
    ax.set_xlim(-0.4, 10.4)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    if label:
        ax.set_xlabel("Generation", labelpad=1.5)


def fig_metrics(out):
    """3 rows (DCGAN, VAE, DDPM) x 4 metrics. DDPM is one seed, replace loop only."""
    ref = json.load(open(os.path.join(RES, RUN["dcgan"], "out", "metrics", "real_reference.json")))
    real_rare = 1 / 6  # 5 rare classes x 200 / 6000
    cols = [("fid", "FID (lower is better)"), ("recall", "Recall"), ("lpips", "LPIPS diversity"),
            ("rare_share", "Rare-class share")]
    rows = [("dcgan", "DCGAN, 3 seeds"), ("vae", "VAE, 3 seeds"), ("ddpm", "DDPM, 1 seed, replace only")]
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 4.55), sharex=True)
    for r, (arch, rlabel) in enumerate(rows):
        loops = ["replace"] if arch == "ddpm" else LOOPS
        data = {l: lineages(RUN[arch], arch, l) for l in loops}
        for c, (key, title) in enumerate(cols):
            ax = axes[r, c]
            for l in loops:
                draw_loop(ax, data[l], key, l)
            if key == "lpips":
                ax.axhline(ref["lpips"], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
            if key == "rare_share":
                ax.axhline(real_rare, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(rlabel, color=INK, fontsize=7, labelpad=3)
            gen_axis(ax, label=(r == 2))
    # shared y-limits within a column would hide the DCGAN detail, so each panel keeps its own scale
    handles = [Line2D([], [], color=STYLE[l][0], ls=STYLE[l][1], marker=STYLE[l][2], lw=1.1, ms=3,
                      markeredgecolor="white", markeredgewidth=0.4, label=LABEL[l]) for l in LOOPS]
    handles.append(Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)), label="Real data (LPIPS 0.328; rare share 0.167)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.035),
               handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(h_pad=0.6, w_pad=0.9)
    fig.savefig(os.path.join(out, "fig_metrics.pdf"))
    plt.close(fig)


def grid_strip(run, name, gen, scale=6, n=8):
    im = mpimg.imread(os.path.join(RES, run, "out", "grids", f"{name}_gen{gen}.png"))
    if im.ndim == 3:
        im = im[..., 0]
    im = im[:, :n * 34 + 2]  # grid cells are 32 px images with a 2 px border; keep the first n
    return np.kron(im, np.ones((scale, scale)))  # integer upsampling keeps pixels sharp in the PDF


def fig_samples(out):
    rows = [("dcgan", "replace", "DCGAN replace"), ("dcgan", "mixture", "DCGAN mixture"),
            ("dcgan", "accumulate", "DCGAN accumulate"), ("dcgan", "accumulate_sizematch", "DCGAN size-matched"),
            ("vae", "replace", "VAE replace"), ("vae", "mixture", "VAE mixture"),
            ("vae", "accumulate", "VAE accumulate"), ("vae", "accumulate_sizematch", "VAE size-matched"),
            ("ddpm", "replace", "DDPM replace")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.16, 4.9),
                             gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for i, (arch, loop, lab) in enumerate(rows):
        name = f"{arch}_accumulate_seed0_sizematch" if loop == "accumulate_sizematch" else f"{arch}_{loop}_seed0"
        for j, g in enumerate([0, 5, 10]):
            ax = axes[i, j]
            ax.imshow(grid_strip(RUN[arch], name, g, n=4), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"Generation {g}", loc="center", fontsize=7)
            if j == 0:
                ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5, color=INK, labelpad=3)
        if i in (3, 7):
            pass
    fig.savefig(os.path.join(out, "fig_samples.pdf"), dpi=300)
    plt.close(fig)


def fig_reliability(out):
    """Classifier reliability per generation: Stanford Dogs pilot (VAE) vs Fashion-MNIST (VAE and DCGAN)."""
    base_v = os.path.join(ROOT, "_output_vae", "model_collapse", "metrics")
    dogs = {}
    for loop, fname, suf in [("replace", "vae_synthetic", ""), ("mixture", "vae_mixture", ""),
                             ("accumulate", "vae_accumulate", ""), ("accumulate_sizematch", "vae_accumulate", "_sizematch")]:
        dogs[loop] = [{int(k): v for k, v in json.load(open(os.path.join(base_v, f"{fname}_seed{s}{suf}.json"))).items()}
                      for s in range(3)]
    fm = {arch: {l: lineages(RUN[arch], arch, l) for l in LOOPS} for arch in ["vae", "dcgan"]}
    clf = json.load(open(os.path.join(RES, RUN["vae"], "out", "classifier", "classifier.json")))
    series = [("Stanford Dogs pilot, VAE", dogs, "#eb6834"),
              ("Fashion-MNIST, VAE", fm["vae"], "#2a78d6"),
              ("Fashion-MNIST, DCGAN", fm["dcgan"], "#1baf7a")]
    panels = [("clf_mean_conf", "Mean max-softmax confidence", "real_test_mean_conf"),
              ("reject_frac", "Rejection fraction (confidence < 0.5)", "real_test_reject_frac"),
              ("clf_nn_agreement", "Nearest-neighbour agreement", "real_test_nn_agreement")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95))
    for ax, (key, title, rkey) in zip(axes, panels):
        for lab, data, col in series:
            for l in LOOPS:
                m, _ = mean_se(data[l], key)
                ax.plot(GENS, m, color=col, lw=1.4 if l == "replace" else 0.7,
                        ls="-" if l == "replace" else (0, (2.5, 1.5)), alpha=1 if l == "replace" else 0.85, zorder=3)
        ax.axhline(clf[rkey], color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
        if key == "clf_mean_conf":
            ax.axhline(0.1, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title)
        gen_axis(ax)
    handles = [Line2D([], [], color=c, lw=1.4, label=lab) for lab, _, c in series]
    handles += [Line2D([], [], color=INK2, lw=1.4, label="Replace loop"),
                Line2D([], [], color=INK2, lw=0.7, ls=(0, (2.5, 1.5)), label="Mixture, accumulate, size-matched"),
                Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.5)), label="Fashion-MNIST real test images"),
                Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 2)), label="Chance (0.1)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.17),
               handlelength=2.4, columnspacing=1.2, fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(out, "fig_reliability.pdf"))
    plt.close(fig)


def ratios(run, arch, loop):
    meta = json.load(open(os.path.join(RES, run, "out", "data", "real_meta.json")))
    real = np.array(meta["distribution"])
    r = np.array([np.array(d[10]["class_dist"]) / real for d in lineages(run, arch, loop)])
    return r.mean(0), r.std(0, ddof=1) / np.sqrt(len(r))


def fig_classes(out):
    """Per-class ratio (gen-10 share / real share), imbalanced vs balanced real set, mean ± SE over 3 seeds."""
    cases = [("dcgan", "replace", "DCGAN replace"), ("vae", "mixture", "VAE mixture"), ("vae", "replace", "VAE replace")]
    order = [1, 7, 9, 3, 5, 0, 2, 4, 6, 8]  # rare group first: Trouser, Sneaker, Ankle boot, Dress, Sandal
    short = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
             7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
    cap = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    x = np.arange(10)
    w = 0.38
    for ax, (arch, loop, title) in zip(axes, cases):
        mi, si = ratios(RUN[arch], arch, loop)
        mb, sb = ratios(RUN[arch + "_bal"], arch, loop)
        ax.axvspan(-0.5, 4.5, color="#f2f1ec", zorder=0, lw=0)
        for k, (m, s, col, hatch, dx, lab) in enumerate([(mi, si, "#2a78d6", None, -w / 2, "Imbalanced real set"),
                                                        (mb, sb, "#eb6834", "//////", w / 2, "Balanced real set")]):
            vals = m[order]
            errs = s[order]
            shown = np.minimum(vals, cap)
            ax.bar(x + dx, shown, w * 0.9, color=col if hatch is None else "white", edgecolor=col,
                   hatch=hatch, linewidth=0.6, zorder=3, label=lab)
            ax.errorbar(x + dx, shown, yerr=np.where(vals > cap, 0, errs), fmt="none", ecolor=INK2,
                        elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=4)
            for xi, v in zip(x + dx, vals):
                if v > cap:  # clipped bar: break mark across its top and the true value above it
                    xs = [xi - w * 0.55, xi + w * 0.55, xi + w * 0.55, xi - w * 0.55]
                    ys = [cap - 0.30, cap - 0.20, cap - 0.10, cap - 0.20]
                    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True, facecolor="white", edgecolor="none", zorder=5))
                    ax.plot(xs[:2], ys[:2], color=INK, lw=0.6, zorder=6)
                    ax.plot(xs[2:], ys[2:], color=INK, lw=0.6, zorder=6)
                    ax.annotate(f"{v:.1f}", (xi, cap), xytext=(0, 1), textcoords="offset points",
                                ha="center", va="bottom", fontsize=5.5, color=INK)
        ax.axhline(1.0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.set_xticks(x, [short[i] for i in order], rotation=50, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, 9.6)
        ax.set_ylim(0, cap + 0.55)
        ax.set_yticks(np.arange(0, cap + 0.01, 0.5))
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.text(2.0, cap + 0.42, "rare group", ha="center", va="center", fontsize=6, color=INK2)
        ax.text(7.0, cap + 0.42, "common group", ha="center", va="center", fontsize=6, color=INK2)
    axes[0].set_ylabel("Share at gen 10 / real share")
    handles = [Patch(facecolor="#2a78d6", edgecolor="#2a78d6", label="Imbalanced real set (1000 / 200 per class)"),
               Patch(facecolor="white", edgecolor="#eb6834", hatch="//////", label="Balanced real set (600 per class)"),
               Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)), label="Keeps its real share (ratio 1)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08),
               fontsize=6.5, handlelength=2.0)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(out, "fig_classes.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_metrics(a.out)
    fig_samples(a.out)
    fig_reliability(a.out)
    fig_classes(a.out)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
