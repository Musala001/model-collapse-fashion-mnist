"""Aggregate per-lineage metrics over seeds, bootstrap the FID change, draw figures.

    .venv/bin/python tools/aggregate.py results/mc-fmnist-dcgan-r1 [more result dirs] [--out results/summary]

Reads <dir>/out/metrics/{arch}_{loop}_seed{s}[_sizematch].json and <dir>/out/data/real_meta.json.
Writes, per arch: summary.json (mean and standard error per loop and generation), table.md,
bootstrap.json, and figures (metrics.png, classes.png).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

LOOPS = ["replace", "mixture", "accumulate", "accumulate_sizematch"]
LOOP_LABEL = {"replace": "Replace", "mixture": "Mixture", "accumulate": "Accumulate",
              "accumulate_sizematch": "Accumulate (size-matched)"}
METRICS = ["fid", "recall", "lpips", "kl", "kl_filtered", "reject_frac", "clf_mean_conf",
           "clf_nn_agreement", "clf_nn_dist", "common_share", "rare_share",
           "common_share_filtered", "rare_share_filtered"]
# categorical slots 1-4 of the reference palette (light), plus line style and marker as a second cue
STYLE = {"replace": ("#2a78d6", "-", "o"), "mixture": ("#eb6834", "--", "s"),
         "accumulate": ("#1baf7a", "-.", "^"), "accumulate_sizematch": ("#eda100", ":", "D")}
NAME_RE = re.compile(r"^(?P<arch>[a-z0-9]+)_(?P<loop>replace|mixture|accumulate)_seed(?P<seed>\d+)(?P<sm>_sizematch)?\.json$")
N_BOOT = 10_000


def load(dirs):
    """runs[arch][loop][seed] = {gen: metrics}; also returns real_meta from the first dir that has it."""
    runs, meta = {}, None
    for d in dirs:
        mp = os.path.join(d, "out", "data", "real_meta.json")
        if meta is None and os.path.exists(mp):
            meta = json.load(open(mp))
        for p in sorted(glob.glob(os.path.join(d, "out", "metrics", "*.json"))):
            m = NAME_RE.match(os.path.basename(p))
            if not m:
                continue
            loop = m["loop"] + ("_sizematch" if m["sm"] else "")
            data = {int(g): v for g, v in json.load(open(p)).items()}
            runs.setdefault(m["arch"], {}).setdefault(loop, {})[int(m["seed"])] = data
    return runs, meta


def mean_se(vals):
    v = np.asarray(vals, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(se), int(len(v))


def summarise(loops):
    """out[loop][gen][metric] = {mean, se, n}; only generations present for every seed."""
    out = {}
    for loop, seeds in loops.items():
        gens = sorted(set.intersection(*[set(s) for s in seeds.values()]))
        out[loop] = {}
        for g in gens:
            row = {}
            for k in METRICS:
                vals = [seeds[s][g].get(k) for s in seeds if seeds[s][g].get(k) is not None]
                if vals:
                    row[k] = dict(zip(("mean", "se", "n"), mean_se(vals)))
            cd = [seeds[s][g]["class_dist"] for s in seeds if "class_dist" in seeds[s][g]]
            if cd:
                row["class_dist"] = np.mean(cd, axis=0).round(5).tolist()
            out[loop][g] = row
    return out


def bootstrap(loops, rng):
    """Per-seed FID change (last common gen minus gen 0), resampled over seeds. Descriptive only."""
    out = {}
    for loop, seeds in loops.items():
        g_last = max(set.intersection(*[set(s) for s in seeds.values()]))
        d = np.array([seeds[s][g_last]["fid"] - seeds[s][0]["fid"] for s in sorted(seeds)])
        boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
        out[loop] = dict(gen_last=g_last, n_seeds=len(d), per_seed=d.round(3).tolist(),
                         mean=float(d.mean()), ci95=np.percentile(boots, [2.5, 97.5]).round(3).tolist(),
                         n_boot=N_BOOT)
    return out


def table(summ, boot):
    cols = ["fid", "recall", "lpips", "kl", "rare_share", "rare_share_filtered", "clf_mean_conf",
            "reject_frac", "clf_nn_agreement"]
    lines = []
    for loop in [l for l in LOOPS if l in summ]:
        gens = sorted(summ[loop])
        show = sorted({gens[0], gens[len(gens) // 2], gens[-1]})
        lines.append(f"\n### {LOOP_LABEL[loop]}\n")
        lines.append("| gen | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for g in show:
            cells = []
            for k in cols:
                c = summ[loop][g].get(k)
                cells.append("-" if c is None else
                             f"{c['mean']:.3f} ± {c['se']:.3f}" if c["n"] > 1 else f"{c['mean']:.3f}")
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        b = boot[loop]
        lines.append(f"\nFID change gen 0 to {b['gen_last']}: mean {b['mean']:+.2f}, "
                     f"bootstrap 95% interval [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] "
                     f"over {b['n_seeds']} seeds (descriptive).")
    return "\n".join(lines) + "\n"


def figures(summ, meta, arch, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    panels = [("fid", "FID"), ("recall", "Recall"), ("lpips", "LPIPS diversity"), ("kl", "Class KL"),
              ("rare_share", "Rare-class share"), ("rare_share_filtered", "Rare-class share (filtered)"),
              ("clf_mean_conf", "Classifier confidence"), ("clf_nn_agreement", "NN agreement")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    loops = [l for l in LOOPS if l in summ]
    for ax, (k, title) in zip(axes.flat, panels):
        for loop in loops:
            gens = sorted(g for g in summ[loop] if k in summ[loop][g])
            if not gens:
                continue
            m = np.array([summ[loop][g][k]["mean"] for g in gens])
            se = np.nan_to_num(np.array([summ[loop][g][k]["se"] for g in gens]))
            col, ls, mk = STYLE[loop]
            ax.fill_between(gens, m - se, m + se, color=col, alpha=0.15, linewidth=0)
            ax.plot(gens, m, color=col, ls=ls, marker=mk, ms=4, lw=2, label=LOOP_LABEL[loop])
        if k.startswith("rare_share") and meta:
            real = sum(meta["distribution"][i] for i in meta["rare"])
            ax.axhline(real, color="#52514e", lw=1, ls=(0, (2, 2)))
            ax.annotate("real", (0, real), xytext=(2, 3), textcoords="offset points",
                        fontsize=8, color="#52514e")
        ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(loops), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"{arch.upper()}: mean over seeds, band = ±1 standard error", y=0.945,
                 fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out_dir, "metrics.png"), dpi=160)
    plt.close(fig)

    if not meta:
        return
    # per-class share at the last common generation, one group per class, rare classes marked
    names = meta["classes"]
    order = meta["common"] + meta["rare"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.8 / (len(loops) + 1)
    x = np.arange(len(order))
    real = np.array(meta["distribution"])[order]
    ax.bar(x - 0.4 + w / 2, real, w * 0.92, color="#8a8984", label="Real")
    for i, loop in enumerate(loops):
        g = max(summ[loop])
        cd = np.array(summ[loop][g].get("class_dist", [np.nan] * len(names)))[order]
        ax.bar(x - 0.4 + w * (i + 1.5), cd, w * 0.92, color=STYLE[loop][0], label=f"{LOOP_LABEL[loop]} (gen {g})")
    ax.set_xticks(x, [names[i] + ("\n(rare)" if i in meta["rare"] else "") for i in order], fontsize=8)
    ax.axvline(len(meta["common"]) - 0.5, color="#8a8984", lw=0.8)
    ax.set_ylabel("Share of samples")
    ax.set_title(f"{arch.upper()}: class distribution of generated samples (classifier labels)",
                 fontsize=10, color="#0b0b0b", loc="left")
    ax.legend(frameon=False, ncol=len(loops) + 1, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "classes.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/summary")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    runs, meta = load(a.dirs)
    rng = np.random.default_rng(0)
    for arch, loops in runs.items():
        d = os.path.join(a.out, arch)
        os.makedirs(d, exist_ok=True)
        summ, boot = summarise(loops), bootstrap(loops, rng)
        json.dump({l: {str(g): v for g, v in s.items()} for l, s in summ.items()},
                  open(os.path.join(d, "summary.json"), "w"), indent=1)
        json.dump(boot, open(os.path.join(d, "bootstrap.json"), "w"), indent=1)
        seeds = {l: sorted(s) for l, s in loops.items()}
        md = f"# {arch.upper()} summary\n\nSources: {', '.join(a.dirs)}\nSeeds per loop: {seeds}\n" \
             f"Values are mean ± standard error over seeds.\n" + table(summ, boot)
        open(os.path.join(d, "table.md"), "w").write(md)
        if not a.no_figures:
            figures(summ, meta, arch, d)
        print(f"[{arch}] wrote {d}")
        print(md)


if __name__ == "__main__":
    main()
