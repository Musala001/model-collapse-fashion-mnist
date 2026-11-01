"""Every number the final report quotes, recomputed from the raw per-lineage metric files.

    .venv/bin/python tools/paper_numbers.py [--out report/numbers]

Reads only raw outputs (results/<run>/out/metrics/*.json, classifier.json, real_meta.json, run logs)
and the Stanford Dogs pilot metrics (_output_dcgan, _output_vae). Writes ledger.md and ledger.json:
one row per number, with the value as printed and the file it comes from. Means are over seeds with
the standard error (sample SD / sqrt(n)); bootstrap intervals use 10,000 resamples over seeds and are
descriptive. With 3 seeds the 95% percentile interval of the mean is the range of the per-seed values.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RUN = {"dcgan": "mc-fmnist-dcgan-r1", "vae": "mc-fmnist-vae-ddpm-r1", "ddpm": "mc-fmnist-vae-ddpm-r1",
       "dcgan_bal": "mc-fmnist-dcgan-balanced-r1", "vae_bal": "mc-fmnist-vae-balanced-r1"}
LOOPS = ["replace", "mixture", "accumulate", "accumulate_sizematch"]
N_BOOT = 10_000
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
COMMON, RARE = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]

rows = []


def rel(p):
    return os.path.relpath(p, ROOT)


def add(key, value, fmt, source, note=""):
    s = fmt.format(value) if isinstance(value, (int, float, np.floating)) else str(value)
    rows.append(dict(key=key, value=s, raw=None if not isinstance(value, (int, float, np.floating)) else float(value),
                     source=source, note=note))
    return value


def mfile(run, arch, loop, seed):
    name = f"{arch}_accumulate_seed{seed}_sizematch" if loop == "accumulate_sizematch" else f"{arch}_{loop}_seed{seed}"
    return os.path.join(RES, run, "out", "metrics", name + ".json")


def lineage(run, arch, loop, seed):
    return {int(k): v for k, v in json.load(open(mfile(run, arch, loop, seed))).items()}


def seeds_of(run, arch, loop):
    pat = re.compile(rf"^{arch}_{'accumulate' if loop == 'accumulate_sizematch' else loop}_seed(\d+)"
                     rf"{'_sizematch' if loop == 'accumulate_sizematch' else ''}\.json$")
    return sorted(int(m[1]) for p in glob.glob(os.path.join(RES, run, "out", "metrics", "*.json"))
                  if (m := pat.match(os.path.basename(p))))


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))


def boot_ci(v, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def pm(m, se, d):
    return f"{m:.{d}f}" if np.isnan(se) else f"{m:.{d}f} ± {se:.{d}f}"


DEC = {"fid": 1, "recall": 3, "lpips": 3, "kl": 3, "kl_filtered": 3, "rare_share": 3, "rare_share_filtered": 3,
       "common_share": 3, "clf_mean_conf": 3, "reject_frac": 3, "clf_nn_agreement": 3}


def main_runs():
    out = {}
    for arch in ["dcgan", "vae", "ddpm"]:
        run = RUN[arch]
        for loop in LOOPS:
            seeds = seeds_of(run, arch, loop)
            if not seeds:
                continue
            L = {s: lineage(run, arch, loop, s) for s in seeds}
            src = rel(mfile(run, arch, loop, seeds[0])).replace(f"seed{seeds[0]}", "seed{" + ",".join(map(str, seeds)) + "}")
            out[(arch, loop)] = L
            for g in [0, 1, 5, 10]:
                for k, d in DEC.items():
                    vals = [L[s][g].get(k) for s in seeds if L[s][g].get(k) is not None]
                    if not vals:
                        continue
                    m, se = mean_se(vals)
                    add(f"{arch}.{loop}.gen{g}.{k}", pm(m, se, d), "{}", src, f"n={len(vals)} seeds")
            ch = [L[s][10]["fid"] - L[s][0]["fid"] for s in seeds]
            add(f"{arch}.{loop}.fid_change.per_seed", ", ".join(f"{c:+.1f}" for c in ch), "{}", src)
            m, _ = mean_se(ch)
            add(f"{arch}.{loop}.fid_change.mean", m, "{:+.1f}", src)
            lo, hi = boot_ci(ch)
            if not np.isnan(lo):
                add(f"{arch}.{loop}.fid_change.boot95", f"[{lo:+.1f}, {hi:+.1f}]", "{}", src,
                    f"percentile bootstrap, {N_BOOT} resamples; equals [min, max] of per-seed values: "
                    f"{np.isclose(lo, min(ch)) and np.isclose(hi, max(ch))}")
            if len(seeds) > 1:
                add(f"{arch}.{loop}.fid_change.se", mean_se(ch)[1], "{:.1f}", src)
            rel_lp = [(L[s][10]["lpips"] - L[s][0]["lpips"]) / L[s][0]["lpips"] for s in seeds]
            add(f"{arch}.{loop}.lpips_rel_change.mean", 100 * np.mean(rel_lp), "{:+.0f}%", src)
            # generation at which mean FID first exceeds gen 0 by 10, and monotonicity
            mf = [np.mean([L[s][g]["fid"] for s in seeds]) for g in range(11)]
            add(f"{arch}.{loop}.fid_mean_by_gen", ", ".join(f"{x:.1f}" for x in mf), "{}", src)
            add(f"{arch}.{loop}.fid_mean_monotone_rise", bool(all(b > a for a, b in zip(mf, mf[1:]))), "{}", src)
    return out


def loop_differences(out):
    """Paired by seed (gen 0 is shared within a seed, so the FID-change difference is the gen-10 FID difference)."""
    for arch in ["dcgan", "vae"]:
        pairs = [("replace", "mixture"), ("replace", "accumulate"), ("mixture", "accumulate"),
                 ("accumulate_sizematch", "accumulate"), ("replace", "accumulate_sizematch"),
                 ("accumulate_sizematch", "mixture")]
        for a, b in pairs:
            A, B = out[(arch, a)], out[(arch, b)]
            seeds = sorted(set(A) & set(B))
            d = [(A[s][10]["fid"] - A[s][0]["fid"]) - (B[s][10]["fid"] - B[s][0]["fid"]) for s in seeds]
            src = f"results/{RUN[arch]}/out/metrics/{arch}_{{{a},{b}}}_seed{{0,1,2}}.json"
            add(f"{arch}.diff.{a}-minus-{b}.per_seed", ", ".join(f"{x:+.1f}" for x in d), "{}", src)
            add(f"{arch}.diff.{a}-minus-{b}.mean", float(np.mean(d)), "{:+.1f}", src)
            lo, hi = boot_ci(d)
            add(f"{arch}.diff.{a}-minus-{b}.boot95", f"[{lo:+.1f}, {hi:+.1f}]", "{}", src)
            add(f"{arch}.diff.{a}-minus-{b}.n_seeds_positive", int(sum(x > 0 for x in d)), "{}", src)
        # recall, rare share and KL differences mixture vs accumulate at gen 10
        for k in ["recall", "rare_share", "kl", "lpips"]:
            A, B = out[(arch, "mixture")], out[(arch, "accumulate")]
            d = [A[s][10][k] - B[s][10][k] for s in sorted(A)]
            add(f"{arch}.diff.mixture-minus-accumulate.gen10.{k}.per_seed", ", ".join(f"{x:+.3f}" for x in d), "{}",
                f"results/{RUN[arch]}/out/metrics/{arch}_{{mixture,accumulate}}_seed{{0,1,2}}.json")
        # share of accumulate's FID protection that the size-matched control keeps
        ch = {l: np.mean([out[(arch, l)][s][10]["fid"] - out[(arch, l)][s][0]["fid"] for s in out[(arch, l)]])
              for l in ["replace", "accumulate", "accumulate_sizematch"]}
        frac = (ch["replace"] - ch["accumulate_sizematch"]) / (ch["replace"] - ch["accumulate"])
        add(f"{arch}.sizematch.share_of_accumulate_protection", 100 * frac, "{:.0f}%",
            f"results/{RUN[arch]}/out/metrics/{arch}_{{replace,accumulate}}_seed*.json (incl. _sizematch)",
            "(replace - sizematch) / (replace - accumulate), mean FID change")
        per_seed = []
        for s in sorted(out[(arch, "replace")]):
            c = {l: out[(arch, l)][s][10]["fid"] - out[(arch, l)][s][0]["fid"] for l in ["replace", "accumulate", "accumulate_sizematch"]}
            per_seed.append((c["replace"] - c["accumulate_sizematch"]) / (c["replace"] - c["accumulate"]))
        add(f"{arch}.sizematch.share_of_accumulate_protection.per_seed", ", ".join(f"{100 * x:.0f}%" for x in per_seed), "{}",
            f"results/{RUN[arch]}/out/metrics/")


def training_set_sizes():
    for arch in ["dcgan", "vae"]:
        for loop in LOOPS:
            L = lineage(RUN[arch], arch, loop, 0)
            add(f"{arch}.{loop}.n_train.gen1,gen10", f"{L[1]['n_train']}, {L[10]['n_train']}", "{}", rel(mfile(RUN[arch], arch, loop, 0)))


def classifier_and_reference():
    for run in [RUN["dcgan"], RUN["vae"], RUN["dcgan_bal"], RUN["vae_bal"]]:
        c = json.load(open(os.path.join(RES, run, "out", "classifier", "classifier.json")))
        src = f"results/{run}/out/classifier/classifier.json"
        add(f"clf[{run}].test_acc", 100 * c["test_acc"], "{:.2f}%", src)
        add(f"clf[{run}].real_test_conf", c["real_test_mean_conf"], "{:.3f}", src)
        add(f"clf[{run}].real_test_reject", c["real_test_reject_frac"], "{:.3f}", src)
        add(f"clf[{run}].real_test_nn_agreement", c["real_test_nn_agreement"], "{:.3f}", src)
        add(f"clf[{run}].per_class_acc.Shirt", c["per_class_acc"]["Shirt"], "{:.3f}", src)
        accs = c["per_class_acc"]
        add(f"clf[{run}].per_class_acc.min_class", min(accs, key=accs.get), "{}", src)
        add(f"clf[{run}].per_class_acc.rare_range",
            f"{min(accs[CLASSES[i]] for i in RARE):.3f} to {max(accs[CLASSES[i]] for i in RARE):.3f}", "{}", src)
        add(f"clf[{run}].epochs", c["epochs"], "{}", src)
        topk = json.load(open(os.path.join(RES, "summary", "classifier", "topk.json")))
        w = next(r for r in topk if open(os.path.join(ROOT, topk[r]["weights"]), "rb").read()
                 == open(os.path.join(RES, run, "out", "classifier", "clf.pt"), "rb").read())
        src = f"results/summary/classifier/topk.json (tools/clf_topk.py on {topk[w]['weights']})"
        add(f"clf[{run}].test_top1_from_weights", 100 * topk[w]["top1"], "{:.2f}%", src, "official test set, n=10000")
        add(f"clf[{run}].test_top5", 100 * topk[w]["top5"], "{:.2f}%", src, "official test set, n=10000")
        r = json.load(open(os.path.join(RES, run, "out", "metrics", "real_reference.json")))
        src = f"results/{run}/out/metrics/real_reference.json"
        for k in ["fid", "recall", "lpips", "kl", "rare_share", "clf_mean_conf", "reject_frac", "clf_nn_agreement"]:
            add(f"realref[{run}].{k}", r[k], "{:.3f}" if k != "fid" else "{:.2f}", src)
    meta = json.load(open(os.path.join(RES, RUN["dcgan"], "out", "data", "real_meta.json")))
    add("real.counts", ", ".join(map(str, meta["counts"])), "{}", f"results/{RUN['dcgan']}/out/data/real_meta.json")
    add("real.common_names", ", ".join(meta["common_names"]), "{}", f"results/{RUN['dcgan']}/out/data/real_meta.json")
    add("real.rare_names", ", ".join(meta["rare_names"]), "{}", f"results/{RUN['dcgan']}/out/data/real_meta.json")
    add("real.rare_share", sum(meta["distribution"][i] for i in meta["rare"]), "{:.3f}", f"results/{RUN['dcgan']}/out/data/real_meta.json")
    mix = meta.get("mixture_indices")
    if mix:
        add("real.mixture_size", len(mix), "{}", f"results/{RUN['dcgan']}/out/data/real_meta.json")
    labels = os.environ.get("FMNIST_TRAIN_LABELS")  # optional: raw train-labels-idx1-ubyte, to get R_fix's class mix
    if mix and labels and os.path.exists(labels):
        raw = open(labels, "rb").read()
        y = np.frombuffer(raw[8:], np.uint8)
        yr = y[np.array(meta["train_indices"])]
        assert np.bincount(yr, minlength=10).tolist() == meta["counts"]
        c = np.bincount(yr[np.array(mix)], minlength=10)
        src = f"results/{RUN['dcgan']}/out/data/real_meta.json (mixture_indices) + Fashion-MNIST train labels"
        add("real.R_fix.counts", ", ".join(map(str, c)), "{}", src)
        add("real.R_fix.rare_share", c[RARE].sum() / c.sum(), "{:.3f}", src)
        add("real.R_fix.rare_class_range", f"{c[RARE].min()} to {c[RARE].max()}", "{}", src)
    bmeta = json.load(open(os.path.join(RES, RUN["dcgan_bal"], "out", "data", "real_meta.json")))
    add("real_bal.counts", ", ".join(map(str, bmeta["counts"])), "{}", f"results/{RUN['dcgan_bal']}/out/data/real_meta.json")
    for run in [RUN["dcgan"], RUN["vae"], RUN["dcgan_bal"], RUN["vae_bal"]]:
        cfg = json.load(open(os.path.join(RES, run, "out", "config.json")))
        add(f"config[{run}]", f"epochs={cfg['epochs']}, n_samples={cfg['n_samples']}, mixture_real={cfg['mixture_real']}, "
                              f"generations={cfg['generations']}, seeds={cfg['seeds']}, loops={cfg['loops']}, "
                              f"arch_seeds={cfg.get('arch_seeds')}, arch_loops={cfg.get('arch_loops')}", "{}",
            f"results/{run}/out/config.json")


def per_class(run, arch, loop, gen=10):
    meta = json.load(open(os.path.join(RES, run, "out", "data", "real_meta.json")))
    real = np.array(meta["distribution"])
    seeds = seeds_of(run, arch, loop)
    ratio, loss0 = [], []
    for s in seeds:
        L = lineage(run, arch, loop, s)
        g0, gG = np.array(L[0]["class_dist"]), np.array(L[gen]["class_dist"])
        ratio.append(gG / real)
        loss0.append(1 - gG / np.maximum(g0, 1e-9))
    return real, np.array(ratio), np.array(loss0), seeds


def q3():
    cases = [("dcgan", "replace", RUN["dcgan"], RUN["dcgan_bal"], 0.40, -0.08),
             ("vae", "mixture", RUN["vae"], RUN["vae_bal"], 0.40, -0.08),
             ("vae", "replace", RUN["vae"], RUN["vae_bal"], 0.97, -0.19)]
    for arch, loop, ri, rb, L_imb_fixed, common_fixed in cases:
        tag = f"q3.{arch}.{loop}"
        real_i, ratio_i, loss0_i, si = per_class(ri, arch, loop)
        real_b, ratio_b, loss0_b, sb = per_class(rb, arch, loop)
        src_i = f"results/{ri}/out/metrics/{arch}_{loop}_seed{{{','.join(map(str, si))}}}.json"
        src_b = f"results/{rb}/out/metrics/{arch}_{loop}_seed{{{','.join(map(str, sb))}}}.json"
        loss_i, loss_b = 1 - ratio_i.mean(0), 1 - ratio_b.mean(0)
        se_i = ratio_i.std(0, ddof=1) / np.sqrt(len(ratio_i))
        se_b = ratio_b.std(0, ddof=1) / np.sqrt(len(ratio_b))
        for c in range(10):
            add(f"{tag}.imb.ratio.{CLASSES[c]}", f"{ratio_i.mean(0)[c]:.2f} ± {se_i[c]:.2f}", "{}", src_i)
            add(f"{tag}.imb.loss.{CLASSES[c]}", loss_i[c], "{:+.2f}", src_i)
            add(f"{tag}.bal.ratio.{CLASSES[c]}", f"{ratio_b.mean(0)[c]:.2f} ± {se_b[c]:.2f}", "{}", src_b)
            add(f"{tag}.bal.loss.{CLASSES[c]}", loss_b[c], "{:+.2f}", src_b)
        L_imb = loss_i[RARE].mean()
        common_imb = loss_i[COMMON].mean()
        L_bal = loss_b[RARE].mean()
        add(f"{tag}.L_rare_imbalanced", L_imb, "{:.3f}", src_i)
        add(f"{tag}.common_level_imbalanced", common_imb, "{:+.3f}", src_i)
        add(f"{tag}.L_rare_balanced", L_bal, "{:.3f}", src_b)
        add(f"{tag}.common_level_balanced", loss_b[COMMON].mean(), "{:+.3f}", src_b)
        f = (L_imb_fixed - L_bal) / (L_imb_fixed - common_fixed)
        add(f"{tag}.f (fixed anchors {L_imb_fixed}, {common_fixed})", f, "{:.2f}", f"{src_b}; anchors from committed rule")
        add(f"{tag}.f (fixed anchors) 3dp", f, "{:.3f}", f"{src_b}; anchors from committed rule")
        f_unrounded = (L_imb - L_bal) / (L_imb - common_imb)
        add(f"{tag}.f (unrounded anchors)", f_unrounded, "{:.2f}", f"{src_i}; {src_b}")
        fs = []
        for k in range(len(ratio_b)):
            Lb_s = (1 - ratio_b[k])[RARE].mean()
            fs.append((L_imb_fixed - Lb_s) / (L_imb_fixed - common_fixed))
        add(f"{tag}.f.per_seed", ", ".join(f"{x:.2f}" for x in fs), "{}", src_b)
        fs_u = [(L_imb - (1 - ratio_b[k])[RARE].mean()) / (L_imb - common_imb) for k in range(len(ratio_b))]
        add(f"{tag}.f.per_seed (unrounded anchors)", ", ".join(f"{x:.2f}" for x in fs_u), "{}", f"{src_i}; {src_b}")
        add(f"{tag}.L_rare_balanced.per_seed", ", ".join(f"{(1 - ratio_b[k])[RARE].mean():.3f}" for k in range(len(ratio_b))), "{}", src_b)
        outcome = "rarity" if f >= 0.75 else ("class property" if f <= 0.25 else "partial")
        add(f"{tag}.outcome", outcome, "{}", "rule: f >= 0.75 rarity, f <= 0.25 class property, else partial")
        # check: loss vs gen 0
        l0_i, l0_b = loss0_i.mean(0), loss0_b.mean(0)
        f0 = (l0_i[RARE].mean() - l0_b[RARE].mean()) / (l0_i[RARE].mean() - l0_i[COMMON].mean())
        add(f"{tag}.check.f_loss_vs_gen0", f0, "{:.2f}", f"{src_i}; {src_b}",
            f"rare {l0_i[RARE].mean():.2f} -> {l0_b[RARE].mean():.2f}; common imb {l0_i[COMMON].mean():+.2f}")
        # check: balanced run's own common level as the anchor
        f_own = (L_imb - L_bal) / (L_imb - loss_b[COMMON].mean())
        add(f"{tag}.check.f_own_anchor", f_own, "{:.2f}", f"{src_i}; {src_b}", "unrounded imbalanced level; balanced common level as anchor")
        add(f"{tag}.check.f_loss_vs_gen0 3dp", f0, "{:.3f}", f"{src_i}; {src_b}")
        # correlation across classes between imbalanced and balanced loss
        r = float(np.corrcoef(loss_i, loss_b)[0, 1])
        add(f"{tag}.corr.loss_imb~loss_bal.pearson", r, "{:+.2f}", f"{src_i}; {src_b}", "n = 10 classes, descriptive")
        r_rare = float(np.corrcoef(loss_i[RARE], loss_b[RARE])[0, 1])
        add(f"{tag}.corr.loss_imb~loss_bal.rare_only.pearson", r_rare, "{:+.2f}", f"{src_i}; {src_b}", "n = 5 classes")
        meta = json.load(open(os.path.join(RES, ri, "out", "data", "real_meta.json")))
        real = np.array(meta["distribution"])
        clf = json.load(open(os.path.join(RES, ri, "out", "classifier", "classifier.json")))
        acc = np.array([clf["per_class_acc"][c] for c in CLASSES])
        add(f"{tag}.corr.loss_imb~real_share.pearson", float(np.corrcoef(loss_i, real)[0, 1]), "{:+.2f}", src_i, "n = 10")
        add(f"{tag}.corr.loss_imb~clf_acc.pearson", float(np.corrcoef(loss_i, acc)[0, 1]), "{:+.2f}", src_i, "n = 10")
        add(f"{tag}.corr.real_share~clf_acc.pearson", float(np.corrcoef(real, acc)[0, 1]), "{:+.2f}", src_i, "n = 10")
        # reliability at gen 10 in the balanced run
        for k in ["clf_mean_conf", "reject_frac", "clf_nn_agreement", "rare_share", "rare_share_filtered"]:
            v = [lineage(rb, arch, loop, s)[10][k] for s in sb]
            m, se = mean_se(v)
            add(f"{tag}.bal.gen10.{k}", pm(m, se, 3), "{}", src_b)
            if k == "clf_nn_agreement":
                add(f"{tag}.bal.gen10.{k}.range", f"{min(v):.3f} to {max(v):.3f}", "{}", src_b)
        for g in [0, 10]:
            v = [lineage(rb, arch, loop, s)[g]["fid"] for s in sb]
            add(f"{tag}.bal.gen{g}.fid", pm(*mean_se(v), 1), "{}", src_b)
        ch = [lineage(rb, arch, loop, s)[10]["fid"] - lineage(rb, arch, loop, s)[0]["fid"] for s in sb]
        add(f"{tag}.bal.fid_change.per_seed", ", ".join(f"{x:+.1f}" for x in ch), "{}", src_b)
        add(f"{tag}.bal.fid_change.mean", float(np.mean(ch)), "{:+.1f}", src_b)
        lo, hi = boot_ci(ch)
        add(f"{tag}.bal.fid_change.boot95", f"[{lo:+.1f}, {hi:+.1f}]", "{}", src_b)
        # top classes at gen 10 in the balanced run (share of samples)
        cd = np.mean([lineage(rb, arch, loop, s)[10]["class_dist"] for s in sb], axis=0)
        top = np.argsort(-cd)[:3]
        add(f"{tag}.bal.gen10.top3_classes", ", ".join(f"{CLASSES[i]} {cd[i]:.3f}" for i in top), "{}", src_b)
        add(f"{tag}.bal.gen10.top3_total", float(cd[top].sum()), "{:.2f}", src_b)
    # DDPM single-seed per-class
    real, ratio, loss0, s = per_class(RUN["ddpm"], "ddpm", "replace")
    for c in range(10):
        add(f"q3.ddpm.replace.imb.ratio.{CLASSES[c]}", ratio[0][c], "{:.2f}", f"results/{RUN['ddpm']}/out/metrics/ddpm_replace_seed0.json")


def pilot_best_val_acc(path):
    """Read 'best_val_acc' from a torch checkpoint's pickle without importing torch."""
    import io, pickletools, zipfile
    with zipfile.ZipFile(path) as z:
        data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    ops = list(pickletools.genops(io.BytesIO(data)))
    i = next(i for i, (op, arg, _) in enumerate(ops) if arg == "best_val_acc")
    return next(arg for op, arg, _ in ops[i + 1:] if op.name == "BINFLOAT")


def dogs():
    """Stanford Dogs pilot. DCGAN files use 'synthetic' for replace; reliability was only logged for VAE and DDPM."""
    base_d = os.path.join(ROOT, "_output_dcgan", "model_collapse", "metrics")
    base_v = os.path.join(ROOT, "_output_vae", "model_collapse", "metrics")
    for run, out in [("dcgan", "_output_dcgan"), ("vae_ddpm", "_output_vae")]:
        ck = f"{out}/model_collapse/classifier/resnet50_breeds.pt"
        add(f"dogs.clf[{run}].best_val_acc", 100 * pilot_best_val_acc(os.path.join(ROOT, ck)), "{:.1f}%", ck,
            "'best_val_acc' stored in the checkpoint; each pilot kernel fine-tuned its own copy")
    for loop, fname in [("replace", "synthetic"), ("mixture", "mixture"), ("accumulate", "accumulate")]:
        L = [{int(k): v for k, v in json.load(open(os.path.join(base_d, f"{fname}_seed{s}.json"))).items()} for s in range(3)]
        src = f"_output_dcgan/model_collapse/metrics/{fname}_seed{{0,1,2}}.json"
        for g in [0, 10]:
            for k in ["fid", "lpips", "recall"]:
                m, se = mean_se([x[g][k] for x in L])
                add(f"dogs.dcgan.{loop}.gen{g}.{k}", pm(m, se, 1 if k == "fid" else 3), "{}", src)
        ch = [x[10]["fid"] - x[0]["fid"] for x in L]
        add(f"dogs.dcgan.{loop}.fid_change.mean", float(np.mean(ch)), "{:+.1f}", src)
        add(f"dogs.dcgan.{loop}.fid_change.per_seed", ", ".join(f"{c:+.1f}" for c in ch), "{}", src)
        add(f"dogs.dcgan.{loop}.has_reliability_keys", "clf_mean_conf" in L[0][0], "{}", src)
    for loop, fname in [("replace", "vae_synthetic"), ("mixture", "vae_mixture"), ("accumulate", "vae_accumulate"),
                        ("accumulate_sizematch", "vae_accumulate")]:
        suffix = "_sizematch" if loop == "accumulate_sizematch" else ""
        L = [{int(k): v for k, v in json.load(open(os.path.join(base_v, f"{fname}_seed{s}{suffix}.json"))).items()} for s in range(3)]
        src = f"_output_vae/model_collapse/metrics/{fname}_seed{{0,1,2}}{suffix}.json"
        for g in [0, 10]:
            for k in ["fid", "lpips", "recall", "clf_mean_conf", "reject_frac", "clf_nn_agreement"]:
                vals = [x[g].get(k) for x in L if x[g].get(k) is not None]
                if vals:
                    m, se = mean_se(vals)
                    add(f"dogs.vae.{loop}.gen{g}.{k}", pm(m, se, 1 if k == "fid" else 3), "{}", src)
                    add(f"dogs.vae.{loop}.gen{g}.{k}.range", f"{min(vals):.3f} to {max(vals):.3f}", "{}", src)
        ch = [x[10]["fid"] - x[0]["fid"] for x in L]
        add(f"dogs.vae.{loop}.fid_change.mean", float(np.mean(ch)), "{:+.1f}", src)
    L = {int(k): v for k, v in json.load(open(os.path.join(base_v, "ddpm_synthetic_seed0.json"))).items()}
    for g in [0, 10]:
        for k in ["fid", "lpips", "clf_mean_conf", "reject_frac", "clf_nn_agreement"]:
            add(f"dogs.ddpm.replace.gen{g}.{k}", L[g][k], "{:.3f}", "_output_vae/model_collapse/metrics/ddpm_synthetic_seed0.json")
    # across all dogs VAE lineages and generations: max confidence, min rejection, max agreement
    allv = []
    for p in glob.glob(os.path.join(base_v, "vae_*.json")):
        d = json.load(open(p))
        for g, v in d.items():
            allv.append((v.get("clf_mean_conf"), v.get("reject_frac"), v.get("clf_nn_agreement")))
    allv = [a for a in allv if a[0] is not None]
    add("dogs.vae.all_lineages_all_gens.max_conf", max(a[0] for a in allv), "{:.3f}", "_output_vae/model_collapse/metrics/vae_*.json")
    add("dogs.vae.all_lineages_all_gens.min_reject", min(a[1] for a in allv), "{:.3f}", "_output_vae/model_collapse/metrics/vae_*.json")
    add("dogs.vae.all_lineages_all_gens.max_agreement", max(a[2] for a in allv), "{:.3f}", "_output_vae/model_collapse/metrics/vae_*.json")
    meta = json.load(open(os.path.join(ROOT, "_output_dcgan", "model_collapse", "data", "real_meta.json")))
    add("dogs.real_meta.keys", ", ".join(k for k in meta if k not in ("files", "paths", "indices")), "{}", "_output_dcgan/model_collapse/data/real_meta.json")
    for k in ["n_total", "counts", "breeds", "image_size"]:
        if k in meta:
            add(f"dogs.real_meta.{k}", json.dumps(meta[k]) if not isinstance(meta[k], (int, float)) else meta[k], "{}",
                "_output_dcgan/model_collapse/data/real_meta.json")


def fmnist_reliability_ranges():
    """Lowest confidence, highest rejection and lowest agreement over all Fashion-MNIST lineages and generations."""
    for arch, run in [("dcgan", RUN["dcgan"]), ("vae", RUN["vae"]), ("ddpm", RUN["ddpm"])]:
        vals = []
        for p in glob.glob(os.path.join(RES, run, "out", "metrics", f"{arch}_*.json")):
            for g, v in json.load(open(p)).items():
                vals.append((v["clf_mean_conf"], v["reject_frac"], v["clf_nn_agreement"], os.path.basename(p), g))
        add(f"fmnist.{arch}.all.min_conf", min(v[0] for v in vals), "{:.3f}", f"results/{run}/out/metrics/{arch}_*.json",
            str(min(vals, key=lambda v: v[0])[3:]))
        add(f"fmnist.{arch}.all.max_reject", max(v[1] for v in vals), "{:.3f}", f"results/{run}/out/metrics/{arch}_*.json",
            str(max(vals, key=lambda v: v[1])[3:]))
        add(f"fmnist.{arch}.all.min_agreement", min(v[2] for v in vals), "{:.3f}", f"results/{run}/out/metrics/{arch}_*.json",
            str(min(vals, key=lambda v: v[2])[3:]))


def extras():
    # reliability floor over every generation of the balanced runs
    for arch, run in [("dcgan", RUN["dcgan_bal"]), ("vae", RUN["vae_bal"])]:
        vals = []
        for p in glob.glob(os.path.join(RES, run, "out", "metrics", f"{arch}_*.json")):
            for g, v in json.load(open(p)).items():
                vals.append((v["clf_mean_conf"], v["reject_frac"], v["clf_nn_agreement"], os.path.basename(p), g))
        src = f"results/{run}/out/metrics/{arch}_*.json"
        add(f"fmnist_bal.{arch}.all.min_conf", min(v[0] for v in vals), "{:.3f}", src, str(min(vals, key=lambda v: v[0])[3:]))
        add(f"fmnist_bal.{arch}.all.max_reject", max(v[1] for v in vals), "{:.3f}", src, str(max(vals, key=lambda v: v[1])[3:]))
        add(f"fmnist_bal.{arch}.all.min_agreement", min(v[2] for v in vals), "{:.3f}", src, str(min(vals, key=lambda v: v[2])[3:]))
    # number of trained generators with metrics (gen 0 counted once per arch and seed)
    total = 0
    for run in [RUN["dcgan"], RUN["vae"], RUN["dcgan_bal"], RUN["vae_bal"]]:
        gen0 = set()
        n = 0
        for p in glob.glob(os.path.join(RES, run, "out", "metrics", "*_seed*.json")):
            d = json.load(open(p))
            arch, seed = os.path.basename(p).split("_")[0], re.search(r"seed(\d+)", p)[1]
            gen0.add((arch, seed))
            n += sum(1 for g in d if g != "0")
        add(f"generators[{run}]", n + len(gen0), "{}", f"results/{run}/out/metrics/*.json", "gens 1-10 per lineage + one shared gen 0 per arch and seed")
        total += n + len(gen0)
    add("generators.total", total, "{}", "results/*/out/metrics/*.json")
    # dogs size-matched share of accumulate's protection (VAE only; the dogs DCGAN had no size-matched arm)
    base_v = os.path.join(ROOT, "_output_vae", "model_collapse", "metrics")
    ch = {}
    for loop, fname, suf in [("replace", "vae_synthetic", ""), ("accumulate", "vae_accumulate", ""),
                             ("sizematch", "vae_accumulate", "_sizematch"), ("mixture", "vae_mixture", "")]:
        v = []
        for s in range(3):
            d = json.load(open(os.path.join(base_v, f"{fname}_seed{s}{suf}.json")))
            v.append(d["10"]["fid"] - d["0"]["fid"])
        ch[loop] = np.mean(v)
    add("dogs.vae.sizematch.share_of_accumulate_protection",
        100 * (ch["replace"] - ch["sizematch"]) / (ch["replace"] - ch["accumulate"]), "{:.0f}%",
        "_output_vae/model_collapse/metrics/vae_*_seed{0,1,2}*.json")


def erasure_timing():
    """First generation at which the seed-mean share of a class falls below half of its real share (None: never by gen 10)."""
    cases = [("dcgan", "replace", RUN["dcgan"], "imb"), ("dcgan", "replace", RUN["dcgan_bal"], "bal"),
             ("vae", "mixture", RUN["vae"], "imb"), ("vae", "mixture", RUN["vae_bal"], "bal"),
             ("vae", "replace", RUN["vae"], "imb"), ("vae", "replace", RUN["vae_bal"], "bal"),
             ("vae", "accumulate", RUN["vae"], "imb"), ("dcgan", "mixture", RUN["dcgan"], "imb"),
             ("dcgan", "accumulate", RUN["dcgan"], "imb"), ("ddpm", "replace", RUN["ddpm"], "imb")]
    for arch, loop, run, tag in cases:
        meta = json.load(open(os.path.join(RES, run, "out", "data", "real_meta.json")))
        real = np.array(meta["distribution"])
        seeds = seeds_of(run, arch, loop)
        cd = np.mean([[lineage(run, arch, loop, s)[g]["class_dist"] for g in range(11)] for s in seeds], axis=0)
        ratio = cd / real
        src = f"results/{run}/out/metrics/{arch}_{loop}_seed{{{','.join(map(str, seeds))}}}.json"
        for c in range(10):
            g = next((g for g in range(11) if ratio[g, c] < 0.5), None)
            add(f"timing.{arch}.{loop}.{tag}.first_gen_below_half.{CLASSES[c]}", "never" if g is None else g, "{}", src,
                "seed-mean share / real share < 0.5")
        add(f"timing.{arch}.{loop}.{tag}.rare_share_by_gen", ", ".join(f"{x:.3f}" for x in cd[:, RARE].sum(1)), "{}", src)


def round2():
    """Numbers added after round-1 review: gen-0 rare-group loss, filtered f, size-matched closure per metric,
    and whether a class stays below half its real share once it first crosses."""
    cases = [("dcgan", "replace", RUN["dcgan"], RUN["dcgan_bal"]), ("vae", "mixture", RUN["vae"], RUN["vae_bal"]),
             ("vae", "replace", RUN["vae"], RUN["vae_bal"])]
    for arch, loop, ri, rb in cases:
        tag = f"q3.{arch}.{loop}"
        L = {}
        for run, side in ((ri, "imb"), (rb, "bal")):
            real = np.array(json.load(open(os.path.join(RES, run, "out", "data", "real_meta.json")))["distribution"])
            seeds = seeds_of(run, arch, loop)
            src = f"results/{run}/out/metrics/{arch}_{loop}_seed{{{','.join(map(str, seeds))}}}.json"
            for g in (0, 10):
                for key in ("class_dist", "class_dist_filtered"):
                    cd = np.mean([lineage(run, arch, loop, s)[g][key] for s in seeds], axis=0)
                    loss = 1 - cd / real
                    L[(side, g, key, "rare")] = loss[RARE].mean()
                    L[(side, g, key, "common")] = loss[COMMON].mean()
                    nm = "filtered" if key.endswith("filtered") else "unfiltered"
                    add(f"{tag}.{side}.gen{g}.L_rare.{nm}", loss[RARE].mean(), "{:.3f}", src)
                    add(f"{tag}.{side}.gen{g}.common_level.{nm}", loss[COMMON].mean(), "{:+.3f}", src)
        for key, nm in (("class_dist_filtered", "filtered"),):
            Li, ci, Lb = L[("imb", 10, key, "rare")], L[("imb", 10, key, "common")], L[("bal", 10, key, "rare")]
            add(f"{tag}.check.f_{nm}", (Li - Lb) / (Li - ci), "{:.2f}", f"results/{ri}; results/{rb}",
                "filtered labels at gen 10, filtered imbalanced levels as anchors")
    # check: zero loss as the reference instead of the committed common-group level
    for arch, loop, ri, rb, Li in [("dcgan", "replace", RUN["dcgan"], RUN["dcgan_bal"], 0.40),
                                   ("vae", "mixture", RUN["vae"], RUN["vae_bal"], 0.40),
                                   ("vae", "replace", RUN["vae"], RUN["vae_bal"], 0.97)]:
        real = np.array(json.load(open(os.path.join(RES, rb, "out", "data", "real_meta.json")))["distribution"])
        seeds = seeds_of(rb, arch, loop)
        cd = np.mean([lineage(rb, arch, loop, s)[10]["class_dist"] for s in seeds], axis=0)
        Lb = (1 - cd / real)[RARE].mean()
        add(f"q3.{arch}.{loop}.check.f_zero_reference", (Li - Lb) / Li, "{:.2f}", f"results/{rb}; committed L_imb",
            "reference = zero loss instead of the imbalanced common-group level")
        add(f"q3.{arch}.{loop}.f_if_L_bal_zero", Li / (Li - {0.40: -0.08, 0.97: -0.19}[Li]), "{:.2f}", "committed rule values")
    # O1: every class present at generation 0 (lowest and highest seed-mean ratio to the real share)
    for arch in ("dcgan", "vae", "ddpm"):
        run = RUN[arch]
        real = np.array(json.load(open(os.path.join(RES, run, "out", "data", "real_meta.json")))["distribution"])
        seeds = seeds_of(run, arch, "replace")
        r = np.mean([lineage(run, arch, "replace", s)[0]["class_dist"] for s in seeds], axis=0) / real
        src = f"results/{run}/out/metrics/{arch}_replace_seed{{{','.join(map(str, seeds))}}}.json"
        add(f"gen0.{arch}.min_class_ratio", f"{CLASSES[r.argmin()]} {r.min():.2f}", "{}", src)
        add(f"gen0.{arch}.max_class_ratio", f"{CLASSES[r.argmax()]} {r.max():.2f}", "{}", src)
    # size-matched closure: share of the replace-to-accumulate gap at gen 10 that size-matched closes
    for arch in ("dcgan", "vae"):
        run = RUN[arch]
        for metric in ("fid", "recall", "lpips", "kl", "rare_share"):
            v = {lp: np.mean([lineage(run, arch, lp, s)[10][metric] for s in seeds_of(run, arch, lp)])
                 for lp in ("replace", "accumulate", "accumulate_sizematch")}
            if metric == "fid":
                v = {lp: v[lp] - np.mean([lineage(run, arch, lp, s)[0]["fid"] for s in seeds_of(run, arch, lp)]) for lp in v}
            add(f"sizematch.{arch}.closure.{metric}", (v["accumulate_sizematch"] - v["replace"]) / (v["accumulate"] - v["replace"]),
                "{:.2f}", f"results/{run}/out/metrics/{arch}_*_seed{{0,1,2}}*.json", "gen-10 seed means; fid uses the change from gen 0")
    # does a class stay below half once it first crosses?
    for arch, loop, run, tag in [("dcgan", "replace", RUN["dcgan"], "imb"), ("dcgan", "replace", RUN["dcgan_bal"], "bal"),
                                 ("vae", "mixture", RUN["vae"], "imb"), ("vae", "mixture", RUN["vae_bal"], "bal"),
                                 ("vae", "replace", RUN["vae"], "imb"), ("vae", "replace", RUN["vae_bal"], "bal"),
                                 ("ddpm", "replace", RUN["ddpm"], "imb")]:
        real = np.array(json.load(open(os.path.join(RES, run, "out", "data", "real_meta.json")))["distribution"])
        seeds = seeds_of(run, arch, loop)
        cd = np.mean([[lineage(run, arch, loop, s)[g]["class_dist"] for g in range(11)] for s in seeds], axis=0)
        ratio = cd / real
        src = f"results/{run}/out/metrics/{arch}_{loop}_seed{{{','.join(map(str, seeds))}}}.json"
        for c in range(10):
            g = next((g for g in range(11) if ratio[g, c] < 0.5), None)
            if g is not None:
                add(f"timing.{arch}.{loop}.{tag}.stays_below_half.{CLASSES[c]}", bool((ratio[g:, c] < 0.5).all()), "{}", src)
            add(f"timing.{arch}.{loop}.{tag}.ratio_by_gen.{CLASSES[c]}", ", ".join(f"{x:.2f}" for x in ratio[:, c]), "{}", src)


def seed_mean_floors():
    """Lowest seed-mean confidence and agreement, and highest seed-mean rejection, over all loops and generations."""
    for tag, specs in [("imbalanced", [("dcgan", RUN["dcgan"]), ("vae", RUN["vae"]), ("ddpm", RUN["ddpm"])]),
                       ("balanced", [("dcgan", RUN["dcgan_bal"]), ("vae", RUN["vae_bal"])]),
                       ("all", [("dcgan", RUN["dcgan"]), ("vae", RUN["vae"]), ("ddpm", RUN["ddpm"]),
                                ("dcgan", RUN["dcgan_bal"]), ("vae", RUN["vae_bal"])])]:
        best = {"clf_mean_conf": (9, None), "clf_nn_agreement": (9, None), "reject_frac": (-9, None)}
        top = {"clf_mean_conf": (-9, None)}
        for arch, run in specs:
            for loop in LOOPS:
                seeds = seeds_of(run, arch, loop)
                if not seeds:
                    continue
                L = [lineage(run, arch, loop, s) for s in seeds]
                for g in range(11):
                    for k in best:
                        m = float(np.mean([d[g][k] for d in L]))
                        where = f"{run}:{arch}_{loop} gen{g}"
                        if (k != "reject_frac" and m < best[k][0]) or (k == "reject_frac" and m > best[k][0]):
                            best[k] = (m, where)
                    m = float(np.mean([d[g]["clf_mean_conf"] for d in L]))
                    if m > top["clf_mean_conf"][0]:
                        top["clf_mean_conf"] = (m, f"{run}:{arch}_{loop} gen{g}")
        for k, (v, where) in best.items():
            add(f"floor.{tag}.seed_mean.{'max' if k == 'reject_frac' else 'min'}.{k}", v, "{:.3f}",
                "results/<runs>/out/metrics/*.json", where)
        add(f"floor.{tag}.seed_mean.max.clf_mean_conf", top["clf_mean_conf"][0], "{:.3f}",
            "results/<runs>/out/metrics/*.json", top["clf_mean_conf"][1])


def param_counts():
    """Parameter counts from the layer shapes in src/mc/models.py and classifier.py (no torch needed)."""
    def conv(ci, co, k, bias=True):
        return ci * co * k * k + (co if bias else 0)

    def bn(c):
        return 2 * c

    def lin(i, o):
        return i * o + o
    g = conv(100, 256, 4, False) + bn(256) + conv(256, 128, 4, False) + bn(128) + conv(128, 64, 4, False) + bn(64) + conv(64, 1, 4, False)
    d = conv(1, 64, 4, False) + conv(64, 128, 4, False) + bn(128) + conv(128, 256, 4, False) + bn(256) + conv(256, 1, 4, False)
    enc = conv(1, 32, 4) + bn(32) + conv(32, 64, 4) + bn(64) + conv(64, 128, 4) + bn(128) + conv(128, 256, 4) + bn(256)
    dec = conv(256, 128, 4) + bn(128) + conv(128, 64, 4) + bn(64) + conv(64, 32, 4) + bn(32) + conv(32, 1, 4)
    vae = enc + 2 * lin(1024, 128) + lin(128, 1024) + dec
    T = 256  # time-embedding width (ch * 4)

    def res(ci, co):
        return 2 * ci + conv(ci, co, 3) + lin(T, co) + 2 * co + conv(co, co, 3) + (conv(ci, co, 1) if ci != co else 0)

    def attn(c):
        return 2 * c + conv(c, 3 * c, 1) + conv(c, c, 1)
    ch, mults, attn_res = 64, (1, 2, 2), (8,)
    total = lin(ch, T) + lin(T, T) + conv(1, ch, 3)
    chans, c, r = [ch], ch, 32
    for i, m in enumerate(mults):
        for _ in range(2):
            total += res(c, ch * m) + (attn(ch * m) if r in attn_res else 0)
            c = ch * m
            chans.append(c)
        if i < len(mults) - 1:
            total += conv(c, c, 3)
            chans.append(c)
            r //= 2
    total += 2 * res(c, c) + attn(c)
    for i, m in reversed(list(enumerate(mults))):
        for _ in range(3):
            total += res(c + chans.pop(), ch * m) + (attn(ch * m) if r in attn_res else 0)
            c = ch * m
        if i > 0:
            total += conv(c, c, 3)
            r *= 2
    total += 2 * c + conv(c, 1, 3)
    clf = (conv(1, 32, 3) + bn(32) + conv(32, 32, 3) + bn(32) + conv(32, 64, 3) + bn(64) + conv(64, 64, 3) + bn(64)
           + conv(64, 128, 3) + bn(128) + conv(128, 128, 3) + bn(128) + lin(128 * 16, 256) + lin(256, 10))
    src = "src/mc/models.py, src/mc/classifier.py (layer shapes)"
    add("params.dcgan_generator", g, "{:,}", src)
    add("params.dcgan_discriminator", d, "{:,}", src)
    add("params.vae", vae, "{:,}", src)
    add("params.ddpm_unet", total, "{:,}", src, "the EMA copy doubles the stored weights but not the model size")
    add("params.classifier", clf, "{:,}", src)
    add("params.unet_over_dcgan_generator", total / g, "{:.1f}x", src)


def logs():
    for run in ["mc-fmnist-smoke", "mc-fmnist-1gpu", "mc-fmnist-par2", "mc-fmnist-dcgan-r1", "mc-fmnist-timing2",
                "mc-fmnist-vae-ddpm-r1", "mc-fmnist-dcgan-balanced-r1", "mc-fmnist-vae-balanced-r1"]:
        p = os.path.join(RES, run, f"{run}.log")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        m = re.findall(r"wall ([0-9.]+) min", txt)
        f = re.findall(r"failed=(\d+)", txt)
        if m:
            add(f"log[{run}].wall_min", float(m[-1]), "{:.1f}", f"results/{run}/{run}.log")
        if f:
            add(f"log[{run}].failed", int(f[-1]), "{}", f"results/{run}/{run}.log")
        g = re.findall(r"Tesla T4", txt)
        add(f"log[{run}].gpu", "Tesla T4 x2" if len(set(re.findall(r"GPU (\d): Tesla T4", txt))) >= 2 else "see log", "{}", f"results/{run}/{run}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "numbers"))
    a = ap.parse_args()
    out = main_runs()
    loop_differences(out)
    training_set_sizes()
    classifier_and_reference()
    q3()
    dogs()
    fmnist_reliability_ranges()
    extras()
    erasure_timing()
    param_counts()
    seed_mean_floors()
    round2()
    logs()
    walls = [r["raw"] for r in rows if r["key"].startswith("log[") and r["key"].endswith(".wall_min") and r["raw"] is not None]
    add("log.total_wall_hours", sum(walls) / 60, "{:.2f}", "results/*/*.log (every Fashion-MNIST kernel, trial runs included)",
        "T4 x2 is billed by session time, so this matches the Kaggle quota used (18.52 h by `kaggle quota`)")
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "ledger.json"), "w"), indent=1)
    lines = ["# Number ledger for the final report", "",
             "Generated by `tools/paper_numbers.py` from raw result files. Mean ± SE over seeds unless noted.", "",
             "| key | value | source | note |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['key']} | {r['value']} | `{r['source']}` | {r['note']} |")
    open(os.path.join(a.out, "ledger.md"), "w").write("\n".join(lines) + "\n")
    print(f"{len(rows)} numbers -> {a.out}")


if __name__ == "__main__":
    main()
