"""Build results_bundle.zip: summaries, figures, grids, raw grid samples, KEY_NUMBERS.md.

    .venv/bin/python tools/bundle.py <repo root> <staging dir>
"""
import json, os, shutil, struct, sys, zipfile, zlib
import numpy as np
import matplotlib.image as mpimg

ROOT = sys.argv[1]
STAGE = sys.argv[2]
R = os.path.join(ROOT, "results")
RUNS = {"imbalanced": ["mc-fmnist-dcgan-r1", "mc-fmnist-vae-ddpm-r1"],
        "balanced": ["mc-fmnist-dcgan-balanced-r1", "mc-fmnist-vae-balanced-r1"]}
LOOPS = ["replace", "mixture", "accumulate", "accumulate_sizematch"]
LABEL = {"replace": "Replace", "mixture": "Mixture", "accumulate": "Accumulate",
         "accumulate_sizematch": "Size-matched accumulate"}
GENS = [0, 5, 10]

shutil.rmtree(STAGE, ignore_errors=True)
B = os.path.join(STAGE, "results_bundle")
os.makedirs(B)


def png_gray(arr, path):
    h, w = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def first8(pt):
    """First 8 images of a uint8 [N,1,32,32] tensor saved by torch.save (raw contiguous storage)."""
    with zipfile.ZipFile(pt) as z:
        name = [n for n in z.namelist() if n.endswith("/data/0")][0]
        with z.open(name) as f:
            buf = f.read(8 * 32 * 32)
    return np.frombuffer(buf, np.uint8).reshape(8, 32, 32)


def name_of(arch, loop, seed):
    return f"{arch}_accumulate_seed{seed}_sizematch" if loop == "accumulate_sizematch" else f"{arch}_{loop}_seed{seed}"


# 1. summaries (tables, rules, results, class-loss files, figures)
shutil.copytree(os.path.join(R, "summary"), os.path.join(B, "summary"))

# 2. grids and raw samples behind the seed-0 grids
n_checked = 0
for setting, runs in RUNS.items():
    gdst = os.path.join(B, "grids", setting)
    os.makedirs(gdst, exist_ok=True)
    for run in runs:
        out = os.path.join(R, run, "out")
        for g in sorted(os.listdir(os.path.join(out, "grids"))):
            shutil.copy(os.path.join(out, "grids", g), gdst)
        cfg = json.load(open(os.path.join(out, "config.json")))
        for arch in cfg["archs"]:
            seeds = cfg.get("arch_seeds", {}).get(arch, cfg["seeds"])
            loops = cfg.get("arch_loops", {}).get(arch, cfg["loops"])
            if 0 not in seeds:
                continue
            for loop in loops:
                nm = name_of(arch, loop, 0)
                for n in GENS:
                    pt = (os.path.join(out, "gen0", f"{arch}_seed0", "samples.pt") if n == 0 else
                          os.path.join(out, "lineages", nm, f"gen{n}", "samples.pt"))
                    x = first8(pt)
                    grid = mpimg.imread(os.path.join(out, "grids", f"{nm}_gen{n}.png"))
                    grid = np.round(grid[..., 0] * 255).astype(np.uint8)
                    for k in range(8):
                        crop = grid[2:34, 2 + 34 * k: 34 + 34 * k]
                        assert np.array_equal(crop, x[k]), (pt, k)
                    d = os.path.join(B, "samples_raw", setting, f"{arch}_{loop}_gen{n}")
                    os.makedirs(d, exist_ok=True)
                    for k in range(8):
                        png_gray(x[k], os.path.join(d, f"img{k}.png"))
                    n_checked += 1
print("raw folders written and matched to grids:", n_checked)

# 3. KEY_NUMBERS.md
L = []
A = L.append


def ms(c, d=2):
    if c is None:
        return "-"
    if c["n"] > 1:
        return f"{c['mean']:.{d}f} ± {c['se']:.{d}f}"
    return f"{c['mean']:.{d}f}"


def load_sum(sub, arch):
    p = os.path.join(R, "summary", sub, arch) if sub else os.path.join(R, "summary", arch)
    return json.load(open(os.path.join(p, "summary.json"))), json.load(open(os.path.join(p, "bootstrap.json")))


def seeds_n(s, loop):
    return s[loop]["0"]["fid"]["n"]


def metric_tables(sub, arch, title):
    s, b = load_sum(sub, arch)
    loops = [l for l in LOOPS if l in s]
    A(f"\n### {title}\n")
    A("| loop | seeds | FID gen 0 | FID gen 10 | recall gen 0 | recall gen 10 | LPIPS gen 0 | LPIPS gen 10 | KL gen 0 | KL gen 10 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for l in loops:
        g0, g10 = s[l]["0"], s[l]["10"]
        A(f"| {LABEL[l]} | {seeds_n(s, l)} | {ms(g0['fid'], 1)} | {ms(g10['fid'], 1)} | {ms(g0['recall'], 3)} | "
          f"{ms(g10['recall'], 3)} | {ms(g0['lpips'], 3)} | {ms(g10['lpips'], 3)} | {ms(g0['kl'], 3)} | {ms(g10['kl'], 3)} |")
    A("\nFID change gen 0 to 10 (per-seed change, bootstrap 10,000 resamples over seeds, descriptive):\n")
    A("| loop | seeds | per-seed change | mean | bootstrap 95% interval |")
    A("|---|---|---|---|---|")
    for l in loops:
        x = b[l]
        ci = f"[{x['ci95'][0]:+.1f}, {x['ci95'][1]:+.1f}]" if x["n_seeds"] > 1 else "n/a (1 seed)"
        A(f"| {LABEL[l]} | {x['n_seeds']} | {', '.join(f'{v:+.1f}' for v in x['per_seed'])} | {x['mean']:+.1f} | {ci} |")
    A("\nRare-class share (real 0.167 imbalanced, 0.500 balanced) and classifier reliability at gen 10:\n")
    A("| loop | rare share gen 0 | rare share gen 10 | rare share gen 10 (filtered) | conf gen 10 | reject frac gen 10 | NN agreement gen 10 |")
    A("|---|---|---|---|---|---|---|")
    for l in loops:
        g0, g10 = s[l]["0"], s[l]["10"]
        A(f"| {LABEL[l]} | {ms(g0['rare_share'], 3)} | {ms(g10['rare_share'], 3)} | {ms(g10['rare_share_filtered'], 3)} | "
          f"{ms(g10['clf_mean_conf'], 3)} | {ms(g10['reject_frac'], 3)} | {ms(g10['clf_nn_agreement'], 3)} |")
    A(f"\nGen 0 reliability: conf {ms(s[loops[0]]['0']['clf_mean_conf'], 3)}, reject {ms(s[loops[0]]['0']['reject_frac'], 3)}, "
      f"NN agreement {ms(s[loops[0]]['0']['clf_nn_agreement'], 3)}.")


def class_table(arch, loop):
    c = json.load(open(os.path.join(R, "summary", arch, f"class_loss_{loop}_gen10_compare.json")))
    meta = json.load(open(os.path.join(R, "mc-fmnist-dcgan-r1", "out", "data", "real_meta.json")))
    A(f"\n#### {arch.upper()} {LABEL[loop].lower()}, gen 10 (imbalanced seeds {c['seeds']}, balanced seeds {c['compare']['seeds']})\n")
    A("| class | imbalanced real share | imbalanced ratio | imbalanced loss | balanced real share | balanced ratio | balanced loss |")
    A("|---|---|---|---|---|---|---|")
    for i in meta["common"] + meta["rare"]:
        cc = c["compare"]
        A(f"| {c['classes'][i]}{' (rare group)' if i in meta['rare'] else ''} | {c['real_share'][i]:.3f} | "
          f"{c['ratio_mean'][i]:.2f} ± {c['ratio_se'][i]:.2f} | {c['loss_vs_real'][i]:+.2f} | {cc['real_share'][i]:.3f} | "
          f"{cc['ratio_mean'][i]:.2f} ± {cc['ratio_se'][i]:.2f} | {cc['loss_vs_real'][i]:+.2f} |")
    rare = meta["rare"]
    li = np.mean([c["loss_vs_real"][i] for i in rare]); lb = np.mean([c["compare"]["loss_vs_real"][i] for i in rare])
    A(f"\nRare-group mean loss: imbalanced {li:.3f}, balanced {lb:.3f}.")


clf = {k: json.load(open(os.path.join(R, r, "out", "classifier", "classifier.json")))
       for k, r in [("DCGAN runs", "mc-fmnist-dcgan-r1"), ("VAE and DDPM runs", "mc-fmnist-vae-ddpm-r1")]}

A("# Key numbers: model collapse on Fashion-MNIST")
A("\nAll values are mean ± standard error over seeds (seed count given). A value with no ± comes from one seed. "
  "Bootstrap intervals are descriptive, not significance tests. Generated from the files in `summary/`.")

A("\n## 1. Settings\n")
A("| setting | value |\n|---|---|")
for k, v in [
    ("Dataset", "Fashion-MNIST, 28x28 grayscale padded to 32x32, scaled to [-1, 1]"),
    ("Imbalanced real set", "5 common classes x 1000 (T-shirt/top, Pullover, Coat, Shirt, Bag) + 5 rare x 200 (Trouser, Dress, Sandal, Sneaker, Ankle boot) = 6000, ratio 5:1, data seed 0"),
    ("Balanced control real set", "600 per class = 6000"),
    ("Samples per generation", "6000"),
    ("Mixture R_fix", "3000 real images, fixed across generations"),
    ("Generations", "10 (gen 0 trained on real data, shared by all loops of an arch and seed)"),
    ("Epochs (same for every generation)", "DCGAN 80, VAE 200, DDPM 200"),
    ("DDPM sampler", "compact UNet, cosine schedule, 50-step DDIM, EMA weights"),
    ("Seeds", "DCGAN 3 (all 4 loops), VAE 3 (all 4 loops), DDPM 1 (seed 0, replace only); balanced controls 3 seeds (DCGAN replace; VAE replace and mixture)"),
    ("Metrics", "FID (Inception-v3 2048-d, grayscale repeated to 3 channels, resized to 299), recall (kNN k=3, Inception space), LPIPS diversity (500 random pairs), class KL vs real"),
    ("Reliability checks", "mean max-softmax confidence, rejection fraction (conf < 0.5), confidence-filtered KL and shares, NN agreement in classifier feature space"),
    ("Compute", "Kaggle, 2x T4 per kernel"),
]:
    A(f"| {k} | {v} |")

A("\n## 2. Classifier\n")
A("Trained on the full balanced Fashion-MNIST training set (60k), 15 epochs. Within each architecture every run uses byte-identical weights.\n")
A("| used by | test accuracy (official test set) | real-test conf | real-test reject | real-test NN agreement |")
A("|---|---|---|---|---|")
for k, c in clf.items():
    A(f"| {k} | {c['test_acc']*100:.2f}% | {c['real_test_mean_conf']:.3f} | {c['real_test_reject_frac']:.3f} | {c['real_test_nn_agreement']:.3f} |")

A("\n## 3. Main runs (imbalanced real set)")
metric_tables(None, "dcgan", "DCGAN")
metric_tables(None, "vae", "VAE")
metric_tables(None, "ddpm", "DDPM (seed 0, replace only)")
A("\n**DDPM LPIPS note.** LPIPS diversity rises from 0.317 to 0.402 over 10 generations of replace. This is not a gain in "
  "real diversity: over the same span recall falls from 0.648 to 0.069 and FID rises from 8.0 to 80.1. The grids show growing "
  "noise and artefacts, and LPIPS scores these as differences between images. So LPIPS goes up because quality drops, "
  "not because the samples cover more of the data.")

A("\n## 4. Balanced control runs (600 per class)")
A("\nGen-0 FID is measured against the balanced real set, so absolute FID is not comparable with the main runs. FID change is.")
metric_tables("balanced", "dcgan", "DCGAN balanced (replace)")
metric_tables("balanced", "vae", "VAE balanced (replace, mixture)")

A("\n## 5. Q3: rarity or class property?\n")
A("Rules were committed before each balanced run (`summary/dcgan/q3_decision_rule.md`, `summary/vae/q3_decision_rule.md`). "
  "loss = 1 - (generated share / real share) at gen 10, unfiltered labels. L = mean loss over the rare-group classes. "
  "f = (L_imbalanced - L_balanced) / (L_imbalanced - common-group level). f >= 0.75: rarity; f <= 0.25: class property; else partial.\n")
A("| arch, loop | L imbalanced | common level | L balanced | f | f per seed | outcome |")
A("|---|---|---|---|---|---|---|")
A("| DCGAN replace | 0.40 | -0.08 | 0.145 | 0.53 | 0.65, 0.51, 0.43 | partial |")
A("| VAE mixture | 0.40 | -0.08 | 0.20 | 0.42 | 0.43, 0.45, 0.38 | partial |")
A("| VAE replace | 0.97 | -0.19 | 0.74 | 0.19 | 0.16, 0.23, 0.19 | class property (near ceiling, see VAE q3_result.md) |")
A("\nChecks: loss vs gen 0 gives DCGAN partial, VAE mixture f = 0.58 (partial), VAE replace f = 0.23. "
  "Own-anchor f: DCGAN 0.47, VAE mixture 0.33, VAE replace 0.13. All 3 seeds fall in the same band in every case.")
A("\nPer-class split, both architectures: Trouser, Sneaker and Ankle boot recover when balanced (rarity drives their loss). "
  "Sandal and Dress lose share either way (class property). DCGAN also loses Pullover either way; VAE also loses Coat and Shirt either way.")
class_table("dcgan", "replace")
class_table("vae", "mixture")
class_table("vae", "replace")

A("\n## 6. Stanford Dogs baseline (earlier study, DCGAN, 3 seeds, for comparison)\n")
A("FID change gen 0 to 10: replace +111.4, mixture +37.6, accumulate +21.1. Fashion-MNIST DCGAN keeps the same order "
  "(replace +83.6, mixture +13.4, accumulate +10.6). The VAE does not: its mixture (+28.2) holds better than accumulate (+38.3).")

A("\n## 7. Compute ledger (Kaggle GPU quota, week resetting 2026-10-17)\n")
A("| kernel | contents | wall time |\n|---|---|---|")
for k, c, t in [("smoke and checks", "smoke test, parallel checks", "1.16 h"),
                ("mc-fmnist-dcgan-r1", "DCGAN, 3 seeds, 4 loops", "4.90 h"),
                ("mc-fmnist-timing2", "VAE and DDPM timing, DDPM gen 0 and gen 1", "1.38 h"),
                ("mc-fmnist-vae-ddpm-r1", "VAE 3 seeds 4 loops; DDPM seed 0 replace", "8.42 h"),
                ("mc-fmnist-dcgan-balanced-r1", "DCGAN balanced, replace, 3 seeds", "0.97 h"),
                ("mc-fmnist-vae-balanced-r1", "VAE balanced, replace and mixture, 3 seeds", "1.87 h")]:
    A(f"| {k} | {c} | {t} |")
A("\nWall times sum to 18.70 h. Kaggle reports **18.52 h used of 30 h** (11.48 h left). The weekly cap set at the start was 26 h. Every run finished with 0 failures.")
open(os.path.join(B, "KEY_NUMBERS.md"), "w").write("\n".join(L) + "\n")

open(os.path.join(B, "README.md"), "w").write("""# results_bundle

- `KEY_NUMBERS.md`: every headline number in one place.
- `summary/<arch>/`: per-arch tables (`table.md`), mean and SE per generation (`summary.json`), bootstrap (`bootstrap.json`),
  figures (`metrics.png`, `classes.png`), class-loss tables, and for DCGAN and VAE the Q3 decision rule and result.
- `summary/balanced/<arch>/`: the same for the balanced control runs.
- `grids/imbalanced/`, `grids/balanced/`: 8-image sample grids, file name `{arch}_{loop}_seed{s}[_sizematch]_gen{n}.png`
  (all seeds; DDPM also has gen 1).
- `samples_raw/<setting>/{arch}_{loop}_gen{n}/img0..7.png`: the 8 images behind each seed-0 grid at gens 0, 5, 10, as 32x32
  grayscale PNGs taken from the saved sample tensors (checked pixel for pixel against the grids).
""")

zp = os.path.join(ROOT, "results_bundle.zip")
if os.path.exists(zp):
    os.remove(zp)
shutil.make_archive(zp[:-4], "zip", STAGE, "results_bundle")
print("wrote", zp)
