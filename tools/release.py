"""Stage the public results release (Kaggle dataset): results_bundle plus the raw metric files, no tensors.

    python tools/release.py <staging dir>

Contents: results_bundle.zip unpacked (summaries, figures, grids, raw grid samples, KEY_NUMBERS.md), and for each
main and balanced kernel its per-lineage metric JSONs, real_reference.json, classifier.json, real_meta.json and
config.json, plus results/summary/classifier/topk.json. Writes dataset-metadata.json for `kaggle datasets create`.
"""
import json, os, shutil, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["mc-fmnist-dcgan-r1", "mc-fmnist-vae-ddpm-r1", "mc-fmnist-dcgan-balanced-r1", "mc-fmnist-vae-balanced-r1"]
SLUG = "musalandouvhada/model-collapse-fashion-mnist-results"

dst = sys.argv[1]
shutil.rmtree(dst, ignore_errors=True)
os.makedirs(dst)
with zipfile.ZipFile(os.path.join(ROOT, "results_bundle.zip")) as z:
    z.extractall(dst)
B = os.path.join(dst, "results_bundle")
for run in RUNS:
    out = os.path.join(ROOT, "results", run, "out")
    r = os.path.join(B, "runs", run)
    shutil.copytree(os.path.join(out, "metrics"), os.path.join(r, "metrics"))
    for f in ["classifier/classifier.json", "data/real_meta.json", "config.json"]:
        shutil.copy(os.path.join(out, f), os.path.join(r, os.path.basename(f)))
shutil.copy(os.path.join(ROOT, "results", "summary", "classifier", "topk.json"), os.path.join(B, "classifier_topk.json"))
shutil.copy(os.path.join(ROOT, "report", "numbers", "ledger.md"), os.path.join(B, "ledger.md"))
bad = [os.path.join(d, f) for d, _, fs in os.walk(dst) for f in fs if f.endswith((".pt", ".pth", ".npy", ".npz"))]
assert not bad, bad
with open(os.path.join(B, "README.md"), "a") as f:
    f.write("- `runs/<kernel>/metrics/`: one metrics JSON per lineage, `{arch}_{loop}_seed{s}[_sizematch].json`, keyed by\n"
            "  generation (fid, recall, lpips, kl, kl_filtered, reject_frac, clf_mean_conf, clf_nn_agreement, clf_nn_dist,\n"
            "  class_dist, common_share, rare_share and filtered shares), plus `real_reference.json` (disjoint real set).\n"
            "  `classifier.json`, `real_meta.json` (class choice and real class distribution) and `config.json` sit beside it.\n"
            "- `classifier_topk.json`: top-1 and top-5 accuracy of the two classifier copies on the official test set.\n"
            "- `ledger.md`: every number in the final report with the file it comes from. `KEY_NUMBERS.md` was written\n"
            "  before the report's number audits; where the two differ, `ledger.md` is the checked version.\n\n"
            "No model weights or sample tensors are included. Kernels: `mc-fmnist-dcgan-r1` (DCGAN, imbalanced),\n"
            "`mc-fmnist-vae-ddpm-r1` (VAE and DDPM, imbalanced), `mc-fmnist-*-balanced-r1` (balanced controls).\n\n"
            "Source: M. Ndouvhada, \"Model Collapse in Recursively Trained Image Generators: Training Loops and Rare-Class Erasure on Fashion-MNIST\", Honours research report,\n"
            "University of the Witwatersrand, 2026. Fashion-MNIST (Xiao et al., 2017) is MIT licensed.\n")
json.dump({"title": "Model collapse in recursive generators: results", "id": SLUG,
           "licenses": [{"name": "CC-BY-4.0"}],
           "subtitle": "Metrics and sample grids for DCGAN, VAE, DDPM retrained on own samples",
           "description": open(os.path.join(B, "README.md")).read()},
          open(os.path.join(dst, "dataset-metadata.json"), "w"), indent=1)
print(sum(len(fs) for _, _, fs in os.walk(dst)), "files staged in", dst)
