"""Entry point: build data, classifier and evaluator, then run gen 0 and all lineages.

    python -m mc.run config.json [out_dir]                # single process
    python -m mc.run config.json out_dir --task '{...}'   # one task (used by the scheduler)

With "parallel_gpus": 2 and two visible GPUs, gen 0 and then the lineages run as
separate processes, one per GPU, so two lineages train at the same time.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

from .classifier import SmallCNN, predict, train_classifier
from .data import CLASS_NAMES, DATA_SEED, build_real_set, load_fashion_mnist
from .loops import Runner
from .metrics import CONF_THRESHOLD, Evaluator, nn_agreement


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def setup_classifier(cfg, out, xtr, ytr, xte, yte, real_u8, real_y, device):
    d = os.path.join(out, "classifier")
    os.makedirs(d, exist_ok=True)
    wpath = os.path.join(d, "clf.pt")
    net = SmallCNN().to(device)
    if os.path.exists(wpath):
        net.load_state_dict(torch.load(wpath, map_location=device))
        print("[clf] loaded existing weights", flush=True)
        return net
    t0 = time.time()
    net, log = train_classifier(xtr, ytr, xte, yte, device, epochs=cfg.get("clf_epochs", 15))
    torch.save(net.state_dict(), wpath)
    # report: accuracy, per-class accuracy, and the reliability checks on real test images
    probs, feats = predict(net, xte, device)
    pred = probs.argmax(1).numpy()
    conf = probs.max(1).values.numpy()
    y = yte.numpy()
    _, real_feats = predict(net, real_u8, device)
    agree, nnd = nn_agreement(feats, pred, real_feats, real_y, device)
    rep = {
        "test_acc": float((pred == y).mean()),
        "per_class_acc": {CLASS_NAMES[c]: float((pred[y == c] == c).mean()) for c in range(10)},
        "real_test_mean_conf": float(conf.mean()),
        "real_test_reject_frac": float((conf < CONF_THRESHOLD).mean()),
        "real_test_nn_agreement": agree,
        "real_test_nn_dist": nnd,
        "epochs": cfg.get("clf_epochs", 15),
        "train_seconds": time.time() - t0,
        "log": log,
    }
    _save_json(rep, os.path.join(d, "classifier.json"))
    print(f"[clf] test_acc {rep['test_acc']:.4f} conf {rep['real_test_mean_conf']:.3f} "
          f"nn_agree {agree:.3f}", flush=True)
    return net


def real_reference(out, ev, xtr, ytr, meta):
    """Metrics for a second, disjoint real set with the same class counts: the floor/ceiling."""
    path = os.path.join(out, "metrics", "real_reference.json")
    if os.path.exists(path):
        return
    rng = np.random.RandomState(DATA_SEED + 1)
    used = set(meta["train_indices"])
    idx = []
    for c in range(10):
        pool = [i for i in np.where(ytr.numpy() == c)[0] if i not in used]
        idx.extend(rng.choice(pool, meta["counts"][c], replace=False).tolist())
    m = ev(xtr[np.array(sorted(idx))], seed=0)
    _save_json(m, path)
    print(f"[real-ref] fid {m.get('fid', 0):.2f} recall {m.get('recall', 0):.3f} "
          f"lpips {m.get('lpips', 0):.3f} kl {m.get('kl', 0):.4f}", flush=True)


def setup(cfg, out, device, write_meta=True):
    """Data, classifier and evaluator. Cheap to repeat: weights and reference files are reused."""
    xtr, ytr, xte, yte = load_fashion_mnist(cfg.get("data_root", "/tmp/fmnist"))
    if cfg.get("debug_subsample"):  # local CPU testing only
        k = cfg["debug_subsample"]
        xte, yte = xte[:k], yte[:k]
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    meta_path = os.path.join(out, "data", "real_meta.json")
    real_u8, real_y, meta = build_real_set(xtr, ytr, meta_path if write_meta else None,
                                          class_counts=cfg.get("class_counts"))
    mix_idx = torch.from_numpy(np.random.RandomState(DATA_SEED + 2).choice(
        real_u8.shape[0], cfg["mixture_real"], replace=False))
    meta["mixture_indices"] = mix_idx.tolist()
    if write_meta:
        _save_json(meta, meta_path)

    clf_x, clf_y = (xtr, ytr) if not cfg.get("debug_subsample") else (xtr[:cfg["debug_subsample"]], ytr[:cfg["debug_subsample"]])
    clf = setup_classifier(cfg, out, clf_x, clf_y, xte, yte, real_u8, real_y, device)

    t0 = time.time()
    ev = Evaluator(real_u8, real_y, meta, clf, device)
    print(f"[eval] reference stats ready in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.join(out, "metrics"), exist_ok=True)
    real_reference(out, ev, xtr, ytr, meta)
    return ev, real_u8, mix_idx


def make_timer(out, tag):
    d = os.path.join(out, "timing")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tag}.json")
    timing = json.load(open(path)) if os.path.exists(path) else []

    def timer(rec):
        timing.append(rec)
        _save_json(timing, path)
    return timer


def run_task(cfg, out, task):
    """One unit of work in its own process: gen 0 of (arch, seed), or one lineage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev, real_u8, mix_idx = setup(cfg, out, device, write_meta=False)
    tag = task["arch"] + ("_gen0" if task["kind"] == "gen0" else "_" + task["loop"]) + f"_seed{task['seed']}"
    runner = Runner(cfg, out, device, ev, real_u8, mix_idx, make_timer(out, tag))
    s0, m0 = runner.gen0(task["arch"], task["seed"])
    if task["kind"] == "gen0":
        return True
    return runner.lineage(task["arch"], task["loop"], task["seed"], s0, m0, cfg["deadline_ts"])


def seeds_of(cfg, arch):
    return cfg.get("arch_seeds", {}).get(arch, cfg["seeds"])


def loops_of(cfg, arch):
    return cfg.get("arch_loops", {}).get(arch, cfg["loops"])


# measured on a Kaggle T4: train seconds per image-epoch, and sample seconds per generation
SEC_PER_IMG_EPOCH = {"dcgan": 2.2e-4, "vae": 1.05e-4, "ddpm": 1.63e-3}
SAMPLE_S = {"dcgan": 1, "vae": 1, "ddpm": 190}


def lineage_cost(cfg, arch, loop):
    """Estimated seconds: training images summed over generations, plus sampling and eval per generation."""
    G, n = cfg["generations"], cfg["n_samples"]
    imgs = {"replace": G * n, "mixture": G * (n + cfg["mixture_real"]),
            "accumulate": n * G * (G + 1) // 2, "accumulate_sizematch": G * n}[loop]
    return (imgs * cfg["epochs"].get(arch, 50) * SEC_PER_IMG_EPOCH.get(arch, 2.2e-4)
            + (SAMPLE_S.get(arch, 1) + 40) * G)


def schedule(cfg, out, cfg_path, tasks, n_gpu):
    """Run tasks as subprocesses, one per GPU, next task to the first free GPU."""
    import subprocess
    pending, running, failed = list(tasks), {}, []
    while pending or running:
        for g in range(n_gpu):
            if g not in running and pending and time.time() < cfg["deadline_ts"]:
                t = pending.pop(0)
                pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                           PYTHONPATH=pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
                p = subprocess.Popen([sys.executable, "-m", "mc.run", cfg_path, out, "--task", json.dumps(t)],
                                     env=env)
                running[g] = (p, t, time.time())
                print(f"[sched] gpu{g} start {t}", flush=True)
        time.sleep(10)
        for g, (p, t, ts) in list(running.items()):
            if p.poll() is not None:
                rc = p.returncode
                what = {0: "done", 3: "stopped (time budget)"}.get(rc, f"FAILED rc={rc}")
                print(f"[sched] gpu{g} {what} {t} in {(time.time() - ts) / 60:.1f} min", flush=True)
                if rc == 3:
                    pending.append(t)
                elif rc != 0:
                    failed.append(t)
                del running[g]
        if not running and pending and time.time() >= cfg["deadline_ts"]:
            print(f"[stop] time budget reached; {len(pending)} tasks left, rerun to resume", flush=True)
            return failed, pending
    return failed, pending


def main(cfg, out):
    t_start = time.time()
    cfg["deadline_ts"] = t_start + cfg.get("time_budget_hours", 10.5) * 3600
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    print(f"[env] device {device} x{n_gpu} "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else ''} torch {torch.__version__}",
          flush=True)
    os.makedirs(out, exist_ok=True)
    if cfg.get("resume_dir") and os.path.isdir(cfg["resume_dir"]):
        print(f"[resume] copying {cfg['resume_dir']} -> {out}", flush=True)
        keep = cfg.get("resume_archs")  # only resume these archs; others start fresh
        def skip(_dir, names):
            return [x for x in names if keep is not None and any(x.startswith(a + "_") for a in
                                                                 ("dcgan", "vae", "ddpm") if a not in keep)]
        # "resume_paths" limits the copy to these top-level folders (for example only the classifier)
        for sub in cfg.get("resume_paths") or [""]:
            src = os.path.join(cfg["resume_dir"], sub)
            if os.path.exists(src):
                shutil.copytree(src, os.path.join(out, sub), dirs_exist_ok=True, ignore=skip)
    cfg_path = os.path.join(out, "config.json")
    _save_json(cfg, cfg_path)

    n_par = min(cfg.get("parallel_gpus", 1), n_gpu)
    if n_par >= 2:
        # shared setup once (classifier, reference stats), then every task in its own process
        ev, _, _ = setup(cfg, out, device)
        del ev
        torch.cuda.empty_cache()
        gen0 = [dict(kind="gen0", arch=a, seed=s) for a in cfg["archs"] for s in seeds_of(cfg, a)]
        lins = [dict(kind="lineage", arch=a, loop=l, seed=s)
                for a in cfg["archs"] for s in seeds_of(cfg, a) for l in loops_of(cfg, a)]
        lins.sort(key=lambda t: -lineage_cost(cfg, t["arch"], t["loop"]))  # longest first
        f0, p0 = schedule(cfg, out, cfg_path, gen0, n_par)
        f1, p1 = schedule(cfg, out, cfg_path, lins if not (f0 or p0) else [], n_par)
        status = dict(failed=f0 + f1, not_started=p0 + p1)
        _save_json(status, os.path.join(out, "run_status.json"))
        print(f"[done] failed={len(status['failed'])} not_started={len(status['not_started'])} "
              f"wall {(time.time() - t_start) / 60:.1f} min", flush=True)
        return

    ev, real_u8, mix_idx = setup(cfg, out, device)
    runner = Runner(cfg, out, device, ev, real_u8, mix_idx, make_timer(out, "main"))
    finished = True
    for arch in cfg["archs"]:
        for seed in seeds_of(cfg, arch):
            s0, m0 = runner.gen0(arch, seed)
            for loop in loops_of(cfg, arch):
                finished &= runner.lineage(arch, loop, seed, s0, m0, cfg["deadline_ts"])
                if not finished:
                    break
            if not finished:
                break
        if not finished:
            break
    print(f"[done] finished={finished} wall {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    out = sys.argv[2] if len(sys.argv) > 2 else cfg.get("out", "out")
    if "--task" in sys.argv:
        ok = run_task(cfg, out, json.loads(sys.argv[sys.argv.index("--task") + 1]))
        sys.exit(0 if ok else 3)
    main(cfg, out)
