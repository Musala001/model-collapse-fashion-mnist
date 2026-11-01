# Model collapse in recursively trained image generators (Fashion-MNIST)

Code for the honours research report "Model Collapse in Recursively Trained Image Generators: Training Loops and
Rare-Class Erasure on Fashion-MNIST" (M. Ndouvhada, University of the Witwatersrand, 2026; supervisor C. Van Alten).

DCGAN, VAE and DDPM generators are retrained on their own samples for ten generations under four loops (replace,
mixture, accumulate, size-matched accumulate) on an imbalanced Fashion-MNIST set (five common classes at 1000 images,
five rare classes at 200). Metrics: FID, recall, LPIPS diversity, class KL, rare-class share, and classifier
reliability checks. Balanced controls (600 images per class) test whether rare classes are lost because they are rare.

Results (per-lineage metric files, sample grids, number ledger) are public as a Kaggle dataset:
https://www.kaggle.com/datasets/musalandouvhada/model-collapse-fashion-mnist-results

## Layout

- `src/mc/`: data (`data.py`), classifier (`classifier.py`), generators (`models.py`), training loops with
  per-generation checkpointing and resume (`loops.py`), metrics (`metrics.py`), entry point (`run.py`).
- `configs/`: one JSON config per run. The main runs are `dcgan.json`, `vae_ddpm.json`, `dcgan_balanced.json` and
  `vae_balanced.json`; `smoke.json`, `onegpu.json`, `parcheck.json` and `timing2.json` are trial and timing runs.
- `kernels/`: the exact Kaggle scripts that were run, generated from `src/mc` and a config by
  `tools/build_kernel.py`.
- `tools/`: aggregation (`aggregate.py`), per-class loss (`class_loss.py`), report figures (`paper_figures.py`),
  every number in the report with its source (`paper_numbers.py`), classifier top-k check (`clf_topk.py`), and the
  results bundle and release scripts (`bundle.py`, `release.py`).

## Running

The runs used Kaggle notebooks with two T4 GPUs. To build a Kaggle script from a config:

    python tools/build_kernel.py configs/dcgan.json
    kaggle kernels push -p kernels/mc-fmnist-dcgan-r1

Each kernel downloads Fashion-MNIST through torchvision, trains the classifier, runs the lineages and writes
`out/` (metrics JSON per lineage, timing logs, sample grids). The aggregation and report scripts read those outputs
from `results/<kernel>/out/`, as laid out in the Kaggle dataset.
