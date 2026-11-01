"""Per-generation metrics, same definitions as the Stanford Dogs study.

- FID: Inception-v3 pool3 (2048-d, pytorch-fid weights), grayscale repeated to
  3 channels and resized to 299, generated set vs the real set R.
- Recall: Kynkaanniemi et al., k=3. Fraction of real features inside the union
  of k-NN balls around generated features.
- LPIPS diversity: mean LPIPS (AlexNet) over 500 random pairs of generated
  images, upsampled to 64x64 to match the input size of the original study.
- Class KL: D_KL(p_gen || p_real) over the classifier's predicted labels.
- Reliability: mean max-softmax confidence, rejection fraction (conf < 0.5),
  KL and shares on confident predictions only, and nearest-neighbour agreement
  (classifier label vs true label of the nearest real image in classifier
  feature space).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .classifier import predict
from .data import to_float

CONF_THRESHOLD = 0.5
RECALL_K = 3
LPIPS_PAIRS = 500


class Inception:
    def __init__(self, device):
        from pytorch_fid.inception import InceptionV3
        self.device = device
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
                               resize_input=True, normalize_input=True).to(device).eval()

    @torch.no_grad()
    def features(self, x_u8, batch=250):
        out = []
        for i in range(0, x_u8.shape[0], batch):
            x = x_u8[i:i + batch].to(self.device).float().div(255.0)  # [0,1]
            x = x.repeat(1, 3, 1, 1)
            out.append(self.net(x)[0].squeeze(-1).squeeze(-1).double().cpu())
        return torch.cat(out).numpy()


def gaussian_stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def fid_from_stats(mu1, s1, mu2, s2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


@torch.no_grad()
def knn_recall(real_feats, gen_feats, device, k=RECALL_K, batch=2000):
    r = torch.as_tensor(real_feats, dtype=torch.float32, device=device)
    g = torch.as_tensor(gen_feats, dtype=torch.float32, device=device)
    # radius of each generated point = distance to its k-th nearest generated neighbour
    radii = []
    for i in range(0, g.shape[0], batch):
        d = torch.cdist(g[i:i + batch], g)
        radii.append(d.kthvalue(k + 1, dim=1).values)  # +1 skips the point itself
    radii = torch.cat(radii)
    covered = 0
    for i in range(0, r.shape[0], batch):
        d = torch.cdist(r[i:i + batch], g)
        covered += (d <= radii[None, :]).any(1).sum().item()
    return covered / r.shape[0]


class LPIPSDiversity:
    def __init__(self, device):
        import lpips
        self.device = device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, x_u8, seed, n_pairs=LPIPS_PAIRS):
        g = torch.Generator().manual_seed(seed)
        n = x_u8.shape[0]
        a = torch.randint(0, n, (n_pairs,), generator=g)
        b = torch.randint(0, n - 1, (n_pairs,), generator=g)
        b = b + (b >= a).long()  # never pair an image with itself

        def prep(idx):
            x = to_float(x_u8[idx].to(self.device)).repeat(1, 3, 1, 1)
            return F.interpolate(x, size=64, mode="bilinear", align_corners=False)

        return float(self.net(prep(a), prep(b)).mean().item())


def kl_div(p_gen, p_real, eps=1e-12):
    p_gen = np.asarray(p_gen, dtype=np.float64)
    p_real = np.asarray(p_real, dtype=np.float64)
    m = p_gen > 0
    return float(np.sum(p_gen[m] * np.log(p_gen[m] / np.maximum(p_real[m], eps))))


def class_stats(probs, real_dist, common, rare):
    conf, pred = probs.max(1)
    conf, pred = conf.numpy(), pred.numpy()
    dist = np.bincount(pred, minlength=10) / len(pred)
    keep = conf >= CONF_THRESHOLD
    out = {
        "class_dist": dist.round(5).tolist(),
        "kl": kl_div(dist, real_dist),
        "clf_mean_conf": float(conf.mean()),
        "reject_frac": float(1.0 - keep.mean()),
        "common_share": float(dist[common].sum()),
        "rare_share": float(dist[rare].sum()),
    }
    if keep.any():
        fd = np.bincount(pred[keep], minlength=10) / keep.sum()
        out.update({
            "class_dist_filtered": fd.round(5).tolist(),
            "kl_filtered": kl_div(fd, real_dist),
            "common_share_filtered": float(fd[common].sum()),
            "rare_share_filtered": float(fd[rare].sum()),
        })
    else:  # every prediction rejected: filtered quantities are undefined
        out.update({"class_dist_filtered": None, "kl_filtered": None,
                    "common_share_filtered": None, "rare_share_filtered": None})
    return out, pred


@torch.no_grad()
def nn_agreement(gen_feats, gen_pred, real_feats, real_labels, device, batch=2000):
    r = real_feats.to(device)
    g = gen_feats.to(device)
    nn_idx, nn_dist = [], []
    for i in range(0, g.shape[0], batch):
        d = torch.cdist(g[i:i + batch], r)
        v, j = d.min(1)
        nn_idx.append(j.cpu()); nn_dist.append(v.cpu())
    nn_idx, nn_dist = torch.cat(nn_idx), torch.cat(nn_dist)
    nn_label = real_labels.cpu()[nn_idx].numpy()
    return float((nn_label == gen_pred).mean()), float(nn_dist.mean())


class Evaluator:
    """Holds the real-set reference statistics and computes all metrics for a sample set."""

    def __init__(self, real_u8, real_y, real_meta, clf, device):
        self.device = device
        self.clf = clf
        self.real_dist = np.array(real_meta["distribution"])
        self.common = real_meta["common"]
        self.rare = real_meta["rare"]
        self.inception = Inception(device)
        self.lpips = LPIPSDiversity(device)
        self.real_inc = self.inception.features(real_u8)
        self.real_mu, self.real_sigma = gaussian_stats(self.real_inc)
        _, self.real_clf_feats = predict(clf, real_u8, device)
        self.real_y = real_y

    def quick_fid(self, x_u8):
        f = self.inception.features(x_u8)
        mu, s = gaussian_stats(f)
        return fid_from_stats(mu, s, self.real_mu, self.real_sigma)

    def __call__(self, x_u8, seed=0):
        gen_inc = self.inception.features(x_u8)
        mu, s = gaussian_stats(gen_inc)
        m = {
            "n": int(x_u8.shape[0]),
            "fid": fid_from_stats(mu, s, self.real_mu, self.real_sigma),
            "recall": knn_recall(self.real_inc, gen_inc, self.device),
            "lpips": self.lpips(x_u8, seed),
        }
        probs, feats = predict(self.clf, x_u8, self.device)
        cs, pred = class_stats(probs, self.real_dist, self.common, self.rare)
        m.update(cs)
        agree, dist = nn_agreement(feats, pred, self.real_clf_feats, self.real_y, self.device)
        m["clf_nn_agreement"] = agree
        m["clf_nn_dist"] = dist
        return m
