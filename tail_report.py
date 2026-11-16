"""tail_report.py -- shape diagnostics of cached VGG16 conv5 maps under candidate transforms.

All metrics are computed after per-channel centering and scaling, so they are
invariant to affine transforms (raw and zscore give identical rows). Only
non-affine per-element transforms (sqrt, log1p) can change them.

Run from D:\\PIE codes\\PIE:
    python tail_report.py "<full path to a train .dat>"
"""
import os
import sys

import numpy as np

SEQ_LEN, H, W, C = 15, 7, 7, 512
N_SAMPLE = 128          # lower to 64 if RAM is tight
SEED = 0
EPS = 1e-6

CASES = [
    ('affine (raw / zscore)', lambda x: x),
    ('sqrt', np.sqrt),
    ('log1p', np.log1p),
]


def open_stacked(path):
    bytes_per_seq = SEQ_LEN * H * W * C * np.dtype(np.float32).itemsize
    n = os.path.getsize(path) // bytes_per_seq
    print(f"[tail] n_seq={n}")
    return np.memmap(path, dtype=np.float32, mode='r', shape=(n, SEQ_LEN, H, W, C))


def shape_metrics(y):
    """y: (..., C). Centered and scaled per channel, so every metric is shift- and scale-free."""
    flat = y.reshape(-1, C).astype(np.float64)
    m = flat.mean(axis=0)
    s = flat.std(axis=0)
    s = np.where(s < EPS, 1.0, s)
    z = (flat - m) / s

    kurt = (z ** 4).mean(axis=0) - 3.0          # excess kurtosis per channel; Gaussian = 0
    sq = (z ** 2).ravel()
    k = max(1, int(0.01 * sq.size))
    top1 = np.partition(sq, -k)[-k:].sum() / sq.sum()   # share of sum(z^2) in the top 1% entries
    zmax = np.abs(z).max(axis=0)                # per-channel max |z|

    return dict(
        kurt_med=float(np.median(kurt)),
        kurt_p90=float(np.percentile(kurt, 90)),
        top1=float(top1),
        zmax_med=float(np.median(zmax)),
        zmax_p99=float(np.percentile(zmax, 99)),
    )


def main(path):
    features = open_stacked(path)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(features.shape[0],
                             size=min(N_SAMPLE, features.shape[0]), replace=False))
    x = np.asarray(features[idx], dtype=np.float32)

    print(f"[tail] sample={len(idx)} sequences | zero fraction={float((x == 0).mean()):.3f}")
    print(f"[tail] Gaussian reference: kurt=0, median zmax ~ 4.5 at this sample size\n")

    header = (f"{'transform':<24}{'kurt_med':>10}{'kurt_p90':>10}"
              f"{'top1%_sq':>10}{'zmax_med':>10}{'zmax_p99':>10}")
    print(header)
    print("-" * len(header))
    for name, fn in CASES:
        m = shape_metrics(fn(x))
        print(f"{name:<24}{m['kurt_med']:>10.1f}{m['kurt_p90']:>10.1f}"
              f"{m['top1']:>10.3f}{m['zmax_med']:>10.1f}{m['zmax_p99']:>10.1f}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python tail_report.py "<path to train .dat>"')
    main(sys.argv[1])