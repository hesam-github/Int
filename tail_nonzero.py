"""tail_nonzero.py -- is the heavy tail of cached VGG16 conv5 maps more than sparsity?

Per channel, compares the observed excess kurtosis (zeros included) with a
zero-inflated half-normal reference that has the SAME zero fraction. The
reference has no extra tail beyond its positive part, so the difference
isolates tail behaviour that sparsity alone cannot explain.

Half-normal excess kurtosis (nonzero part only) is about 0.87; use it as yardstick.

Run from D:\\PIE codes\\PIE:
    python tail_nonzero.py "<full path to a train .dat>"
"""
import os
import sys

import numpy as np

SEQ_LEN, H, W, C = 15, 7, 7, 512
N_SAMPLE = 64
SEED = 0
EPS = 1e-6


def open_stacked(path):
    bytes_per_seq = SEQ_LEN * H * W * C * np.dtype(np.float32).itemsize
    n = os.path.getsize(path) // bytes_per_seq
    print(f"[tail_nz] n_seq={n}")
    return np.memmap(path, dtype=np.float32, mode='r', shape=(n, SEQ_LEN, H, W, C))


def excess_kurtosis_and_zmax(v):
    """v: (N, K) float64. Each column is centered and scaled by its own mean and std."""
    m = v.mean(axis=0)
    s = v.std(axis=0)
    s = np.where(s < EPS, 1.0, s)
    z = (v - m) / s
    return (z ** 4).mean(axis=0) - 3.0, np.abs(z).max(axis=0)


def main(path):
    features = open_stacked(path)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(features.shape[0],
                             size=min(N_SAMPLE, features.shape[0]), replace=False))
    flat = np.asarray(features[idx], dtype=np.float32).reshape(-1, C)
    assert np.isfinite(flat).all() and (flat >= 0).all(), "non-finite or negative activations"
    n = flat.shape[0]
    p0 = (flat == 0).mean(axis=0)

    kurt_obs, zmax_obs = excess_kurtosis_and_zmax(flat.astype(np.float64))

    kurt_ref = np.empty(C)
    zmax_ref = np.empty(C)
    for c in range(C):
        u = rng.random(n)
        ref = np.where(u < p0[c], 0.0, np.abs(rng.standard_normal(n)))
        k, zm = excess_kurtosis_and_zmax(ref[:, None])
        kurt_ref[c] = k[0]
        zmax_ref[c] = zm[0]

    kurt_nz = np.full(C, np.nan)
    for c in range(C):
        v = flat[:, c][flat[:, c] > 0].astype(np.float64)
        if v.size >= 10:
            z = (v - v.mean()) / max(v.std(), EPS)
            kurt_nz[c] = (z ** 4).mean() - 3.0

    live = p0 < 1.0
    diff = kurt_obs - kurt_ref

    print(f"[tail_nz] sample={idx.size} sequences, {n} positions per channel")
    print(f"[tail_nz] zero fraction: median={np.median(p0):.3f} "
          f"p10={np.percentile(p0, 10):.3f} p90={np.percentile(p0, 90):.3f}")
    print(f"[tail_nz] always-zero channels (excluded): {int((~live).sum())}\n")

    rows = [
        ('kurt observed (zeros included)', kurt_obs[live]),
        ('kurt reference (ZI half-normal)', kurt_ref[live]),
        ('kurt observed - reference', diff[live]),
        ('kurt observed, nonzero only', kurt_nz[live]),
        ('zmax observed', zmax_obs[live]),
        ('zmax reference', zmax_ref[live]),
    ]
    print(f"{'quantity':<36}{'median':>10}{'p90':>10}{'p99':>10}")
    print("-" * 66)
    for name, arr in rows:
        arr = arr[np.isfinite(arr)]
        print(f"{name:<36}{np.median(arr):>10.2f}{np.percentile(arr, 90):>10.2f}"
              f"{np.percentile(arr, 99):>10.2f}")

    d = diff[live]
    print(f"\n[tail_nz] channels with observed - reference > 5: "
          f"{int((d > 5).sum())}/{int(live.sum())}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python tail_nonzero.py "<path to train .dat>"')
    main(sys.argv[1])