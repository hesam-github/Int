"""check_visual_norm.py -- sanity checks for visual_norm.fit_channel_stats.

Run:
    python check_visual_norm.py "<full path to vgg16_none_prep1_stacked_*.dat>"
"""
import os
import sys

import numpy as np
import tensorflow as tf

from visual_norm import fit_channel_stats, ChannelStandardize

SEQ_LEN = 15            # must equal max_size_observe
H, W, C = 7, 7, 512     # VGG16 conv5 map for a 224x224 input
N_SEQ_SAMPLE = 64
SEED = 0
TOL_STATS = 1e-3
TOL_DIFF = 1e-6
TOL_KERAS = 1e-4


def open_stacked(path):
    bytes_per_seq = SEQ_LEN * H * W * C * np.dtype(np.float32).itemsize
    nbytes = os.path.getsize(path)
    if nbytes % bytes_per_seq:
        raise ValueError(f"file size {nbytes} is not a multiple of {bytes_per_seq} bytes; "
                         f"check SEQ_LEN / H / W / C")
    n = nbytes // bytes_per_seq
    print(f"[check] n_seq={n} shape=({n}, {SEQ_LEN}, {H}, {W}, {C})")
    return np.memmap(path, dtype=np.float32, mode='r', shape=(n, SEQ_LEN, H, W, C))


def moments(features, fn=lambda x: x, chunk=64):
    s1 = np.zeros(C)
    s2 = np.zeros(C)
    count = 0
    for start in range(0, features.shape[0], chunk):
        y = fn(np.asarray(features[start:start + chunk], dtype=np.float64)).reshape(-1, C)
        s1 += y.sum(axis=0)
        s2 += (y * y).sum(axis=0)
        count += y.shape[0]
    mean = s1 / count
    var = np.maximum(s2 / count - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def main(path):
    features = open_stacked(path)

    print("\n[check] fitting stats on the full memmap (one pass per statistic)")
    stats = fit_channel_stats(features)
    mu = np.asarray(stats['mean'])
    sd = np.asarray(stats['std'])

    raw_mean, raw_std = moments(features)
    dead = raw_std < 1e-6
    live = ~dead
    print(f"[check] dead channels: {int(dead.sum())}/{C}")

    # 1. standardized train data: mean 0 and std 1 on live channels
    z_mean, z_std = moments(features, lambda x: (x - mu) / sd)
    mean_err = np.abs(z_mean[live]).max()
    std_err = np.abs(z_std[live] - 1.0).max()
    ok1 = mean_err < TOL_STATS and std_err < TOL_STATS
    print(f"\n[1] standardized train: max|mean|={mean_err:.2e}  max|std-1|={std_err:.2e} "
          f"-> {'OK' if ok1 else 'FAIL'}")

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(features.shape[0],
                             size=min(N_SEQ_SAMPLE, features.shape[0]), replace=False))
    x = np.asarray(features[idx], dtype=np.float64)

    # 2. temporal differences scale exactly by 1/std
    z = (x - mu) / sd
    dx = np.diff(x, axis=1)
    dz = np.diff(z, axis=1)
    diff_err = np.abs(dz - dx / sd).max()
    ok2 = diff_err < TOL_DIFF
    print(f"[2] temporal diff: max|dz - dx/std|={diff_err:.2e} -> {'OK' if ok2 else 'FAIL'}")

    # 3. Keras layer output matches the numpy transform
    x32 = np.asarray(features[idx[:2]], dtype=np.float32)
    layer = ChannelStandardize(stats['mean'], stats['std'])
    keras_out = layer(tf.constant(x32)).numpy().astype(np.float64)
    np_out = ((x32 - mu.astype(np.float32)) / sd.astype(np.float32)).astype(np.float64)
    keras_err = np.abs(keras_out - np_out).max()
    ok3 = keras_err < TOL_KERAS
    print(f"[3] keras vs numpy: max|diff|={keras_err:.2e} -> {'OK' if ok3 else 'FAIL'}")

    # 4. distribution shape on live channels (sample)
    flat = x.reshape(-1, C)
    zero_frac = (flat == 0).mean(axis=0)[live]
    max_z = ((flat.max(axis=0) - mu) / sd)[live]
    print("[4] live channels, sample of", len(idx), "sequences:")
    print(f"    raw zero fraction : median={np.median(zero_frac):.3f} "
          f"p90={np.percentile(zero_frac, 90):.3f} max={zero_frac.max():.3f}")
    print(f"    max z per channel : median={np.median(max_z):.1f} "
          f"p99={np.percentile(max_z, 99):.1f} max={max_z.max():.1f}")
    print(f"    raw mean range    : [{raw_mean[live].min():.4f}, {raw_mean[live].max():.4f}]")

    verdict = 'ALL OK' if (ok1 and ok2 and ok3) else 'SOME CHECKS FAILED'
    print(f"\n[summary] {verdict}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python check_visual_norm.py "<path to .dat>"')
    main(sys.argv[1])