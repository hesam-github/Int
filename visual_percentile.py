"""visual_percentile.py -- per-channel positive-percentile scaling for VGG features.

Divides each channel by the q-th percentile of its strictly positive values
(fitted on TRAIN only). Zeros stay zero. No centering, no clipping.
"""
import numpy as np
import tensorflow as tf


def fit_positive_percentile_scales(features, q=99.0, chunk=64, eps=1e-6):
    """features: (N, T, H, W, C) train memmap or array.
    Returns dict with 'scales' (list of length C) and 'q'.
    Channels with no positive values get scale=1.0.
    """
    n_total = features.shape[0]
    C = features.shape[-1]
    scales = np.ones(C, dtype=np.float64)

    # Collect positive values per channel in chunks to limit memory
    positives = [[] for _ in range(C)]
    for start in range(0, n_total, chunk):
        block = np.asarray(features[start:start + chunk], dtype=np.float32)
        flat = block.reshape(-1, C)
        for c in range(C):
            pos = flat[:, c]
            pos = pos[pos > 0]
            if pos.size:
                positives[c].append(pos)

    for c in range(C):
        if positives[c]:
            all_pos = np.concatenate(positives[c])
            scales[c] = float(np.percentile(all_pos, q))
            if scales[c] < eps:
                scales[c] = 1.0
        # else leave 1.0

    print(f"[visual_percentile] q={q} fitted on {n_total} windows x {C} channels | "
          f"scales in [{scales.min():.4f}, {scales.max():.4f}] | "
          f"channels with no positives: {int((scales == 1.0).sum())}")
    return {'scales': scales.tolist(), 'q': float(q)}


class PositivePercentileScale(tf.keras.layers.Layer):
    """(..., C) -> x / scales. Zeros stay zero. No centering or clipping."""

    def __init__(self, scales, **kwargs):
        super().__init__(**kwargs)
        self.scales_list = [float(v) for v in scales]
        self._scales = tf.constant(self.scales_list, dtype=tf.float32)

    def call(self, x):
        return x / self._scales

    def get_config(self):
        config = super().get_config()
        config.update(scales=self.scales_list)
        return config