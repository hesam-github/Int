"""visual_norm.py -- per-channel standardization of cached VGG16 conv5 maps.

Statistics are fitted on the TRAIN memmap only, once per run, and stored in
the layer config. The layer lives inside the Keras graph, so val/test and
reloaded checkpoints use the same constants and the feature cache is unchanged.
"""
import numpy as np
import tensorflow as tf

RAW = 'raw'
LOG1P = 'log1p'


def log1p_np(x):
    return np.log1p(np.maximum(x, 0.0))


def _to_space(x, space):
    if space == RAW:
        return x
    if space == LOG1P:
        return log1p_np(x)
    raise ValueError(f"unknown stats space: {space!r}")


def fit_channel_stats(features, chunk=64, space=RAW):
    """features: (N, T, H, W, C). Statistics are computed in `space`:
    'raw' -> on x, 'log1p' -> on log1p(max(x, 0)). TRAIN memmap only."""
    n_total = features.shape[0]
    C = features.shape[-1]

    s1 = np.zeros(C, np.float64)
    count = 0
    for start in range(0, n_total, chunk):
        flat = _to_space(np.asarray(features[start:start + chunk], dtype=np.float64),
                         space).reshape(-1, C)
        s1 += flat.sum(axis=0)
        count += flat.shape[0]
    mean = s1 / count

    s2 = np.zeros(C, np.float64)
    for start in range(0, n_total, chunk):
        flat = _to_space(np.asarray(features[start:start + chunk], dtype=np.float64),
                         space).reshape(-1, C)
        s2 += ((flat - mean) ** 2).sum(axis=0)
    std = np.sqrt(s2 / count)

    dead = std < 1e-6
    print(f"[visual_norm] stats space = {space} | fitted on {count} positions x {C} channels | "
          f"mean in [{mean.min():.4f}, {mean.max():.4f}] | "
          f"std in [{std.min():.4f}, {std.max():.4f}] | "
          f"dead channels (left unscaled): {int(dead.sum())}")
    if space == LOG1P:
        print("[visual_norm] WARNING: these statistics live in log1p space. They are only valid "
              "for inputs that went through log1p FIRST; never apply them to raw features.")
    std[dead] = 1.0
    return {'mean': mean.tolist(), 'std': std.tolist(), 'space': space}


def stats_space_for(capacity):
    return LOG1P if (capacity or {}).get('log1p') else RAW


def visual_op_order(vn, capacity):
    """Single source of truth for the order of visual input ops.
    Returns a list of 'log1p' / 'standardize'. Raises on inconsistent combinations."""
    want_log = bool((capacity or {}).get('log1p'))
    if not vn:
        return ['log1p'] if want_log else []
    space = vn.get('space')
    if space is None:                      # run saved before this fix
        if want_log:
            print("[visual_pipeline] WARNING: LEGACY run (stats without 'space' + log1p): "
                  "order is standardize -> log1p, which clips every standardized value <= 0. "
                  "Not comparable with log1p -> standardize runs.")
            return ['standardize', 'log1p']
        return ['standardize']
    if space == LOG1P and want_log:
        return ['log1p', 'standardize']
    if space == RAW and not want_log:
        return ['standardize']
    raise ValueError(f"inconsistent visual pipeline: stats space={space!r} but "
                     f"capacity log1p={want_log}")


def apply_visual_ops_np(x, vn, capacity):
    """NumPy twin of the Keras graph, for scripts that re-implement the forward pass."""
    for op in visual_op_order(vn, capacity):
        if op == 'log1p':
            x = log1p_np(x)
        else:
            x = (x - np.asarray(vn['mean'], np.float32)) / np.asarray(vn['std'], np.float32)
    return x


def visual_experiment_code(vnorm, log1p):
    return {(False, False): 'A', (True, False): 'B',
            (True, True): 'C', (False, True): 'D'}[(bool(vnorm), bool(log1p))]


def confirm_visual_pipeline(vnorm, capacity):
    """Prints the resolved visual pipeline. Asks for confirmation only for C
    (standardize + log1p), the one case where the order changes the experiment."""
    log1p = bool((capacity or {}).get('log1p'))
    code = visual_experiment_code(vnorm, log1p)
    chain = {'A': 'VGG -> ConvLSTM  (raw)',
             'B': 'VGG -> Standardize -> ConvLSTM',
             'C': 'VGG -> Log1p -> Standardize -> ConvLSTM',
             'D': 'VGG -> Log1p -> ConvLSTM  (no standardization)'}[code]
    print(f"\n[visual_pipeline] experiment {code}: {chain}", flush=True)
    if not (vnorm and log1p):
        if log1p:
            print("[visual_pipeline] note: log1p without standardization; compare with C to "
                  "separate the log effect from the standardization effect.", flush=True)
        return True
    print("[visual_pipeline] statistics will be fitted on log1p(max(x, 0)) of the TRAIN cache.",
          flush=True)
    print("[visual_pipeline] WARNING: the old order (standardize -> log1p) is NOT used; "
          "runs with the same flags made before this fix are not comparable.", flush=True)
    raw = input("[visual_pipeline] proceed with log1p -> standardize? [Y/n]: ").strip().lower()
    return raw in ('', 'y', 'yes')

class ChannelStandardize(tf.keras.layers.Layer):
    """(..., C) -> (x - mean) / std, constants fixed at fit time."""

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean_list = [float(v) for v in mean]
        self.std_list = [float(v) for v in std]
        self._mean = tf.constant(self.mean_list, dtype=tf.float32)
        self._std = tf.constant(self.std_list, dtype=tf.float32)

    def call(self, x):
        return (x - self._mean) / self._std

    def get_config(self):
        config = super().get_config()
        config.update(mean=self.mean_list, std=self.std_list)
        return config