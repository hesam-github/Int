"""check_bbox_scale.py -- temporal signal carried by the decoder bbox columns.

For every 15-frame training window (stride 1) it computes the within-window
temporal std of each normalized bbox column, using the same transform as
PIEIntent._normalize_decoder_bbox (normalize_decoder_bbox=True).

Columns:
  temporal   within-window std, averaged over windows (normalized units)
  global     std over all frames (the divisor used by zscore)
  zscore_u   temporal / global: temporal std after per-column zscore
  init_sig   GLOROT_STD * temporal: init-scale pre-activation signal, raw column
  init_z     GLOROT_STD * zscore_u: init-scale pre-activation signal, z-scored column

Run from D:\\PIE codes\\PIE in the pie_tf environment.
"""
import os
import sys

import numpy as np

PIE_PATH = r"D:\PIE codes\PIE_dataset"
CODE_DIR = r"D:\PIE codes\PIE"
os.environ["PIE_PATH"] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from pie_data_Hesam import PIE
from utils import PIE_IMG_WIDTH, PIE_IMG_HEIGHT

SEQ_LEN = 15
MAX_WINDOWS = 50000
SEED = 0
DATA_OPTS = dict(fstride=1, sample_type='all', height_rng=[0, float('inf')],
                 squarify_ratio=0, data_split_type='default', seq_type='intention',
                 min_track_size=15)

# Glorot-uniform std of the decoder kernel: fan_in = 2304 + 4 = 2308, fan_out = 4 * 128 = 512
GLOROT_STD = float(np.sqrt(2.0 / (2308 + 512)))
COLUMNS = ['centre_x', 'ground_y', 'log_h', 'log_w']


def normalize_bbox(b):
    """Same transform as PIEIntent._normalize_decoder_bbox; raw boxes (x1,y1,x2,y2) -> 4 columns."""
    x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    W, H = PIE_IMG_WIDTH, PIE_IMG_HEIGHT
    w = np.maximum(x2 - x1, 1.0) / W
    h = np.maximum(y2 - y1, 1.0) / H
    out = np.empty(b.shape, np.float64)
    out[..., 0] = 0.5 * (x1 + x2) / W
    out[..., 1] = y2 / H
    out[..., 2] = np.log(h) / 5.0
    out[..., 3] = np.log(w) / 5.0
    return out


def collect_windows(seq):
    windows = []
    for track in seq['bbox']:
        b = np.asarray(track, np.float64)
        for s in range(0, len(b) - SEQ_LEN + 1):
            windows.append(b[s:s + SEQ_LEN])
    if not windows:
        sys.exit('no windows built: check DATA_OPTS against the training run')
    wins = np.stack(windows)
    if len(wins) > MAX_WINDOWS:
        rng = np.random.default_rng(SEED)
        wins = wins[rng.choice(len(wins), MAX_WINDOWS, replace=False)]
    return wins


def main():
    imdb = PIE(data_path=PIE_PATH)
    seq = imdb.generate_data_trajectory_sequence('train', **DATA_OPTS)
    wins = collect_windows(seq)

    norm = normalize_bbox(wins)
    temporal = norm.std(axis=1).mean(axis=0)
    global_std = norm.reshape(-1, 4).std(axis=0)
    zunits = temporal / np.maximum(global_std, 1e-12)
    init_sig = GLOROT_STD * temporal
    init_z = GLOROT_STD * zunits
    raw_px = wins.std(axis=1).mean(axis=0)

    print(f"[bbox] tracks={len(seq['bbox'])} windows_used={len(wins)} "
          f"glorot_std={GLOROT_STD:.4f}")
    print(f"{'column':<12}{'temporal':>10}{'global':>10}{'zscore_u':>10}"
          f"{'init_sig':>10}{'init_z':>10}")
    for i, name in enumerate(COLUMNS):
        print(f"{name:<12}{temporal[i]:>10.4f}{global_std[i]:>10.4f}{zunits[i]:>10.4f}"
              f"{init_sig[i]:>10.5f}{init_z[i]:>10.5f}")
    print(f"raw pixel temporal std (x1, y1, x2, y2): {np.round(raw_px, 2)}")
    print("\nReference: decoder recurrent temporal std R_t was ~0.045 in the vis_only run.")
    print("init_sig / init_z far below ~0.045 means the column is invisible to the gates at init scale.")


if __name__ == '__main__':
    main()