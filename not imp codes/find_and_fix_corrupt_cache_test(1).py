# find_and_fix_corrupt_cache_test.py (test-feature-cache version)
"""
Scan the TEST-SPLIT feature cache (the one built by
pie_intent_experiment.py / test_pie_intent_tensorflow.py) for corrupted
or truncated .npy files and delete them.

This targets a DIFFERENT cache directory than the original
find_and_fix_corrupt_cache.py (which scans vgg_cache/{backbone}).
This one scans:
    {data_path}/data/data/pie/{split}/features_{crop_type}_{crop_mode}/{backbone}_none

Run this from the PIE codes folder, then re-run test_pie_intent_tensorflow.py --
the missing files will be re-extracted automatically.
"""
import os
import sys
import numpy as np
from config import Config
from pie_dataset import BACKBONES

cfg = Config()

# These must match the data_opts used in test_pie_intent_tensorflow.py
SPLIT = 'test'
CROP_TYPE = 'context'
CROP_MODE = 'pad_resize'
BACKBONE = cfg.backbone  # 'vgg16'

if BACKBONE not in BACKBONES:
    print(f"Unknown backbone '{BACKBONE}'. Known: {list(BACKBONES.keys())}")
    sys.exit(1)

EXPECTED_DIM = BACKBONES[BACKBONE][1]

cache_root = os.path.join(
    cfg.data_path, 'data', 'data', 'pie', SPLIT,
    f'features_{CROP_TYPE}_{CROP_MODE}', f'{BACKBONE}_none'
)

if not os.path.isdir(cache_root):
    print(f"Cache dir not found: {cache_root}")
    sys.exit(1)

print(f"Scanning: {cache_root}")
print(f"backbone={BACKBONE}  expected_feature_dim={EXPECTED_DIM}\n")

total = 0
bad = []

for dirpath, _, filenames in os.walk(cache_root):
    for fn in filenames:
        if not fn.endswith('.npy'):
            continue
        total += 1
        path = os.path.join(dirpath, fn)

        size = os.path.getsize(path)
        if size == 0:
            bad.append((path, 'empty file (0 bytes)'))
            continue

        try:
            arr = np.load(path)
            if arr.dtype != np.float32:
                bad.append((path, f'unexpected dtype {arr.dtype}'))
            elif arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
                bad.append((path,
                    f'unexpected shape {arr.shape} '
                    f'(expected (T, {EXPECTED_DIM}))'))
            elif not np.isfinite(arr).all():
                bad.append((path, 'contains NaN/Inf'))
        except Exception as e:
            bad.append((path, f'{type(e).__name__}: {e}'))

print(f"Checked {total} cache files.")
print(f"Found {len(bad)} corrupted file(s):\n")
for path, reason in bad:
    print(f"  {path}\n    -> {reason}")

if bad:
    answer = input(f"\nDelete these {len(bad)} corrupted file(s)? [y/N] ")
    if answer.strip().lower() == 'y':
        for path, _ in bad:
            os.remove(path)
        print(f"Deleted {len(bad)} file(s). Re-run test_pie_intent_tensorflow.py now.")
    else:
        print("Nothing deleted.")
else:
    print("Cache looks clean for this split/crop/backbone combo.")