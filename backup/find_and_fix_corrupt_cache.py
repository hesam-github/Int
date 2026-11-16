# find_and_fix_corrupt_cache.py
"""
Scan the feature cache for corrupted / truncated .npy files and delete them.

Run this ONCE from the PIE codes folder (same place train.py lives):
    python find_and_fix_corrupt_cache.py

After it finishes, just re-run train.py -- _ensure_cache() will see the
deleted files as "missing" and re-extract them automatically (it will
briefly reload the backbone just for those tracks).

FIX (this version): the expected feature dimension is now read from
pie_dataset.BACKBONES[cfg.backbone] instead of being hardcoded to 4096.
The old hardcoded version would flag every *correctly-shaped* resnet50
cache file (dim=2048) as "corrupted" and offer to delete it -- turning
a cache-health tool into a cache-destruction tool the moment anyone runs
an experiment with backbone='resnet50'. This version is backbone-aware
and will not do that.
"""
import os
import sys
import numpy as np
from config import Config
from pie_dataset import BACKBONES

cfg = Config()

if cfg.backbone not in BACKBONES:
    print(f"Unknown backbone '{cfg.backbone}' in Config. "
          f"Known backbones: {list(BACKBONES.keys())}")
    sys.exit(1)

# Single source of truth for "what should a valid feature vector look like
# for THIS backbone" -- same dict pie_dataset.py itself uses, so this
# script can never drift out of sync with the actual extractor.
EXPECTED_DIM = BACKBONES[cfg.backbone][1]

cache_root = os.path.join(cfg.data_path, 'vgg_cache', cfg.backbone)

if not os.path.isdir(cache_root):
    print(f"Cache dir not found: {cache_root}")
    sys.exit(1)

print(f"Scanning: {cache_root}")
print(f"backbone={cfg.backbone}  expected_feature_dim={EXPECTED_DIM}\n")

total = 0
bad = []

for dirpath, _, filenames in os.walk(cache_root):
    for fn in filenames:
        if not fn.endswith('.npy'):
            continue
        total += 1
        path = os.path.join(dirpath, fn)

        # Fast check: a 0-byte (or absurdly small) file is definitely broken,
        # no need to even try loading it.
        size = os.path.getsize(path)
        if size == 0:
            bad.append((path, 'empty file (0 bytes)'))
            continue

        # Slow but thorough check: actually try to load the array and make
        # sure its data matches its declared shape/dtype for THIS backbone.
        try:
            arr = np.load(path)
            if arr.dtype != np.float32:
                bad.append((path, f'unexpected dtype {arr.dtype}'))
            elif arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
                bad.append((path,
                    f'unexpected shape {arr.shape} '
                    f'(expected (T, {EXPECTED_DIM}) for backbone={cfg.backbone})'))
        except Exception as e:
            bad.append((path, f'{type(e).__name__}: {e}'))

print(f"Checked {total} cache files.")
print(f"Found {len(bad)} corrupted file(s):\n")
for path, reason in bad:
    print(f"  {path}\n    -> {reason}")

if bad:
    answer = input(f"\nDelete these {len(bad)} corrupted file(s) so they get "
                    f"re-extracted on the next train.py run? [y/N] ")
    if answer.strip().lower() == 'y':
        for path, _ in bad:
            os.remove(path)
        print(f"Deleted {len(bad)} file(s). Re-run train.py now.")
    else:
        print("Nothing deleted.")
else:
    print("Cache looks clean - the crash must have come from something else "
          "(or the cache for this backbone hasn't been built yet).")