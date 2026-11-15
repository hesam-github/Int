import glob, os
import numpy as np

root = r'D:\PIE codes\PIE_dataset\extra\pose'
files = glob.glob(os.path.join(root, '*', '*', '*.npz'))
tot = ok = 0
for f in files:
    k = np.load(f)['kpts']                        # (T, 17, 3)
    good = (k[:, :, 2] >= 0.3).sum(axis=1) >= 8   # frame is "usable" if at least 8 of 17 joints are confident
    tot += len(good)
    ok += int(good.sum())
print(f'{len(files)} track files, {ok}/{tot} frames usable ({100 * ok / max(tot, 1):.1f}%)')