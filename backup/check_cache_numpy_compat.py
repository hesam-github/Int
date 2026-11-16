"""


Scans the three pickle-cache categories that PIE/PIEIntent code relies on
and reports which files fail to unpickle under the currently installed
numpy (e.g. after downgrading numpy 2.x -> 1.26.4). Does NOT delete or
regenerate anything - it only reports, so you can decide what to
regenerate and how (e.g. by re-running extract_pie_reference_features.py,
which already has its own corrupted-file cleanup logic).

Run this after activating the environment where you just installed
numpy==1.26.4 / opencv-python==4.9.0.80.
"""

import os
import pickle
import sys

import numpy as np

PIE_PATH = r'D:\PIE codes\PIE_dataset'


def check_file(path: str) -> tuple[bool, str]:
    """Returns (ok, error_message). error_message is '' if ok."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray) and data.size == 0:
            return False, "unpickled to an empty array"
        if isinstance(data, (list, tuple)) and len(data) == 0:
            return False, "unpickled to an empty list/tuple"
        if data is None:
            return False, "unpickled to None"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def scan_folder(folder: str, label: str, max_report: int = 20) -> dict:
    print(f"\n{'='*60}")
    print(f"Scanning: {label}")
    print(f"Folder  : {folder}")
    print(f"{'='*60}")

    if not os.path.isdir(folder):
        print(f"  Folder does not exist - skipping ({folder})")
        return {'label': label, 'total': 0, 'bad': [], 'skipped': True}

    pkl_files = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.endswith('.pkl'):
                pkl_files.append(os.path.join(root, fn))

    total = len(pkl_files)
    print(f"  Found {total} .pkl files. Checking...")

    bad = []
    for i, path in enumerate(pkl_files, 1):
        ok, err = check_file(path)
        if not ok:
            bad.append((path, err))
        if i % 2000 == 0 or i == total:
            print(f"    checked {i}/{total} (bad so far: {len(bad)})")

    print(f"\n  RESULT for {label}: {total - len(bad)}/{total} OK, {len(bad)} BAD")
    if bad:
        print(f"  Sample of bad files (up to {max_report}):")
        for path, err in bad[:max_report]:
            print(f"    - {path}\n        {err}")

    return {'label': label, 'total': total, 'bad': bad, 'skipped': False}


def scan_single_file(path: str, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Checking single file: {label}")
    print(f"Path: {path}")
    print(f"{'='*60}")

    if not os.path.isfile(path):
        print(f"  File does not exist - skipping ({path})")
        return {'label': label, 'total': 0, 'bad': [], 'skipped': True}

    ok, err = check_file(path)
    if ok:
        print("  OK")
        return {'label': label, 'total': 1, 'bad': [], 'skipped': False}
    else:
        print(f"  BAD: {err}")
        return {'label': label, 'total': 1, 'bad': [(path, err)], 'skipped': False}


if __name__ == '__main__':
    print(f"Python      : {sys.version}")
    print(f"numpy       : {np.__version__}")
    try:
        import cv2
        print(f"opencv-python: {cv2.__version__}")
    except ImportError:
        print("opencv-python: not importable (skipped)")

    results = []

    # 1. VGG16 feature cache (from extract_pie_reference_features.py)
    features_root = os.path.join(PIE_PATH, 'data', 'data', 'pie')
    results.append(scan_folder(features_root, "VGG16 feature cache (features_context_pad_resize/vgg16_none)"))

    # 2. pie_database.pkl (built by the PIE class)
    pie_db_path = os.path.join(PIE_PATH, 'data_cache', 'pie_database.pkl')
    results.append(scan_single_file(pie_db_path, "pie_database.pkl"))

    # 3. sequence-level cache (seq_*.pkl, from generate_data_trajectory_sequence)
    seq_cache_dir = os.path.join(PIE_PATH, 'data_cache')
    if os.path.isdir(seq_cache_dir):
        seq_files = [f for f in os.listdir(seq_cache_dir) if f.startswith('seq_') and f.endswith('.pkl')]
        print(f"\n{'='*60}")
        print(f"Scanning: sequence-level cache (seq_*.pkl)")
        print(f"Folder  : {seq_cache_dir}")
        print(f"{'='*60}")
        print(f"  Found {len(seq_files)} seq_*.pkl files. Checking...")
        bad = []
        for fn in seq_files:
            path = os.path.join(seq_cache_dir, fn)
            ok, err = check_file(path)
            if not ok:
                bad.append((path, err))
        print(f"\n  RESULT: {len(seq_files) - len(bad)}/{len(seq_files)} OK, {len(bad)} BAD")
        for path, err in bad[:20]:
            print(f"    - {path}\n        {err}")
        results.append({'label': 'seq_*.pkl cache', 'total': len(seq_files), 'bad': bad, 'skipped': False})
    else:
        print(f"\nSequence cache folder does not exist - skipping ({seq_cache_dir})")

    # ---- summary ----
    print(f"\n\n{'#'*60}")
    print("SUMMARY")
    print(f"{'#'*60}")
    any_bad = False
    for r in results:
        if r['skipped']:
            print(f"  {r['label']:55s}: SKIPPED (folder/file not found)")
            continue
        n_bad = len(r['bad'])
        status = "OK" if n_bad == 0 else f"{n_bad} FILE(S) NEED REGENERATION"
        print(f"  {r['label']:55s}: {status}")
        if n_bad:
            any_bad = True

    if any_bad:
        print("\nNext step: re-run extract_pie_reference_features.py for the feature")
        print("cache (it auto-detects and regenerates only corrupted files), and")
        print("delete pie_database.pkl / bad seq_*.pkl files so PIE() rebuilds them")
        print("(cheap, no GPU needed).")
    else:
        print("\nAll checked caches are readable under the current numpy/opencv versions.")