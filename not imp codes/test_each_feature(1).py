"""test_each_feature.py — linear test of every numeric track feature.

Run from the code folder, in pie_tf or pie_pose (no TensorFlow):

    python test_each_feature.py

For each feature it fits logistic regression on TRAIN tracks only and
reports AUC on val and test. Two fits:

  scaled  StandardScaler fit on train. This is "does the feature carry
          the label?", with bbox's pixel range taken out of the way.
  raw     no scaling. This is "would a model that sees raw numbers let
          the large-range feature drown the others?"

The unit is one track (mean over time), not one overlapping window, so
the same pedestrian is not counted many times. This is not the ConvLSTM.
A feature with scaled AUC near 0.5 has no linear signal. A high AUC on
look/gesture is a human-label shortcut, not a deployable gain.
"""
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PIE_PATH = r'D:\PIE codes\PIE_dataset'
CODE_DIR = r'D:\PIE codes\PIE'
# Match the run under discussion: random 0.5/0.4/0.1. Set to 'default'
# for the fixed set split (set01+02+04 / set05+06 / set03).
SPLIT_TYPE = 'random'
RANDOM_RATIOS = [0.5, 0.4, 0.1]  # PIE order: train, test, val
MIN_TRACK_SIZE = 15
ATTACH_POSE = True
OUT_CSV = 'test_each_feature.csv'

SKIP = {
    'image', 'images', 'ped_id', 'pid',
    'intention_binary', 'intention_prob',
}
HUMAN = {'look', 'action', 'gesture', 'cross'}
LEAK = {'obd_speed', 'gps_speed', 'speed'}
REDUNDANT = {'center'}


def kind_of(name):
    if name == 'bbox':
        return 'bbox'
    if name in HUMAN:
        return 'human-label'
    if name in LEAK:
        return 'leakage'
    if name in REDUNDANT:
        return 'from-bbox'
    return 'feature'


def track_matrix(beh, key):
    rows = []
    for track in beh[key]:
        arr = np.asarray(track, dtype=np.float64)
        if arr.dtype == object or arr.size == 0:
            raise ValueError(f'{key}: non-numeric or empty track')
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        rows.append(arr.mean(axis=0))
    return np.vstack(rows)


def labels_of(beh):
    y = []
    for track in beh['intention_binary']:
        y.append(int(np.asarray(track).reshape(-1)[0] > 0.5))
    return np.asarray(y, dtype=int)


def numeric_keys(beh):
    keys = []
    n = len(beh['intention_binary'])
    for key, value in beh.items():
        if key in SKIP or not isinstance(value, (list, tuple)) or len(value) != n:
            continue
        try:
            matrix = track_matrix(beh, key)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(matrix).all():
            continue
        if np.nanstd(matrix) < 1e-8:
            continue
        keys.append(key)
    return keys


def load_splits():
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    os.environ['PIE_PATH'] = PIE_PATH
    from pie_data_Hesam import PIE
    from train_helpers import build_split_sequences

    opts = {
        'fstride': 1,
        'sample_type': 'all',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': SPLIT_TYPE,
        'seq_type': 'intention',
        'min_track_size': MIN_TRACK_SIZE,
        'random_params': {
            'ratios': RANDOM_RATIOS,
            'val_data': True,
            'regen_data': False,
        },
        'kfold_params': {'num_folds': 5, 'fold': 1},
    }
    imdb = PIE(data_path=PIE_PATH)
    train, val, test = build_split_sequences(imdb, opts)
    if ATTACH_POSE:
        pose_dir = os.path.join(PIE_PATH, 'extra', 'pose')
        if os.path.isdir(pose_dir):
            from extra_features import attach_pose
            for split in (train, val, test):
                attach_pose(split, pose_dir, scale='bbox_relative')
        else:
            print(f'[pose] skipped, no directory {pose_dir}')
    return train, val, test


def fit_auc(x_train, y_train, x_val, y_val, x_test, y_test, scale):
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=4000, solver='lbfgs')
    clf.fit(x_train, y_train)

    def _auc(x, y):
        if len(np.unique(y)) < 2:
            return float('nan')
        return float(roc_auc_score(y, clf.predict_proba(x)[:, 1]))

    return _auc(x_val, y_val), _auc(x_test, y_test)


def main():
    train, val, test = load_splits()
    keys = numeric_keys(train)
    y_train, y_val, y_test = labels_of(train), labels_of(val), labels_of(test)
    print(f'split={SPLIT_TYPE}  tracks train/val/test = '
          f'{len(y_train)}/{len(y_val)}/{len(y_test)}')
    print(f'positive rate train/val/test = '
          f'{y_train.mean():.3f}/{y_val.mean():.3f}/{y_test.mean():.3f}')
    print(f'features: {keys}')

    mats = {
        'train': {k: track_matrix(train, k) for k in keys},
        'val': {k: track_matrix(val, k) for k in keys},
        'test': {k: track_matrix(test, k) for k in keys},
    }
    bbox_auc = None
    if 'bbox' in keys:
        bbox_auc = fit_auc(
            mats['train']['bbox'], y_train,
            mats['val']['bbox'], y_val,
            mats['test']['bbox'], y_test, scale=True)[1]

    rows = []
    for key in keys:
        val_s, test_s = fit_auc(
            mats['train'][key], y_train, mats['val'][key], y_val,
            mats['test'][key], y_test, scale=True)
        _, test_raw = fit_auc(
            mats['train'][key], y_train, mats['val'][key], y_val,
            mats['test'][key], y_test, scale=False)
        plus_test = float('nan')
        delta = float('nan')
        if 'bbox' in keys and key != 'bbox':
            plus_test = fit_auc(
                np.hstack([mats['train']['bbox'], mats['train'][key]]), y_train,
                np.hstack([mats['val']['bbox'], mats['val'][key]]), y_val,
                np.hstack([mats['test']['bbox'], mats['test'][key]]), y_test,
                scale=True)[1]
            delta = plus_test - bbox_auc
        rows.append({
            'feature': key,
            'dim': int(mats['train'][key].shape[1]),
            'kind': kind_of(key),
            'val_auc_scaled': val_s,
            'test_auc_scaled': test_s,
            'test_auc_raw': test_raw,
            'test_auc_bbox_plus': plus_test,
            'delta_vs_bbox': delta,
            'train_std': float(mats['train'][key].std()),
        })

    if keys:
        all_test = fit_auc(
            np.hstack([mats['train'][k] for k in keys]), y_train,
            np.hstack([mats['val'][k] for k in keys]), y_val,
            np.hstack([mats['test'][k] for k in keys]), y_test,
            scale=True)[1]
        rows.append({
            'feature': 'ALL',
            'dim': int(sum(mats['train'][k].shape[1] for k in keys)),
            'kind': 'bundle',
            'val_auc_scaled': fit_auc(
                np.hstack([mats['train'][k] for k in keys]), y_train,
                np.hstack([mats['val'][k] for k in keys]), y_val,
                np.hstack([mats['test'][k] for k in keys]), y_test,
                scale=True)[0],
            'test_auc_scaled': all_test,
            'test_auc_raw': fit_auc(
                np.hstack([mats['train'][k] for k in keys]), y_train,
                np.hstack([mats['val'][k] for k in keys]), y_val,
                np.hstack([mats['test'][k] for k in keys]), y_test,
                scale=False)[1],
            'test_auc_bbox_plus': float('nan'),
            'delta_vs_bbox': (all_test - bbox_auc) if bbox_auc is not None else float('nan'),
            'train_std': float('nan'),
        })

    rows.sort(key=lambda r: (r['feature'] == 'ALL', -(r['test_auc_scaled'] or 0)))
    header = ('feature', 'dim', 'kind', 'val_auc_scaled', 'test_auc_scaled',
              'test_auc_raw', 'test_auc_bbox_plus', 'delta_vs_bbox', 'train_std')
    print()
    print(f"{'feature':22} {'dim':4} {'kind':12} {'valS':7} {'testS':7} "
          f"{'testRaw':7} {'bbox+':7} {'delta':7} {'std':8}")
    for row in rows:
        def fmt(value):
            return '   nan' if value != value else f'{value:7.3f}'
        print(f"{row['feature']:22} {row['dim']:4d} {row['kind']:12} "
              f"{fmt(row['val_auc_scaled'])} {fmt(row['test_auc_scaled'])} "
              f"{fmt(row['test_auc_raw'])} {fmt(row['test_auc_bbox_plus'])} "
              f"{fmt(row['delta_vs_bbox'])} {fmt(row['train_std'])}")

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fid:
        writer = csv.DictWriter(fid, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwrote {os.path.abspath(OUT_CSV)}')
    print('delta_vs_bbox = scaled test AUC of (bbox + feature) minus bbox alone.')
    print('A positive delta is extra linear signal. human-label rows are not deployable.')


if __name__ == '__main__':
    main()
