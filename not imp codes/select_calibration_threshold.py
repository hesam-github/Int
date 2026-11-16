"""
select_calibration_threshold.py

Purpose: find the best decision threshold on the VAL split only (never
on test), and save it to a calibration.json file next to the run's
outputs, so every downstream evaluation script reads the threshold from
this single source instead of hardcoding it in multiple places.

Run this once, after training finishes for a given run.
"""

import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from train_helpers import build_calibration_dict

def load_probs_and_labels(pkl_path):
    """Loads ped_intents_{split}.pkl and returns raw probabilities + true labels."""
    with open(pkl_path, 'rb') as fid:
        data = pickle.load(fid)
    y_prob = np.array([np.asarray(r).ravel()[0] for r in data['results']])
    y_true = np.array(data['gt']).astype(int).ravel()
    return y_prob, y_true


def find_best_threshold(y_true, y_prob, metric='f1', thresh_range=(0.05, 0.95), step=0.01):
    """
    Sweeps every threshold in the given range and returns the one that
    maximizes the chosen metric ('accuracy' or 'f1') on y_true/y_prob.
    """
    best_thresh, best_score = 0.5, -1.0
    for thresh in np.arange(thresh_range[0], thresh_range[1] + step, step):
        y_pred = (y_prob >= thresh).astype(int)
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"unknown metric {metric}")
        if score > best_score:
            best_score, best_thresh = score, thresh
    return best_thresh, best_score

from train_helpers import build_calibration_dict

def select_and_save_calibration(run_dir, metric='f1'):
    val_path = os.path.join(run_dir, 'ped_intents_val.pkl')
    val_prob, val_true = load_probs_and_labels(val_path)

    best_thresh, best_val_score = find_best_threshold(val_true, val_prob, metric=metric)

    calibration = build_calibration_dict(
        threshold=best_thresh, metric=metric, val_true=val_true, val_prob=val_prob)

    calib_path = os.path.join(run_dir, 'calibration.json')
    with open(calib_path, 'w', encoding='utf-8') as fid:
        json.dump(calibration, fid, indent=2)

    print(f"[calibration] threshold={best_thresh:.2f} (metric={metric}, val_score={best_val_score:.4f})")
    print(f"[calibration] saved to {calib_path}")
    return calibration


if __name__ == '__main__':
    RUN_DIR = r'D:\PIE codes\PIE_dataset\experiments\tensorflow\convlstm_encdec_h128_f64_k2\article_params_17_5_1405-22h45m29s'
    select_and_save_calibration(RUN_DIR, metric='f1')