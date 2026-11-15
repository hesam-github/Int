"""
eval_results_io.py

Purpose: shared readers for the ped_intents*.pkl prediction files written
by PIEIntentExperiment.test_chunk() / PIEIntent.test_chunk() (a dict with
'results' -- raw sigmoid outputs -- and 'gt' -- ground-truth labels --
one entry per evaluated track). Every post-hoc threshold-tuning /
calibration / comparison script re-parses this same file format; this
module gives that parsing logic one home instead of N slightly-drifting
copies.
"""

import pickle
import numpy as np


def load_predictions_dict(pkl_path):
    """
    Reads one ped_intents*.pkl file and returns (probs, labels) as flat
    float/int numpy arrays, one value per track -- each 'results'/'gt'
    entry is ravelled and its first element taken, since test_chunk()
    stores per-track predictions/labels as (possibly nested) arrays of
    length 1.

    Used by apply_val_threshold_to_test.py (via a thin run_dir/split_name
    wrapper), finetune_from_baseline_gpu_lr1e4.py, and
    evaluate_and_compare.py -- previously three separate, byte-identical
    copies of these same two lines.
    """
    with open(pkl_path, 'rb') as fid:
        saved = pickle.load(fid)
    probs = np.asarray([float(np.ravel(p)[0]) for p in saved['results']])
    labels = np.asarray([int(np.ravel(g)[0]) for g in saved['gt']])
    return probs, labels


def load_probs_and_labels(pkl_path):
    """
    Alternate reader for the same ped_intents*.pkl format, used by
    threshold_tuning.py and prior_shift_correction.py (previously two
    separate, byte-identical copies of this function). Kept separate
    from load_predictions_dict() above -- rather than merged into it --
    because the two were written independently and differ slightly in
    how 'gt' is coerced (elementwise ravel+cast here vs. a single
    array-level cast in load_predictions_dict); merging them could
    silently change the result for a 'gt' shape where the two approaches
    disagree.
    """
    with open(pkl_path, 'rb') as fid:
        data = pickle.load(fid)
    y_prob = np.array([np.asarray(r).ravel()[0] for r in data['results']])
    y_true = np.array(data['gt']).astype(int).ravel()
    return y_prob, y_true
