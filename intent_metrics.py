"""
intent_metrics.py

Framework-agnostic post-hoc metrics computed from already-produced
(ground-truth, predicted-probability) pairs -- e.g. from ped_intents.pkl
saved by pie_intent.PIEIntent.test_chunk(). This does NOT touch the
paper's training loss/metrics (binary_crossentropy / accuracy); it only
adds extra evaluation numbers on top of the same predictions the
reference test_chunk() already produced, so it works identically for a
TensorFlow run or a PyTorch run as long as you pass it (gt, prob) arrays.
"""

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    log_loss, brier_score_loss
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 \
            else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_extra_metrics(y_true, y_prob, save_path=None):
    """
    :param y_true: array-like of 0/1 ground-truth intention labels
    :param y_prob: array-like of predicted probabilities (pre-rounding)
    :param save_path: if given, dumps metrics as JSON there
        (use paths['metrics_path'] from experiment_paths.py)
    :return: dict of metrics
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = np.round(y_prob).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'ece': expected_calibration_error(y_true, y_prob, n_bins=10),
    }

    # These require both classes present in y_true; guard against
    # degenerate test splits.
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError as e:
        metrics['roc_auc'] = None
        metrics['roc_auc_error'] = str(e)

    try:
        metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
    except ValueError as e:
        metrics['pr_auc'] = None
        metrics['pr_auc_error'] = str(e)

    try:
        metrics['log_loss'] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError as e:
        metrics['log_loss'] = None
        metrics['log_loss_error'] = str(e)

    metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()  # [[tn, fp], [fn, tp]]
    metrics['n_test_samples'] = int(len(y_true))
    metrics['positive_rate_gt'] = float(y_true.mean())

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def load_and_compute_from_ped_intents(ped_intents_path, save_path=None):
    """
    Convenience loader for the pickle format pie_intent.PIEIntent.test_chunk()
    already writes: {'ped_id', 'images', 'results', 'gt'}.
    """
    import pickle
    with open(ped_intents_path, 'rb') as fid:
        data = pickle.load(fid)
    y_prob = np.array([np.asarray(r).ravel()[0] for r in data['results']])
    y_true = np.array(data['gt']).ravel()
    return compute_extra_metrics(y_true, y_prob, save_path=save_path)