# evaluate.py
"""
Metrics for binary pedestrian-intention. Primary: F1, AUC.
Plus ECE/NLL for the uncertainty-quantification contribution and the
majority-class baseline for context.
"""
import os
import json
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        features = batch['features'].to(device)
        bbox = batch['bbox'].to(device)
        logit = model(features, bbox)
        prob = torch.sigmoid(logit).cpu().numpy()
        label = batch['label'].numpy()
        probs.append(prob)
        labels.append(label)
    return np.concatenate(probs), np.concatenate(labels)  # (N,), (N,)


def nll_binary(probs, labels, eps=1e-7):
    p = np.clip(np.asarray(probs), eps, 1 - eps)
    y = np.asarray(labels)
    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(nll)


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidence = np.where(probs >= 0.5, probs, 1 - probs)
    prediction = (probs >= 0.5).astype(int)
    correct = (prediction == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidence >= lo) & (confidence <= hi) if i == 0 \
            else (confidence > lo) & (confidence <= hi)
        if in_bin.sum() == 0:
            continue
        bin_weight = in_bin.sum() / len(probs)
        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidence[in_bin].mean()
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)


def calibrate_threshold(probs, labels, metric='accuracy'):
    """
    Finds the threshold in [0.1, 0.9] that maximizes accuracy/F1 on the
    GIVEN (probs, labels) pair.

    CRITICAL: only ever call this with VALIDATION data. Calling it with the
    labels of the split you are about to REPORT (e.g. test) is data
    leakage -- the threshold gets fit to exactly the labels it is later
    scored against, producing an optimistic number that says nothing about
    generalization. Choose the threshold on val, freeze it, then apply the
    SAME frozen value to test.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score, best_threshold = -1.0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = accuracy_score(labels, preds) if metric == 'accuracy' \
            else f1_score(labels, preds, zero_division=0)
        if score > best_score:
            best_score, best_threshold = score, t
    return float(best_threshold)


def compute_metrics(probs, labels, threshold=0.5):
    """
    threshold=0.5 (default) gives RAW accuracy/precision/recall/F1 -- the
    numbers directly comparable to the reference paper and unaffected by
    pos_weight (pos_weight only reshapes the loss, never the 0.5 decision
    boundary). Pass a different threshold only for deliberate, clearly
    labeled "calibrated" numbers, and always fit that threshold on
    validation data, never on the split being scored.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = (probs >= threshold).astype(int)

    two_class_present = len(np.unique(labels)) > 1
    return {
        'n': int(len(labels)),
        'accuracy': float(accuracy_score(labels, preds)),
        'majority_acc': float(max(labels.mean(), 1 - labels.mean())),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'auc': float(roc_auc_score(labels, probs)) if two_class_present else float('nan'),
        'ap': float(average_precision_score(labels, probs)) if two_class_present else float('nan'),
        'nll': nll_binary(probs, labels),
        'ece': expected_calibration_error(probs, labels),
        'decision_threshold': float(threshold),
    }


def print_metrics(tag, m):
    print(f"\n[{tag}]  n={m['n']}  (threshold={m['decision_threshold']:.2f})")
    print(f"F1={m['f1']:.4f}    AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
    print(f"acc={m['accuracy']:.4f}  (majority_baseline={m['majority_acc']:.4f})")
    print(f"precision={m['precision']:.4f}  recall={m['recall']:.4f}")
    print(f"NLL={m['nll']:.4f}  ECE={m['ece']:.4f}\n")

def evaluate_checkpoint(cfg, ckpt_path=None, save_json=True):
    from torch.utils.data import DataLoader
    from pie_data_Hesam import PIE
    from train import build_model

    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.out_dir, 'best.pt')

    imdb = PIE(data_path=cfg.data_path)

    if cfg.model_name == 'convlstm_encdec':
        from pie_convlstm_dataset import PIEConvLSTMDataset
        test_ds = PIEConvLSTMDataset(
            imdb, 'test',
            observe_length=cfg.observe_length,
            seq_overlap_rate=cfg.seq_overlap_rate,
            normalize_bbox=False,
            balance_classes=False,   # NEVER balance the test split
        )
    else:
        from pie_dataset import PIEIntentionDataset
        test_ds = PIEIntentionDataset(
            imdb, 'test',
            observe_length=cfg.observe_length,
            backbone=cfg.backbone,
            seq_overlap_rate=cfg.seq_overlap_rate,
            normalize_bbox=cfg.normalize_bbox,
        )

    test_dl = DataLoader(
        dataset=test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state'])

    probs, labels = collect_predictions(model, test_dl, device)
    m = compute_metrics(probs, labels)
    print_metrics(f"TEST    ({cfg.model_name},  obs={cfg.observe_length})", m)

    if save_json:
        out_path = os.path.join(os.path.dirname(ckpt_path), 'test_metrics.json')
        with open(out_path, 'w') as f:
            json.dump(m, f, indent=2)
        print(f"test metrics -> {out_path}")
    return m