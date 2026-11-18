import argparse
import json
import pickle

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

PAPER = {'accuracy': 0.79, 'f1': 0.87}  # ICCV 2019, Table 2, imgcontext + loc

ROWS = ['n', 'positive_rate', 'majority_acc', 'trivial_f1',
        'accuracy', 'precision', 'recall', 'specificity', 'balanced_acc',
        'f1', 'roc_auc', 'pr_auc', 'nll', 'ece_15bins', 'brier',
        'pmean', 'pstd']


def load_pkl(path):
    with open(path, 'rb') as f:
        try:
            d = pickle.load(f)
        except Exception:
            f.seek(0)
            d = pickle.load(f, encoding='bytes')
    y_prob = np.array([np.asarray(r).ravel()[0] for r in d['results']], dtype=float)
    y_true = np.array([np.asarray(g).ravel()[0] for g in d['gt']], dtype=int)
    return y_prob, y_true


def ece(y, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


def compute(y_prob, y_true, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prev = float(y_true.mean())
    two_classes = len(np.unique(y_true)) > 1
    return {
        'n': int(len(y_true)),
        'positive_rate': prev,
        'majority_acc': max(prev, 1 - prev),
        'trivial_f1': 2 * prev / (1 + prev),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': rec,
        'specificity': spec,
        'balanced_acc': 0.5 * (rec + spec),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if two_classes else float('nan'),
        'pr_auc': average_precision_score(y_true, y_prob) if two_classes else float('nan'),
        'nll': log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        'ece_15bins': ece(y_true, y_prob),
        'brier': float(np.mean((y_prob - y_true) ** 2)),
        'pmean': float(y_prob.mean()),
        'pstd': float(y_prob.std()),
        'confusion_matrix': cm.tolist(),
    }


def fmt(v):
    if v is None:
        return '--'
    if isinstance(v, int):
        return str(v)
    return f'{v:.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='paper model ped_intents.pkl')
    ap.add_argument('--own', nargs='+', default=[],
                    help='NAME=path/to/ped_intents_test.pkl (one or more)')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()

    results = {}
    p, y = load_pkl(args.ref)
    results['paper_model'] = compute(p, y, args.thr)
    for item in args.own:
        name, path = item.split('=', 1)
        p, y = load_pkl(path)
        results[name] = compute(p, y, args.thr)

    names = list(results.keys())
    col = 14
    print(f'\nthreshold = {args.thr}')
    print('metric'.ljust(16) + 'paper(text)'.rjust(col)
          + ''.join(n[:col - 1].rjust(col) for n in names))
    for r in ROWS:
        line = r.ljust(16) + fmt(PAPER.get(r)).rjust(col)
        line += ''.join(fmt(results[n][r]).rjust(col) for n in names)
        print(line)
    print('\nconfusion matrices [[TN, FP], [FN, TP]]:')
    for n in names:
        print(f'  {n}: {results[n]["confusion_matrix"]}')

    print('\nchecks:')
    ref = results['paper_model']
    for n in names:
        r = results[n]
        if abs(r['positive_rate'] - ref['positive_rate']) > 0.02 or r['n'] != ref['n']:
            print(f'  [WARN] {n}: test set differs from paper_model '
                  f'(n={r["n"]} vs {ref["n"]}, pos={r["positive_rate"]:.3f} '
                  f'vs {ref["positive_rate"]:.3f}). accuracy/F1 are not comparable; '
                  f'compare AUC and the trivial baselines.')
        if r['accuracy'] <= r['majority_acc']:
            print(f'  [NOTE] {n}: accuracy <= majority_acc '
                  f'({r["accuracy"]:.4f} <= {r["majority_acc"]:.4f})')
        if r['f1'] <= r['trivial_f1']:
            print(f'  [NOTE] {n}: F1 <= trivial all-positive F1 '
                  f'({r["f1"]:.4f} <= {r["trivial_f1"]:.4f})')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'threshold': args.thr, 'paper_text': PAPER,
                       'results': results}, f, indent=2)
        print(f'\nsaved: {args.out}')


if __name__ == '__main__':
    main()