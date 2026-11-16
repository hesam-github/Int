'''
plot_metrics.py

Purpose: all plotting logic for train_pie_intent_tensorflow_gpu.py lives here,
so the main training script only needs one import + one function call.

Produces PNGs under <run_dir>/metrics_plots/:
  1. loss_curve.png -- train/val Keras loss per epoch (from history.pkl). The
     loss includes the L2 term. Test NLL is not drawn here: it depends on the
     test class prior and is not comparable with train/val loss.
  2. epoch_curves.png -- per-epoch AUC / NLL / loss for train, val, test from
     epoch_metrics.json. train/test points exist only on evaluation epochs.
  3. metrics_comparison_thresh05.png -- bar chart of accuracy, precision,
     recall, f1, roc_auc (AUC), pr_auc (AP), nll, ece across train/val/test,
     from the evaluate_split_full_metrics() dicts (threshold=0.5). The title
     shows the majority-class accuracy of each split; only AUC is comparable
     across splits.
  4. metrics_comparison_calibrated.png -- same, at the calibrated threshold
     (only if calibrated_metrics is passed).

ASSUMPTION (verify against your actual compute_full_metrics output): dict
keys are 'accuracy','precision','recall','f1','roc_auc','pr_auc','nll','ece'
-- matches BOTH copies (pie_intent_helpers.py and train_helpers.py) as
pasted, so this should be safe as-is. If you later edit one copy and not
the other (the project's documented open duplication issue), only
METRIC_KEYS below needs editing.
'''
import json
import math
import os
import matplotlib
matplotlib.use('Agg')  # headless-safe, no GUI backend required
import matplotlib.pyplot as plt

METRIC_KEYS = [
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1', 'F1'),
    ('roc_auc', 'AUC'),
    ('pr_auc', 'AP'),
    ('nll', 'NLL'),
    ('ece', 'ECE'),
]


def _unwrap_history(history):
    if history is None:
        return {}
    if hasattr(history, 'history'):
        return history.history
    return history


def plot_loss_curve(history, test_nll, save_path):
    hist = _unwrap_history(history)
    train_loss = hist.get('loss')
    val_loss = hist.get('val_loss')

    if train_loss is None:
        print(f"[plot_metrics] 'loss' key not found in history (keys: "
              f"{list(hist.keys())}) -- skipping {save_path}")
        return

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label='train loss')
    if val_loss is not None:
        ax.plot(epochs, val_loss, label='val loss')
    else:
        print("[plot_metrics] 'val_loss' key not found in history -- plotting train only.")

    if test_nll is not None:
        ax.scatter([len(train_loss)], [test_nll], color='red', zorder=5,
                   label=f'test NLL (final, overlap=1) = {test_nll:.4f}')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training loss per epoch (Keras loss, includes L2)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_metrics] saved -> {save_path}")


def plot_epoch_curves(epoch_metrics_path, save_path):
    """Per-epoch AUC / NLL / loss for train, val, test from epoch_metrics.json.
    train_*/test_* exist only on evaluation epochs; missing values are skipped, never drawn as 0."""
    if not os.path.isfile(epoch_metrics_path):
        print(f"[plot_metrics] {epoch_metrics_path} not found -- skipping epoch curves.")
        return
    with open(epoch_metrics_path, 'r', encoding='utf-8') as fid:
        rows = json.load(fid)
    if not rows:
        return
    panels = [('roc_auc', 'AUC (comparable across splits)'),
              ('nll', 'NLL (depends on class prior)'),
              ('loss', 'loss (train/val include L2)')]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, panels):
        for split in ('train', 'val', 'test'):
            pts = [(r['epoch'], r[f'{split}_{key}']) for r in rows
                   if isinstance(r.get(f'{split}_{key}'), (int, float))
                   and not math.isnan(r[f'{split}_{key}'])]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker='o', markersize=3, label=split)
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_metrics] saved -> {save_path}")


def plot_metrics_comparison(train_metrics, val_metrics, test_metrics, save_path, title):
    splits = ['train', 'val', 'test']
    split_dicts = [train_metrics, val_metrics, test_metrics]

    missing = [(s, k) for k, _ in METRIC_KEYS for s, d in zip(splits, split_dicts) if k not in d]
    if missing:
        print(f"[plot_metrics] WARNING -- missing (split, key) pairs, not drawn: {missing}")

    n_metrics = len(METRIC_KEYS)
    x = list(range(n_metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (split_name, d) in enumerate(zip(splits, split_dicts)):
        values = [d.get(key, float('nan')) for key, _ in METRIC_KEYS]
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, [0.0 if v != v else v for v in values],
                      width=width, label=split_name)
        ax.bar_label(bars, labels=['' if v != v else f'{v:.3f}' for v in values],
                     fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRIC_KEYS])
    ax.set_ylabel('value')
    prior = '  '.join(f"{s}={d.get('majority_acc', float('nan')):.3f}"
                      for s, d in zip(splits, split_dicts))
    ax.set_title(f"{title}\nmajority-class accuracy: {prior}  "
                 f"(only AUC is comparable across splits)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_metrics] saved -> {save_path}")


def plot_training_metrics(history, train_metrics, val_metrics, test_metrics, run_dir,
                           calibrated_metrics=None):
    plots_dir = os.path.join(run_dir, 'metrics_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Test NLL is not drawn on the loss curve: it depends on the test class prior,
    # so it is not comparable with the train/val Keras loss.
    plot_loss_curve(history, None, os.path.join(plots_dir, 'loss_curve.png'))
    plot_epoch_curves(os.path.join(run_dir, 'epoch_metrics.json'),
                      os.path.join(plots_dir, 'epoch_curves.png'))

    plot_metrics_comparison(
        train_metrics, val_metrics, test_metrics,
        os.path.join(plots_dir, 'metrics_comparison_thresh05.png'),
        title='Final metrics per split (threshold=0.5)')

    if calibrated_metrics is not None:
        train_cal, val_cal, test_cal = calibrated_metrics
        plot_metrics_comparison(
            train_cal, val_cal, test_cal,
            os.path.join(plots_dir, 'metrics_comparison_calibrated.png'),
            title='Final metrics per split (calibrated threshold)')
