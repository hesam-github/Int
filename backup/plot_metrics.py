'''
plot_metrics.py

Purpose: all plotting logic for train_pie_intent_tensorflow_gpu.py lives here,
so the main training script only needs one import + one function call.

Produces PNGs under <run_dir>/metrics_plots/:
  1. loss_curve.png -- train/val loss per epoch (from history.pkl), with the
     final post-training test NLL marked as a reference point (test is
     intentionally never evaluated per-epoch during training).
  2. metrics_comparison_thresh05.png -- bar chart of accuracy, precision,
     recall, f1, roc_auc (AUC), pr_auc (AP), nll, ece across train/val/test,
     from the evaluate_split_full_metrics() dicts (threshold=0.5).
  3. metrics_comparison_calibrated.png -- same, at the calibrated threshold
     (only if calibrated_metrics is passed).

ASSUMPTION (verify against your actual compute_full_metrics output): dict
keys are 'accuracy','precision','recall','f1','roc_auc','pr_auc','nll','ece'
-- matches BOTH copies (pie_intent_helpers.py and train_helpers.py) as
pasted, so this should be safe as-is. If you later edit one copy and not
the other (the project's documented open duplication issue), only
METRIC_KEYS below needs editing.
'''
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
    ax.set_title('Training loss (train/val per-epoch; test = final NLL only)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_metrics] saved -> {save_path}")


def plot_metrics_comparison(train_metrics, val_metrics, test_metrics, save_path, title):
    splits = ['train', 'val', 'test']
    split_dicts = [train_metrics, val_metrics, test_metrics]

    missing = [(s, k) for k, _ in METRIC_KEYS for s, d in zip(splits, split_dicts) if k not in d]
    if missing:
        print(f"[plot_metrics] WARNING -- missing (split, key) pairs, shown as 0: {missing}")

    n_metrics = len(METRIC_KEYS)
    x = list(range(n_metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (split_name, d) in enumerate(zip(splits, split_dicts)):
        values = [d.get(key, 0.0) for key, _ in METRIC_KEYS]
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, values, width=width, label=split_name)
        ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRIC_KEYS])
    ax.set_ylabel('value')
    ax.set_title(title)
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

    test_nll = test_metrics.get('nll')
    plot_loss_curve(history, test_nll, os.path.join(plots_dir, 'loss_curve.png'))

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