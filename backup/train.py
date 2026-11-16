# train.py -- first part imports, MODEL_REGISTRY، build_model، make_loaders
import os
import json
import random
import argparse
import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pie_dataset import PIEIntentionDataset, BACKBONES

from config import Config
from pie_data_Hesam import PIE
from pie_dataset import PIEIntentionDataset
from pie_convlstm_dataset import PIEConvLSTMDataset
from gru_baseline import GRUIntentionModel
from pie_convlstm_encdec import PIEConvLSTMEncDec
# FIX: calibrate_threshold was called at runtime (see below) but never
# imported -- this is a guaranteed NameError the very first time an epoch
# tries to compute a calibrated threshold, independent of any model bug.
from evaluate import collect_predictions, compute_metrics, print_metrics, calibrate_threshold

MODEL_REGISTRY = {
    'gru': GRUIntentionModel,
    'convlstm_encdec': PIEConvLSTMEncDec,
}

def build_model(cfg):
    cls = MODEL_REGISTRY[cfg.model_name]

    if cfg.model_name == 'convlstm_encdec':
        return cls(
            convlstm_filters=cfg.convlstm_filters,
            convlstm_kernel_size=cfg.convlstm_kernel_size,
            decoder_hidden=cfg.hidden_dim,
            input_dropout=cfg.input_dropout,
            recurrent_dropout=cfg.recurrent_dropout,
        )

    feature_dim = BACKBONES[cfg.backbone][1]
    kwargs = dict(
        feature_dim=feature_dim,
        hidden_dim=cfg.hidden_dim,
        use_bbox=cfg.use_bbox,
        input_dropout=cfg.input_dropout,
        head_dropout=cfg.head_dropout,
    )
    if cfg.model_name == 'gru':
        kwargs.update(num_layers=cfg.num_layers, gru_dropout=cfg.gru_dropout)
    model = cls(**kwargs)

    # FIX: this assert previously referenced `model` and `feature_dim`
    # BEFORE either was defined -- `model` was never even assigned in this
    # function (it went straight to `return cls(**kwargs)`), so this was a
    # guaranteed NameError on the very first call to build_model() for
    # model_name='gru' (or any non-convlstm model), independent of any
    # backbone/checkpoint issue. Moved here, after the model is actually
    # built, so it does what the original comment intended: catch the case
    # where BaseIntentionModel silently falls back to its own hardcoded
    # default feature_dim (4096) instead of respecting cfg.backbone -- which
    # works by accident for backbone='vgg16' but crashes deep inside
    # input_proj with a cryptic matmul shape error for anything else
    # (confirmed traceback: "mat1 and mat2 shapes cannot be multiplied
    # (1920x25092 and 4100x128)" for backbone='vgg16_conv'). This assert
    # turns that into an immediate, readable message at build time instead.
    assert model.feature_dim == feature_dim, (
        f"model.feature_dim={model.feature_dim} != expected {feature_dim} "
        f"for backbone='{cfg.backbone}' -- build_model/BACKBONES mismatch."
    )
    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_loaders(cfg):
    imdb = PIE(data_path=cfg.data_path)

    if cfg.model_name == 'convlstm_encdec':
        common = {
            'observe_length': cfg.observe_length,
            'seq_overlap_rate': cfg.seq_overlap_rate,
            'normalize_bbox': False,
            'balance_classes': cfg.balance_classes,
            'balance_seed': cfg.balance_seed,
        }
        train_ds = PIEConvLSTMDataset(imdb, 'train', **common)
        val_ds = PIEConvLSTMDataset(imdb, 'val', **common)
    else:
        common = {
            'observe_length': cfg.observe_length,
            'backbone': cfg.backbone,
            'seq_overlap_rate': cfg.seq_overlap_rate,
            'normalize_bbox': cfg.normalize_bbox,
            'balance_classes': cfg.balance_classes,
            'balance_seed': cfg.balance_seed,
        }
        train_ds = PIEIntentionDataset(imdb, 'train', **common)
        val_ds = PIEIntentionDataset(imdb, 'val', **common)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    return train_ds, train_dl, val_dl

def compute_pos_weight(dataset):
    labels = np.array([track['label'] for track in dataset.tracks])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    return torch.tensor([pos_weight], dtype=torch.float)


def run_epoch(model, loader, criterion, optimizer, device, train, grad_clip=0.0):
    model.train() if train else model.eval()
    torch.set_grad_enabled(train)

    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        features = batch['features'].to(device)
        bbox = batch['bbox'].to(device)
        labels = batch['label'].float().to(device)

        if train:
            optimizer.zero_grad()

        logit = model(features, bbox)
        loss = criterion(logit, labels)

        if train:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

    torch.set_grad_enabled(True)
    return total_loss / max(n_samples, 1)


METRIC_KEYS = ['accuracy', 'majority_acc', 'precision', 'recall',
               'f1', 'auc', 'ap', 'nll', 'ece']

# 'accuracy' added: this is the metric the reference PIE paper actually
# reports (keras metrics=['acc']); every other key here is on top of it,
# not a replacement for it.
PLOTTED_METRIC_KEYS = ['accuracy', 'auc', 'f1', 'ap', 'precision', 'recall', 'nll', 'ece']

RUN_CONFIG_KEYS = [
    'observe_length',
    'backbone',
    'seq_overlap_rate',
    'input_dropout',
    # FIX: 'variational_input_dropout' removed -- the current config.py no
    # longer defines this field (variational dropout is now hardwired to
    # BaseIntentionModel's default of True, no longer tunable per-run).
    # Referencing it here would raise AttributeError building entry['config']
    # on epoch 1 for EVERY run, gru and convlstm_encdec alike. If you want
    # this toggleable again, add the field back to Config and restore it
    # here; until then it's just removed from the logged snapshot.
    'rmsprop_alpha',
    'monitor',
    'monitor_mode',
    'early_stop_patience',
    'min_lr',
    'gru_dropout',
    'optimizer',
]


class LivePlotter:
    def __init__(self, enabled, title='', total_epochs=1):
        self.enabled = bool(enabled)
        self.title = title
        self.total_epochs = max(1, int(total_epochs))
        self._handle = None
        if not self.enabled:
            return
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                raise RuntimeError
            import matplotlib.pyplot as plt          # noqa: F401
            from IPython.display import display     # noqa: F401
        except Exception:
            self.enabled = False

    def update(self, history):
        if not self.enabled or not history:
            return
        import matplotlib.pyplot as plt
        from IPython.display import display

        fig = _plot_history(history, self.title, total_epochs=self.total_epochs)
        if self._handle is None:
            self._handle = display(fig, display_id=True)
        else:
            self._handle.update(fig)
        plt.close(fig)


def _set_full_range_xlim(axes, total_epochs):
    for ax in axes:
        ax.set_xlim(1, total_epochs)


def _plot_history(history, title='', total_epochs=1):
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]

    def get(key):
        return [h[key] for h in history]

    # 3x3 grid: Loss, Accuracy(+majority baseline), AUC, F1, AP, Precision,
    # Recall, NLL, ECE -- 9 panels total (was 2x4=8; Accuracy is the new one).
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    flat_axes = axes.flatten()

    def plot_train_val(ax, key, ptitle, ylim01=True):
        ax.plot(epochs, get(f'train_{key}'), label='train', color='tab:blue')
        ax.plot(epochs, get(f'val_{key}'), label='val', color='tab:orange')
        ax.set_title(ptitle); ax.set_xlabel('epoch')
        if ylim01:
            ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 0. Loss
    ax = flat_axes[0]
    ax.plot(epochs, get('train_loss'), label='train', color='tab:blue')
    ax.plot(epochs, get('val_loss'), label='val', color='tab:orange')
    ax.set_title('Loss'); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 1. Accuracy (raw, threshold=0.5) + majority-class baseline. This is
    # the metric the reference PIE paper reports and compares against
    # (~0.79 accuracy) -- shown first/prominently, right after Loss, since
    # every other panel in this grid is a diagnostic ON TOP of this one,
    # not a replacement for it.
    ax = flat_axes[1]
    ax.plot(epochs, get('train_accuracy'), label='train', color='tab:blue')
    ax.plot(epochs, get('val_accuracy'), label='val', color='tab:orange')
    ax.plot(epochs, get('train_majority_acc'), label='train majority',
            color='tab:blue', linestyle=':', alpha=0.6)
    ax.plot(epochs, get('val_majority_acc'), label='val majority',
            color='tab:orange', linestyle=':', alpha=0.6)
    ax.set_title('Accuracy (raw, thresh=0.5) vs majority baseline')
    ax.set_xlabel('epoch'); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 2. AUC
    ax = flat_axes[2]
    plot_train_val(ax, 'auc', 'AUC')
    ax.axhline(0.5, color='red', linewidth=0.8, linestyle=':', label='chance')
    ax.legend(fontsize=8)

    plot_train_val(flat_axes[3], 'f1', 'F1')
    plot_train_val(flat_axes[4], 'ap', 'Average Precision (AP)')
    plot_train_val(flat_axes[5], 'precision', 'Precision')
    plot_train_val(flat_axes[6], 'recall', 'Recall')
    plot_train_val(flat_axes[7], 'nll', 'NLL', ylim01=False)
    plot_train_val(flat_axes[8], 'ece', 'ECE', ylim01=False)

    _set_full_range_xlim(flat_axes, total_epochs)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_curves(history, out_dir, title='', total_epochs=1):
    if not history:
        return None
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt

    fig = _plot_history(history, title, total_epochs=total_epochs)
    path = os.path.join(out_dir, 'curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ensure_isolated_out_dir(out_dir):
    stale_files = ['best.pt', 'best.pt.tmp', 'history.json', 'curves.png', 'test_metrics.json']
    found = [f for f in stale_files if os.path.isfile(os.path.join(out_dir, f))]
    if found:
        print(f"[isolation] stale artifacts found in {out_dir}: {found} -- removing them so this run starts clean.")
        for f in found:
            os.remove(os.path.join(out_dir, f))


def main(cfg):
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(cfg.out_dir, exist_ok=True)
    _ensure_isolated_out_dir(cfg.out_dir)

    print(f"device={device}     model={cfg.model_name}      backbone={cfg.backbone}      obs={cfg.observe_length}")
    print(f"out_dir={cfg.out_dir}")

    train_ds, train_dl, val_dl = make_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"params: total={model.num_params():,}    encoder={model.encoder_params():,}")

    pos_weight = compute_pos_weight(train_ds).to(device)
    print(f"pos_weight={pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if cfg.model_name == 'convlstm_encdec':
        # FIX: the reference applies L2(reg_value) to the ConvLSTM encoder
        # and decoder LSTM ONLY -- the Dense head has NO regularizer in
        # pie_convlstm_encdec(). Passing model.parameters() as one group
        # (as the else-branch does) would apply weight_decay to the head
        # too, which the reference never does. Use the model's own
        # per-group split instead.
        param_groups = model.reference_param_groups(weight_decay_regularized=2 * cfg.weight_decay)
        optimizer = torch.optim.RMSprop(param_groups, lr=cfg.lr,
                                        alpha=cfg.rmsprop_alpha, eps=cfg.rmsprop_eps)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr,
                                        alpha=cfg.rmsprop_alpha, eps=cfg.rmsprop_eps,
                                        weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=cfg.monitor_mode, factor=cfg.sched_factor,
        patience=cfg.sched_patience, min_lr=cfg.min_lr)

    best = -np.inf if cfg.monitor_mode == 'max' else np.inf
    bad_epochs = 0
    history = []
    plotter = LivePlotter(getattr(cfg, 'live_plot', False), title=cfg.run_tag(), total_epochs=cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_dl, criterion, optimizer, device, train=True, grad_clip=cfg.grad_clip)
        val_loss = run_epoch(model, val_dl, criterion, optimizer, device, train=False)

        train_probs, train_labels = collect_predictions(model, train_dl, device)
        train_metrics = compute_metrics(train_probs, train_labels, threshold=0.5)

        val_probs, val_labels = collect_predictions(model, val_dl, device)
        val_metrics = compute_metrics(val_probs, val_labels, threshold=0.5)

        val_cal_threshold = calibrate_threshold(val_probs, val_labels, metric='accuracy')
        val_metrics_cal = compute_metrics(val_probs, val_labels, threshold=val_cal_threshold)

        monitored = val_metrics['auc'] if cfg.monitor == 'val_auc' else val_loss
        scheduler.step(monitored)

        if cfg.monitor_mode == 'max':
            improved = monitored > best + cfg.early_stop_min_delta
        else:
            improved = monitored < best - cfg.early_stop_min_delta

        if improved:
            best = monitored
            bad_epochs = 0
            ckpt_data = {
                'model_state': model.state_dict(),
                'cfg': cfg.__dict__,
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'val_calibrated_threshold': val_cal_threshold,
                'val_metrics_calibrated': val_metrics_cal,
            }
            tmp_path = os.path.join(cfg.out_dir, 'best.pt.tmp')
            final_path = os.path.join(cfg.out_dir, 'best.pt')
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, final_path)
        else:
            bad_epochs += 1

        entry = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                 'lr': optimizer.param_groups[0]['lr'],
                 'val_calibrated_threshold': val_cal_threshold,
                 'val_calibrated_accuracy': val_metrics_cal['accuracy'],
                 'val_calibrated_f1': val_metrics_cal['f1']}
        for k in METRIC_KEYS:
            entry[f'train_{k}'] = train_metrics[k]
            entry[f'val_{k}'] = val_metrics[k]

        if not history:
            entry['config'] = {k: getattr(cfg, k) for k in RUN_CONFIG_KEYS}

        history.append(entry)
        plotter.update(history)

        if epoch % 5 == 0 or improved:
            flag = '*' if improved else ''
            print(f"ep {epoch:>3}   tr_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {flag}")
            print(f"  TRAIN  acc={train_metrics['accuracy']:.3f} (maj={train_metrics['majority_acc']:.3f})  "
                  f"F1={train_metrics['f1']:.3f}  AUC={train_metrics['auc']:.3f}  AP={train_metrics['ap']:.3f}  "
                  f"prec={train_metrics['precision']:.3f}  rec={train_metrics['recall']:.3f}")
            print(f"  VAL    acc={val_metrics['accuracy']:.3f} (maj={val_metrics['majority_acc']:.3f})  "
                  f"F1={val_metrics['f1']:.3f}  AUC={val_metrics['auc']:.3f}  AP={val_metrics['ap']:.3f}  "
                  f"prec={val_metrics['precision']:.3f}  rec={val_metrics['recall']:.3f}")
            print(f"  NLL/ECE   train={train_metrics['nll']:.3f}/{train_metrics['ece']:.3f}   "
                  f"val={val_metrics['nll']:.3f}/{val_metrics['ece']:.3f}   "
                  f"val_calibrated(acc@t={val_cal_threshold:.2f})={val_metrics_cal['accuracy']:.3f}   "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}\n")

        if bad_epochs >= cfg.early_stop_patience:
            print(f"early stop at epoch:{epoch}")
            break

    with open(os.path.join(cfg.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    png = save_curves(history, cfg.out_dir, title=cfg.run_tag(), total_epochs=cfg.epochs)
    if png:
        print(f"curves -> {png}")

    best_entry = max(history, key=lambda h: h['val_auc']) if cfg.monitor == 'val_auc' \
        else min(history, key=lambda h: h['val_loss'])
    print(f"\nbest-> {cfg.monitor}={best:.4f}  (epoch {best_entry['epoch']})")
    print(f"  TRAIN  accuracy={best_entry['train_accuracy']:.4f}  (majority_baseline={best_entry['train_majority_acc']:.4f})  "
          f"F1={best_entry['train_f1']:.4f}  AUC={best_entry['train_auc']:.4f}  AP={best_entry['train_ap']:.4f}")
    print(f"  VAL    accuracy={best_entry['val_accuracy']:.4f}  (majority_baseline={best_entry['val_majority_acc']:.4f})  "
          f"F1={best_entry['val_f1']:.4f}  AUC={best_entry['val_auc']:.4f}  AP={best_entry['val_ap']:.4f}")
    print(f"  val_calibrated accuracy={best_entry['val_calibrated_accuracy']:.4f}  "
          f"(threshold={best_entry['val_calibrated_threshold']:.2f})")
    print(f"  precision/recall  train={best_entry['train_precision']:.4f}/{best_entry['train_recall']:.4f}   "
          f"val={best_entry['val_precision']:.4f}/{best_entry['val_recall']:.4f}")
    print(f"  NLL/ECE           train={best_entry['train_nll']:.4f}/{best_entry['train_ece']:.4f}   "
          f"val={best_entry['val_nll']:.4f}/{best_entry['val_ece']:.4f}")
    print(f"  checkpoint -> {os.path.join(cfg.out_dir, 'best.pt')}")
    return history


def parse_args(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=defaults.model_name)
    parser.add_argument('--backbone', type=str, default=defaults.backbone)
    parser.add_argument('--observe_length', type=int, default=defaults.observe_length)
    parser.add_argument('--seq_overlap_rate', type=float, default=defaults.seq_overlap_rate)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--no_live_plot', action='store_true')

    args, _unknown = parser.parse_known_args(argv)

    cfg = Config(
        model_name=args.model_name,
        backbone=args.backbone,
        observe_length=args.observe_length,
        seq_overlap_rate=args.seq_overlap_rate,
        lr=args.lr,
        epochs=args.epochs,
        out_dir=args.out_dir,
        use_timestamp=not args.no_timestamp,
        live_plot=not args.no_live_plot,
    )
    return cfg


if __name__ == '__main__':
    main(parse_args())