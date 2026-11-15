"""
compare_helpers.py

Notebook entry for the old evaluate_and_compare.py script.
All comparison logic lives here. The notebook only sets PIE_PATH and calls
run_comparison().

Set os.environ['PIE_PATH'] before importing this module.
"""

import os
import sys

PIE_PATH = os.environ.get('PIE_PATH')
if not PIE_PATH:
    raise RuntimeError(
        "Set os.environ['PIE_PATH'] BEFORE importing compare_helpers.")

import csv
import gc
import glob
import hashlib
import json
import math
import pickle
import shutil
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pie_data_Hesam import PIE
from pie_intent_experiment import PIEIntentExperiment
from pie_intent_helpers import compute_full_metrics
from train_helpers import Tee
from run_selection_helpers import parse_indices
from eval_results_io import load_predictions_dict

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FRAMEWORK = 'tensorflow'
EXPERIMENTS_ROOT = os.path.join(PIE_PATH, 'experiments')
OFFICIAL_MODEL_DIR = r'D:\PIE codes\PIE\runs\downloaded'
OUTPUT_ROOT = os.path.join(PIE_PATH, 'evaluation_comparisons')

SAVED_PRED_CANDIDATES = ['ped_intents_test.pkl', 'ped_intents.pkl']  # preferred first
VISUALIZE = False                      # test_chunk(visualize=...) only matters when re-running
SHOW_PLOTS = 'ipykernel' in sys.modules  # plt.show() blocks plain scripts
SEED_NOISE_AUC = 0.02                  # AUC differences below this are within seed noise
OFFICIAL_NORMALIZE_BBOX = False        # reference code feeds raw pixel bbox to the decoder
DEFAULT_NORMALIZE_BBOX = False         # raw bbox when hyperparams.json has no flag
IGNORE_DIFF_KEYS = {'data.balance'}    # dead key (BALANCE_MODE is the real control)
NON_METRIC_KEYS = {'threshold', 'tp', 'tn', 'fp', 'fn', 'nll', 'roc_auc',
                   'threshold_used', 'positive_rate_gt'}
INTRO_MAX_KEYS = 8                     # differing config keys named in the summary sentence

_CONFIGURABLE = {
    'OFFICIAL_MODEL_DIR', 'VISUALIZE', 'SEED_NOISE_AUC',
    'OFFICIAL_NORMALIZE_BBOX', 'DEFAULT_NORMALIZE_BBOX',
    'SAVED_PRED_CANDIDATES', 'INTRO_MAX_KEYS', 'IGNORE_DIFF_KEYS',
}
_CONSOLE = {}


def configure(**kwargs):
    """Override module constants from the notebook before run_comparison()."""
    unknown = sorted(set(kwargs) - _CONFIGURABLE)
    if unknown:
        raise KeyError(f"not configurable: {unknown}. Allowed: {sorted(_CONFIGURABLE)}")
    globals().update(kwargs)


def start_console_log(log_path):
    """Copy stdout/stderr into log_path. A previous redirect is undone first."""
    stop_console_log()
    fid = open(log_path, 'w', encoding='utf-8')
    _CONSOLE.update(stdout=sys.stdout, stderr=sys.stderr, file=fid)
    sys.stdout = Tee(sys.stdout, fid)
    sys.stderr = Tee(sys.stderr, fid)


def stop_console_log():
    if not _CONSOLE:
        return
    sys.stdout, sys.stderr = _CONSOLE['stdout'], _CONSOLE['stderr']
    _CONSOLE['file'].close()
    _CONSOLE.clear()


# ------------------------------------------------------------------
# Interactive selection helpers
# ------------------------------------------------------------------
def interactive_pick_option(title, options, default=0):
    print(f"\n[{title}]")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}" + ("   <-- default" if i == default else ""))
    while True:
        raw = input(f"choose [0-{len(options) - 1}] (Enter = {default}): ").strip()
        if raw == '':
            return default
        if raw.isdigit() and int(raw) < len(options):
            return int(raw)
        print("  ! invalid choice")


def interactive_pick_architecture_filter():
    """
    Lists every architecture folder under EXPERIMENTS_ROOT/FRAMEWORK and lets
    the user pick which one(s) to pull checkpoints from (space/comma-separated
    indices, or 'all').
    Returns (selected_architectures, include_official, architecture_was_all).
    """
    framework_dir = os.path.join(EXPERIMENTS_ROOT, FRAMEWORK)
    architectures = []
    if os.path.isdir(framework_dir):
        architectures = sorted(
            d for d in os.listdir(framework_dir)
            if os.path.isdir(os.path.join(framework_dir, d))
        )

    entries = []
    if os.path.isdir(OFFICIAL_MODEL_DIR):
        entries.append(('official', None))
    for arch in architectures:
        entries.append(('arch', arch))

    if not entries:
        raise FileNotFoundError(
            f"Nothing to select from -- no official model at {OFFICIAL_MODEL_DIR} "
            f"and no architecture folders under {framework_dir}."
        )

    print(f"\n[architecture] available sources under {framework_dir} "
          f"(and the official pretrained model, if present):")
    for i, (kind, arch) in enumerate(entries):
        if kind == 'official':
            print(f"  [{i}] (official pretrained model)   -- {OFFICIAL_MODEL_DIR}")
        else:
            arch_dir = os.path.join(framework_dir, arch)
            n_loadable = sum(
                1 for d in glob.glob(os.path.join(arch_dir, '*'))
                if os.path.isdir(d)
                and os.path.isfile(os.path.join(d, 'model.h5'))
                and os.path.isfile(os.path.join(d, 'configs.pkl'))
            )
            print(f"  [{i}] {arch}   ({n_loadable} loadable checkpoint(s))")

    architecture_was_all = False
    while True:
        raw = input(
            "\n[architecture] enter the indices of the source(s) you want to pull "
            "checkpoints from (space- or comma-separated, e.g. 0 2  or  0,2), "
            "or type 'all' to include everything listed above: "
        ).strip()
        if raw.lower() == 'all':
            indices = list(range(len(entries)))
            architecture_was_all = True
            break
        try:
            indices = parse_indices(raw, len(entries) - 1)
        except ValueError as e:
            print(f"  ! {e}")
            continue
        break

    include_official = False
    selected_architectures = []
    for idx in indices:
        kind, arch = entries[idx]
        if kind == 'official':
            include_official = True
        else:
            selected_architectures.append(arch)

    print(f"\n[architecture selected] official={include_official}  "
          f"architectures={selected_architectures}")
    return selected_architectures, include_official, architecture_was_all


def gather_all_model_dirs(architecture_filter=None, include_official=True):
    """Flat list of (model_dir, tag): the official model (optional) plus every
    completed checkpoint (model.h5 + configs.pkl) under the chosen architectures,
    newest first."""
    items = []
    if include_official and os.path.isdir(OFFICIAL_MODEL_DIR):
        items.append((OFFICIAL_MODEL_DIR, 'official'))

    framework_dir = os.path.join(EXPERIMENTS_ROOT, FRAMEWORK)
    if os.path.isdir(framework_dir):
        architectures = sorted(
            d for d in os.listdir(framework_dir)
            if os.path.isdir(os.path.join(framework_dir, d))
        )
        if architecture_filter is not None:
            allowed = set(architecture_filter)
            architectures = [a for a in architectures if a in allowed]

        for arch in architectures:
            arch_dir = os.path.join(framework_dir, arch)
            all_dirs = [d for d in glob.glob(os.path.join(arch_dir, '*')) if os.path.isdir(d)]
            loadable = [
                d for d in all_dirs
                if os.path.isfile(os.path.join(d, 'model.h5'))
                and os.path.isfile(os.path.join(d, 'configs.pkl'))
            ]
            loadable.sort(key=os.path.getctime, reverse=True)
            for d in loadable:
                items.append((d, f"{arch}/{os.path.basename(d)}"))
    return items


def sanitize_tag_for_filename(tag):
    safe = tag.replace('/', '__').replace('\\', '__')
    for ch in ':*?"<>|':
        safe = safe.replace(ch, '_')
    return safe


def split_tag(tag):
    """'arch/run_name' -> (arch, run_name). The official model has no folder level."""
    if '/' in tag:
        arch, run = tag.split('/', 1)
    else:
        arch, run = tag, tag
    if arch.startswith('official'):
        run = os.path.basename(OFFICIAL_MODEL_DIR)
    return arch, run


def interactive_pick_multiple_model_dirs():
    """Two-step selection (architecture filter, then checkpoints).
    Returns (selected [(model_dir, tag), ...], run_was_all)."""
    selected_architectures, include_official, architecture_was_all = interactive_pick_architecture_filter()

    items = gather_all_model_dirs(
        architecture_filter=selected_architectures,
        include_official=include_official,
    )
    if not items:
        raise FileNotFoundError("No models found for the architecture source(s) you selected.")

    print("\n[models] checkpoints available from the selected source(s):")
    for i, (model_dir, tag) in enumerate(items):
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(model_dir)))
            ts_str = f"   (created {ts})"
        except OSError:
            ts_str = ""
        print(f"  [{i}] {tag}{ts_str}")

    models_was_all = False
    while True:
        raw = input(
            "\n[models] enter the indices of the models you want to compare "
            "(space- or comma-separated, e.g. 0 1 2 5 8  or  0,1,2,5,8), "
            "or type 'all' to select every model listed above: "
        ).strip()
        if raw.lower() == 'all':
            indices = list(range(len(items)))
            models_was_all = True
            break
        try:
            indices = parse_indices(raw, len(items) - 1)
        except ValueError as e:
            print(f"  ! {e}")
            continue
        break

    selected = []
    existing_tags = set()
    for idx in indices:
        model_dir, tag = items[idx]
        base_tag, n = tag, 2
        while tag in existing_tags:
            tag = f"{base_tag}_{n}"
            n += 1
        existing_tags.add(tag)
        selected.append((model_dir, tag))

    print(f"\n[selected] {[t for _, t in selected]}")
    return selected, (architecture_was_all or models_was_all)


# ------------------------------------------------------------------
# Run metadata: configs, hyperparams, history, saved split metrics
# ------------------------------------------------------------------
_SCALARS = (int, float, str, bool, type(None))


def flatten_config(obj, prefix):
    """Flatten nested dict/list config into {'a.b.c': scalar-or-short-string}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_config(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        if len(obj) <= 16 and all(isinstance(x, _SCALARS) for x in obj):
            out[prefix[:-1]] = str(list(obj))
        else:
            out[prefix[:-1]] = f"<{type(obj).__name__} len={len(obj)}>"
    elif isinstance(obj, np.generic):
        out[prefix[:-1]] = obj.item()
    elif isinstance(obj, _SCALARS):
        out[prefix[:-1]] = obj
    else:
        out[prefix[:-1]] = f"<{type(obj).__name__}>"
    return out


def summarize_history(model_dir):
    out = {'epochs': None, 'best_epoch': None, 'best_monitor': None, 'monitor_key': None}
    path = os.path.join(model_dir, 'history.pkl')
    if not os.path.isfile(path):
        return out
    try:
        with open(path, 'rb') as fid:
            hist = pickle.load(fid)
        if hasattr(hist, 'history'):
            hist = hist.history
        if not isinstance(hist, dict):
            return out
        for key, mode in (('val_auc', 'max'), ('val_roc_auc', 'max'), ('val_loss', 'min')):
            if key in hist and len(hist[key]) > 0:
                arr = np.asarray(hist[key], dtype=float)
                idx = int(np.nanargmax(arr) if mode == 'max' else np.nanargmin(arr))
                out.update(epochs=int(len(arr)), best_epoch=idx + 1,
                           best_monitor=float(arr[idx]), monitor_key=key)
                break
    except Exception as e:
        print(f"  ! could not summarize history.pkl: {e}")
    return out


def load_saved_split_metrics(model_dir):
    """metrics_<split>.json written by the main training script (if present)."""
    out = {}
    for split in ('train', 'val', 'test'):
        path = os.path.join(model_dir, f'metrics_{split}.json')
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as fid:
                    out[split] = json.load(fid)
            except Exception as e:
                print(f"  ! could not read {path}: {e}")
    return out


def pick_auc(metrics_dict):
    if not isinstance(metrics_dict, dict):
        return None
    for k in ('roc_auc', 'auc'):
        v = metrics_dict.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def read_run_meta(model_dir):
    meta = {'data_opts': {}, 'configs_flat': {},
            'history': summarize_history(model_dir),
            'saved_metrics': load_saved_split_metrics(model_dir)}
    flat = {}
    try:
        with open(os.path.join(model_dir, 'configs.pkl'), 'rb') as fid:
            model_config, train_config, data_opts = pickle.load(fid)
        meta['data_opts'] = data_opts
        flat.update(flatten_config(model_config, 'model.'))
        flat.update(flatten_config(train_config, 'train.'))
        flat.update(flatten_config(data_opts, 'data.'))
    except Exception as e:
        print(f"  ! could not read configs.pkl in {model_dir}: {e}")

    hp_path = os.path.join(model_dir, 'hyperparams.json')
    if os.path.isfile(hp_path):
        try:
            with open(hp_path, 'r', encoding='utf-8') as fid:
                flat.update(flatten_config(json.load(fid), 'hp.'))
        except Exception as e:
            print(f"  ! could not read hyperparams.json: {e}")
    meta['configs_flat'] = flat
    return meta


def resolve_normalize_flag(model_dir, tag):
    """normalize_decoder_bbox must match what the run was TRAINED with."""
    hp_path = os.path.join(model_dir, 'hyperparams.json')
    if os.path.isfile(hp_path):
        try:
            with open(hp_path, 'r', encoding='utf-8') as fid:
                hp = json.load(fid)
            if 'normalize_decoder_bbox' in hp:
                return bool(hp['normalize_decoder_bbox']), 'hyperparams.json'
        except Exception:
            pass
    if tag == 'official' or tag.startswith('official'):
        return OFFICIAL_NORMALIZE_BBOX, 'official default (raw pixel bbox)'
    return DEFAULT_NORMALIZE_BBOX, 'WARNING: hyperparams.json has no flag -> raw bbox'


def diff_configs(flat_by_tag):
    keys = sorted(set().union(*[set(f) for f in flat_by_tag.values()]))
    diff = {}
    for k in keys:
        if k in IGNORE_DIFF_KEYS or any(s in k.lower() for s in ('path', 'dir')):
            continue
        vals = {t: f.get(k, '<absent>') for t, f in flat_by_tag.items()}
        if len({json.dumps(v, default=str) for v in vals.values()}) > 1:
            diff[k] = vals
    return diff


def read_val_threshold(model_dir):
    """Look for a validation-derived threshold in calibration.json.
    Schema is not assumed: first numeric value in (0,1) under a key containing 'thresh'.
    Returns (value, key_path) or None."""
    path = os.path.join(model_dir, 'calibration.json')
    if not os.path.isfile(path):
        return None

    def walk(obj, trail):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{trail}.{k}" if trail else str(k)
                if ('thresh' in str(k).lower() and isinstance(v, (int, float))
                        and not isinstance(v, bool) and 0.0 < float(v) < 1.0):
                    return float(v), p
            for k, v in obj.items():
                r = walk(v, f"{trail}.{k}" if trail else str(k))
                if r:
                    return r
        return None

    try:
        with open(path, 'r', encoding='utf-8') as fid:
            cal = json.load(fid)
        found = walk(cal, '')
        if found is None:
            print(f"  ! no threshold-like key in calibration.json (top keys: "
                  f"{list(cal)[:10] if isinstance(cal, dict) else type(cal).__name__})")
        return found
    except Exception as e:
        print(f"  ! could not read calibration.json: {e}")
        return None


# ------------------------------------------------------------------
# Predictions: reuse saved file, or run inference via test_chunk()
# ------------------------------------------------------------------
def find_saved_predictions(model_dir):
    for name in SAVED_PRED_CANDIDATES:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path
    return None


def read_predictions(path):
    probs, labels = load_predictions_dict(path)
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if probs.shape != labels.shape:
        raise ValueError(f"probs {probs.shape} and labels {labels.shape} differ in shape")
    return probs, labels


def run_inference(model_dir, tag, data_opts, imdb_cache, out_dir):
    """Run test_chunk() for this run. Any existing ped_intents.pkl is MOVED to
    <out_dir>/backup_ped_intents/ (never deleted) so a fresh file is guaranteed."""
    if not data_opts:
        raise RuntimeError("data_opts missing (configs.pkl unreadable) -- cannot rebuild test split.")

    if '_imdb' not in imdb_cache:
        imdb_cache['_imdb'] = PIE(data_path=PIE_PATH)

    # data_opts holds lists (e.g. height_rng) -> canonical JSON string as cache key
    key = json.dumps(data_opts, sort_keys=True, default=str)
    if key not in imdb_cache:
        print("  building test sequence for this run's data_opts...")
        imdb_cache[key] = imdb_cache['_imdb'].generate_data_trajectory_sequence('test', **data_opts)
    beh_seq_test = imdb_cache[key]

    flag, why = resolve_normalize_flag(model_dir, tag)
    print(f"  [normalize_decoder_bbox] {flag}  ({why})")

    experiment = PIEIntentExperiment(
        framework='tensorflow', architecture='eval_compare', run_name='eval_compare',
        run_dir=model_dir,
    )
    experiment.normalize_decoder_bbox = flag

    results_path = os.path.join(model_dir, 'ped_intents.pkl')
    if os.path.exists(results_path):
        backup_dir = os.path.join(out_dir, 'backup_ped_intents')
        os.makedirs(backup_dir, exist_ok=True)
        dst = os.path.join(backup_dir, f"{sanitize_tag_for_filename(tag)}__ped_intents_"
                                       f"{time.strftime('%H%M%S')}.pkl")
        shutil.move(results_path, dst)
        print(f"  [backup] old ped_intents.pkl moved -> {dst}")

    acc, f1 = experiment.test_chunk(beh_seq_test, data_opts, model_dir,
                                    visualize=VISUALIZE, vis_num_correct=10,
                                    vis_num_incorrect=10)
    print(f"  test_chunk() (threshold=0.5) -> acc={acc:.4f}  f1={f1:.4f}")

    if not os.path.isfile(results_path):
        raise RuntimeError(f"test_chunk() finished but {results_path} was not written.")
    return read_predictions(results_path), results_path


# ------------------------------------------------------------------
# Metrics on FIXED probabilities
# ------------------------------------------------------------------
def safe_auc(labels, probs):
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return None


def evaluate_at_threshold(probs, labels, t):
    preds = (probs >= t).astype(int)
    return {
        'threshold': float(t),
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
    }


def sweep_thresholds(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [evaluate_at_threshold(probs, labels, t) for t in thresholds]
    auc = safe_auc(labels, probs)
    for r in rows:
        r['auc'] = auc
    return rows


def row_at(sweep_rows, target_t):
    return min(sweep_rows, key=lambda r: abs(r['threshold'] - target_t))


def best_threshold_row(sweep_rows, metric='f1'):
    return max(sweep_rows, key=lambda r: r[metric])


def get_full_metrics(probs, labels, tag, out_dir):
    """compute_full_metrics once on the cached probs; saved next to the sweep plot."""
    metrics_path = os.path.join(out_dir, f'full_metrics_{sanitize_tag_for_filename(tag)}.json')
    try:
        full_metrics = compute_full_metrics(labels, probs, save_path=metrics_path)
    except TypeError:
        # signature without save_path -> save the JSON ourselves
        full_metrics = compute_full_metrics(labels, probs)
        with open(metrics_path, 'w', encoding='utf-8') as fid:
            json.dump(full_metrics, fid, indent=2, default=float)
    print(f"  [full metrics] saved -> {metrics_path}")
    for k, v in (full_metrics or {}).items():
        print(f"    {k}: {v}")
    return full_metrics or {}


def build_summary_row(idx, tag, source, probs, labels, sweep, full_metrics, meta, val_thr):
    p = float(labels.mean())
    at05 = row_at(sweep, 0.5)
    oracle = best_threshold_row(sweep, 'f1')
    saved = meta['saved_metrics']
    hist = meta['history']
    arch, run_name = split_tag(tag)
    row = {
        'idx': idx, 'tag': tag, 'architecture': arch, 'run_name': run_name, 'source': source,
        'n_windows': int(labels.size), 'pos_rate': p,
        'majority_acc': max(p, 1.0 - p),
        'allpos_f1': (2 * p / (1 + p)) if p > 0 else 0.0,
        'auc': at05['auc'],
        'acc@0.5': at05['accuracy'], 'f1@0.5': at05['f1'],
        'ece': full_metrics.get('ece'), 'nll': full_metrics.get('nll'),
        'val_thr': None, 'acc@val_thr': None, 'f1@val_thr': None,
        'oracle_thr(TEST-TUNED)': oracle['threshold'],
        'f1@oracle(TEST-TUNED)': oracle['f1'],
        'epochs': hist['epochs'], 'best_epoch': hist['best_epoch'],
        'monitor_key': hist['monitor_key'], 'best_monitor': hist['best_monitor'],
        'train_auc_saved': pick_auc(saved.get('train')),
        'val_auc_saved': pick_auc(saved.get('val')),
        'test_auc_saved': pick_auc(saved.get('test')),
        'test_fp': hashlib.md5(labels.tobytes()).hexdigest()[:8],
    }
    if val_thr is not None:
        r = evaluate_at_threshold(probs, labels, val_thr[0])
        row.update({'val_thr': val_thr[0], 'acc@val_thr': r['accuracy'], 'f1@val_thr': r['f1']})
    return row


# ------------------------------------------------------------------
# Console tables
# ------------------------------------------------------------------
def fmt(v, maxlen=None):
    if v is None:
        return '-'
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}" if np.isfinite(v) else 'nan'
    s = str(v)
    if maxlen and len(s) > maxlen:
        s = s[:maxlen - 3] + '...'
    return s


def print_table(rows, cols, title, maxlen=46):
    print(f"\n{title}")
    widths = [max(len(c), *(len(fmt(r.get(c), maxlen)) for r in rows)) for c in cols]
    print('  ' + ' | '.join(c.ljust(w) for c, w in zip(cols, widths)))
    print('  ' + '-+-'.join('-' * w for w in widths))
    for r in rows:
        print('  ' + ' | '.join(fmt(r.get(c), maxlen).ljust(w) for c, w in zip(cols, widths)))


def print_auc_deltas(rows):
    ref = rows[0]
    if len(rows) < 2 or ref['auc'] is None:
        return
    print(f"\n[delta AUC vs run [{ref['idx']}]]  "
          f"(|delta| < {SEED_NOISE_AUC:.2f} is within seed noise, ~0.010 std per project notes)")
    for r in rows[1:]:
        if r['auc'] is None:
            continue
        d = r['auc'] - ref['auc']
        verdict = 'within noise' if abs(d) < SEED_NOISE_AUC else ('better' if d > 0 else 'worse')
        same = '' if r['test_fp'] == ref['test_fp'] and r['n_windows'] == ref['n_windows'] \
            else '   (DIFFERENT TEST SET -- not comparable)'
        print(f"  [{r['idx']}] {d:+.4f}  {verdict}{same}")


# ------------------------------------------------------------------
# Plotting (legend always in its own row BELOW the axes)
# ------------------------------------------------------------------
def finish_fig(fig, path):
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def legend_columns(n_entries):
    return 1 if n_entries <= 10 else (2 if n_entries <= 40 else 3)


def legend_height(n_entries, ncol):
    return 0.25 + 0.17 * math.ceil(n_entries / ncol)


def draw_legend_below(ax_source, ax_leg, ncol, fontsize=6.5):
    handles, labels = ax_source.get_legend_handles_labels()
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, loc='upper center', ncol=ncol, fontsize=fontsize,
                  frameon=False, borderaxespad=0.0, handlelength=1.2, columnspacing=1.5)


def plot_threshold_sweep(sweep_rows, tag, out_dir):
    ths = [r['threshold'] for r in sweep_rows]
    fig = plt.figure(figsize=(9, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(ths, [r[metric] for r in sweep_rows], marker='o', markersize=3, label=metric)
    best = best_threshold_row(sweep_rows, 'f1')
    plt.axvline(best['threshold'], color='gray', linestyle='--', alpha=0.5,
                label=f"ORACLE best F1 @ t={best['threshold']:.2f} (tuned on TEST)")
    plt.xlabel('decision threshold')
    plt.ylabel('metric value')
    plt.title(f'Threshold sweep on TEST -- {tag}', fontsize=9)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f'threshold_sweep_{sanitize_tag_for_filename(tag)}.png')
    finish_fig(fig, path)
    return path


def _bounded_metric_names(rows_by_tag):
    names = []
    for row in rows_by_tag.values():
        for k, v in row.items():
            if k in NON_METRIC_KEYS or k.startswith('n_') or k in names:
                continue
            if isinstance(v, (float, np.floating)) and np.isfinite(v) and 0.0 <= v <= 1.0:
                names.append(k)
    return names


def plot_comparison(all_results, summary_rows, out_dir):
    """Three grouped bar charts: fixed 0.5, val-calibrated threshold (runs that have
    one), and the ORACLE test-tuned threshold (diagnostic only). Legend sits in a
    separate row below the plot, so it never covers the bars."""
    tags = list(all_results.keys())
    row_by_tag = {r['tag']: r for r in summary_rows}

    modes = [
        ('fixed', 'comparison_at_threshold_0.5.png', 'Comparison at threshold = 0.5'),
        ('val', 'comparison_at_val_threshold.png',
         'Comparison at VAL-calibrated threshold (from calibration.json)'),
        ('oracle', 'comparison_at_oracle_threshold.png',
         'ORACLE: threshold tuned on TEST (diagnostic only, NOT reportable)'),
    ]
    for mode, out_name, title in modes:
        rows_by_tag = {}
        for tag in tags:
            res = all_results[tag]
            sweep = res['sweep']
            if mode == 'fixed':
                row = dict(row_at(sweep, 0.5))
                for k, v in (res.get('full_metrics') or {}).items():
                    if k not in row and k not in NON_METRIC_KEYS:
                        row[k] = v
                row['majority_acc'] = row_by_tag[tag]['majority_acc']
            elif mode == 'val':
                if res.get('val_thr') is None:
                    continue
                row = evaluate_at_threshold(res['probs'], res['labels'], res['val_thr'][0])
            else:
                row = dict(best_threshold_row(sweep, 'f1'))
            rows_by_tag[tag] = row

        if not rows_by_tag:
            print(f"  ! no data for '{out_name}' (no run has a usable val threshold?), skipping.")
            continue
        metric_names = _bounded_metric_names(rows_by_tag)
        if not metric_names:
            continue

        plot_tags = list(rows_by_tag.keys())
        n_runs = len(plot_tags)
        fig_w = min(24, max(10, 1.2 * len(metric_names) + 0.5 * n_runs))
        if n_runs > 10:
            fig_w = max(fig_w, 18)
        ncol = legend_columns(n_runs)
        leg_h = legend_height(n_runs, ncol)
        plot_h = 6.0
        fig, (ax, ax_leg) = plt.subplots(
            2, 1, figsize=(fig_w, plot_h + leg_h),
            gridspec_kw={'height_ratios': [plot_h, leg_h]})

        x = np.arange(len(metric_names))
        width = 0.8 / max(n_runs, 1)
        for i, tag in enumerate(plot_tags):
            row = rows_by_tag[tag]
            vals = [row.get(m, 0.0) if isinstance(row.get(m), (float, np.floating)) else 0.0
                    for m in metric_names]
            lab = f"[{row_by_tag[tag]['idx']}] {tag}"
            if mode != 'fixed':
                lab += f" (t={row['threshold']:.2f})"
            ax.bar(x + i * width, vals, width=width, label=lab)
        ax.set_xticks(x + width * (n_runs - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=30, ha='right')
        ax.set_ylabel('value')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        draw_legend_below(ax, ax_leg, ncol)
        fig.tight_layout(h_pad=1.5)
        finish_fig(fig, os.path.join(out_dir, out_name))


def plot_all_models_overlay(all_results, summary_rows, out_dir):
    tags = list(all_results.keys())
    idx_of = {r['tag']: r['idx'] for r in summary_rows}
    metric_names = ['accuracy', 'precision', 'recall', 'f1']  # AUC is constant across thresholds
    n_m = len(metric_names)
    n_runs = len(tags)
    ncol = legend_columns(n_runs)
    leg_h = legend_height(n_runs, ncol)
    fig_w = 12 if n_runs <= 10 else 18
    fig = plt.figure(figsize=(fig_w, 4.5 * n_m + leg_h))
    gs = fig.add_gridspec(n_m + 1, 1, height_ratios=[4.5] * n_m + [leg_h])
    axes = []
    for i in range(n_m):
        axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None))
    ax_leg = fig.add_subplot(gs[n_m, 0])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, metric in zip(axes, metric_names):
        for i, tag in enumerate(tags):
            sweep = all_results[tag]['sweep']
            ax.plot([r['threshold'] for r in sweep], [r[metric] for r in sweep],
                    marker='o', markersize=3, label=f"[{idx_of[tag]}] {tag}",
                    color=colors[i % len(colors)])
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs threshold -- all selected models (TEST)', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('decision threshold')
    draw_legend_below(axes[0], ax_leg, ncol)
    fig.tight_layout(h_pad=1.5)
    path = os.path.join(out_dir, 'all_models_overlay.png')
    finish_fig(fig, path)
    return path


# ------------------------------------------------------------------
# Final comparison table: summary sentence + one row per run, every output a column
# ------------------------------------------------------------------
def to_py(v):
    if isinstance(v, np.floating):
        v = float(v)
    elif isinstance(v, np.integer):
        v = int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (list, tuple, dict, set)):
        return str(v)
    return v


def collect_numeric(d, prefix):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[f"{prefix}{k}"] = to_py(v)
    return out


def short_val(v, n=30):
    s = f"{v:.3g}" if isinstance(v, float) else str(v)
    return s if len(s) <= n else s[:n - 3] + '...'


def build_final_rows(summary_rows, all_results, diff, flat_by_tag):
    ref = summary_rows[0]
    rows = []
    for r in summary_rows:
        tag = r['tag']
        res = all_results[tag]
        meta = res['meta']
        at05 = row_at(res['sweep'], 0.5)
        oracle = best_threshold_row(res['sweep'], 'f1')
        same_test = (r['test_fp'] == ref['test_fp'] and r['n_windows'] == ref['n_windows'])

        row = {
            'idx': r['idx'], 'architecture': r['architecture'], 'run_name': r['run_name'],
            'tag': tag, 'model_dir': res['model_dir'], 'predictions_source': r['source'],
            'n_windows': r['n_windows'], 'pos_rate': r['pos_rate'],
            'majority_acc': r['majority_acc'], 'allpos_f1': r['allpos_f1'],
            'test_fp': r['test_fp'],
            'same_test_as_first': same_test,
        }
        for k in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
            row[f'test@0.5_{k}'] = at05[k]

        d = None if (r['auc'] is None or ref['auc'] is None) else r['auc'] - ref['auc']
        row['delta_auc_vs_first'] = d
        if d is None:
            row['auc_vs_first_verdict'] = ''
        elif not same_test:
            row['auc_vs_first_verdict'] = 'n/a (different test set)'
        elif r['idx'] == ref['idx']:
            row['auc_vs_first_verdict'] = 'reference'
        elif abs(d) < SEED_NOISE_AUC:
            row['auc_vs_first_verdict'] = 'within noise'
        else:
            row['auc_vs_first_verdict'] = 'better' if d > 0 else 'worse'

        vt = res.get('val_thr')
        if vt:
            m = evaluate_at_threshold(res['probs'], res['labels'], vt[0])
            row.update({'val_thr': vt[0], 'val_thr_source_key': vt[1],
                        'val_thr_accuracy': m['accuracy'], 'val_thr_precision': m['precision'],
                        'val_thr_recall': m['recall'], 'val_thr_f1': m['f1']})
        else:
            row.update({'val_thr': None, 'val_thr_source_key': None, 'val_thr_accuracy': None,
                        'val_thr_precision': None, 'val_thr_recall': None, 'val_thr_f1': None})

        row.update({
            'ORACLE_TEST_TUNED_thr': oracle['threshold'],
            'ORACLE_TEST_TUNED_accuracy': oracle['accuracy'],
            'ORACLE_TEST_TUNED_precision': oracle['precision'],
            'ORACLE_TEST_TUNED_recall': oracle['recall'],
            'ORACLE_TEST_TUNED_f1': oracle['f1'],
        })

        row.update(collect_numeric(res.get('full_metrics'), 'full_'))

        hist = meta['history']
        row.update({'epochs_run': hist['epochs'], 'best_epoch': hist['best_epoch'],
                    'monitor_key': hist['monitor_key'], 'best_monitor_value': hist['best_monitor']})
        for split in ('train', 'val', 'test'):
            row.update(collect_numeric(meta['saved_metrics'].get(split), f'saved_{split}_'))

        for k in diff:
            row[f'cfg:{k}'] = flat_by_tag[tag].get(k, '<absent>')

        rows.append({k: to_py(v) for k, v in row.items()})
    return rows


def build_intro_sentence(summary_rows, diff):
    n = len(summary_rows)
    arch_runs = {}
    for r in summary_rows:
        arch_runs.setdefault(r['architecture'], []).append(r['idx'])
    parts = []
    for a, ix in arch_runs.items():
        shown = ','.join(str(i) for i in ix[:8]) + (',...' if len(ix) > 8 else '')
        parts.append(f"{a} ({len(ix)} run{'s' if len(ix) != 1 else ''}: [{shown}])")
    s = (f"Compared {n} run{'s' if n != 1 else ''} from {len(arch_runs)} source folder"
         f"{'s' if len(arch_runs) != 1 else ''}: " + '; '.join(parts) + '.')

    if diff:
        def deviation(k):
            vals = [json.dumps(v, default=str) for v in diff[k].values()]
            return len(vals) - Counter(vals).most_common(1)[0][1]
        keys = sorted(diff, key=lambda k: (-deviation(k), k))
        chunks = []
        for k in keys[:INTRO_MAX_KEYS]:
            distinct = []
            for v in diff[k].values():
                sv = short_val(v)
                if sv not in distinct:
                    distinct.append(sv)
            more = '' if len(distinct) <= 4 else f", +{len(distinct) - 4} more"
            chunks.append(f"{k} = {{{', '.join(distinct[:4])}{more}}}")
        s += f" They differ in {len(diff)} config key(s); most varied: " + '; '.join(chunks)
        if len(diff) > INTRO_MAX_KEYS:
            s += f"; {len(diff) - INTRO_MAX_KEYS} more are listed in the 'config_diff' sheet / config_diff.json"
        s += '.'
    else:
        s += ' Their configs are identical.'

    groups = {}
    for r in summary_rows:
        groups.setdefault((r['n_windows'], r['test_fp']), []).append(r)
    if len(groups) == 1:
        s += f" All runs were scored on the same {summary_rows[0]['n_windows']} test windows."
    else:
        s += (f" WARNING: runs were scored on {len(groups)} different test sets "
              f"(window count / label sequence differ), so cross-run acc/F1/AUC are NOT directly comparable.")

    largest = max(groups.values(), key=len)
    with_auc = [r for r in largest if r['auc'] is not None]
    if len(with_auc) >= 2:
        hi = max(with_auc, key=lambda r: r['auc'])
        lo = min(with_auc, key=lambda r: r['auc'])
        s += (f" Test AUC over the largest comparable group ({len(largest)} runs): "
              f"max [{hi['idx']}] {hi['auc']:.4f}, min [{lo['idx']}] {lo['auc']:.4f} "
              f"(spread {hi['auc'] - lo['auc']:.4f}; differences below {SEED_NOISE_AUC:.2f} are within seed noise).")
    return s


def ordered_columns(rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def save_final_table(rows, sentence, diff, id_of, out_dir):
    cols = ordered_columns(rows)

    csv_path = os.path.join(out_dir, 'final_comparison_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fid:
        writer = csv.DictWriter(fid, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved final comparison table (CSV)  -> {csv_path}")

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        txt_path = os.path.join(out_dir, 'final_comparison_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as fid:
            fid.write(sentence + '\n')
        print(f"  ! openpyxl not installed (pip install openpyxl) -> no .xlsx; "
              f"summary sentence saved to {txt_path}")
        return csv_path

    wb = Workbook()
    ws = wb.active
    ws.title = 'comparison'
    ws.cell(row=1, column=1, value=sentence)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=min(len(cols), 14))
    ws['A1'].alignment = Alignment(wrap_text=True, vertical='top')
    ws.row_dimensions[1].height = 15 * (len(sentence) // 150 + 2)

    header_row = 3
    head_fill = PatternFill('solid', fgColor='DDEBF7')
    for j, c in enumerate(cols, start=1):
        cell = ws.cell(row=header_row, column=j, value=c)
        cell.font = Font(bold=True)
        cell.fill = head_fill
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    for i, r in enumerate(rows, start=header_row + 1):
        for j, c in enumerate(cols, start=1):
            v = r.get(c)
            cell = ws.cell(row=i, column=j, value=v)
            if isinstance(v, float):
                cell.number_format = '0.0000'
    for j, c in enumerate(cols, start=1):
        longest = max([len(c)] + [len(str(r.get(c, ''))) for r in rows])
        ws.column_dimensions[get_column_letter(j)].width = min(longest + 2, 45)
    ws.freeze_panes = ws.cell(row=header_row + 1, column=4)
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(cols))}{header_row + len(rows)}"

    ws2 = wb.create_sheet('config_diff')
    run_idx = [id_of[t] for t in next(iter(diff.values()))] if diff else []
    ws2.cell(row=1, column=1, value='config key').font = Font(bold=True)
    for j, i in enumerate(run_idx, start=2):
        c = ws2.cell(row=1, column=j, value=f'[{i}]')
        c.font = Font(bold=True)
    for i, (k, vals) in enumerate(diff.items(), start=2):
        ws2.cell(row=i, column=1, value=k)
        for j, v in enumerate(vals.values(), start=2):
            ws2.cell(row=i, column=j, value=to_py(v) if not isinstance(v, str) else v)
    ws2.column_dimensions['A'].width = 50
    ws2.freeze_panes = 'B2'

    xlsx_path = os.path.join(out_dir, 'final_comparison_table.xlsx')
    wb.save(xlsx_path)
    print(f"Saved final comparison table (XLSX) -> {xlsx_path}")
    return xlsx_path


FINAL_CONSOLE_COLS = ['idx', 'architecture', 'test@0.5_auc', 'test@0.5_accuracy', 'test@0.5_f1',
                      'full_ece', 'full_nll', 'val_thr', 'val_thr_f1', 'best_epoch',
                      'delta_auc_vs_first', 'auc_vs_first_verdict']


def run_comparison(selections=None, force_rerun=None):
    """Interactive by default. Menus run before the log file is opened."""
    import tensorflow as tf
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    run_was_all = False
    if selections is None:
        selections, run_was_all = interactive_pick_multiple_model_dirs()
    if force_rerun is None:
        force_rerun = interactive_pick_option(
            'predictions',
            ['reuse saved ped_intents*.pkl when available (no inference); re-run only if missing',
             'always re-run inference (old ped_intents.pkl is backed up, not deleted)'],
            default=0,
        ) == 1

    run_stamp = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    base_out_dir = os.path.join(OUTPUT_ROOT, run_stamp)
    out_dir = os.path.join(base_out_dir, 'all') if run_was_all else base_out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[config] saving all outputs under: {out_dir}")

    log_path = os.path.join(out_dir, f"console_{run_stamp}.txt")
    start_console_log(log_path)
    try:
        return _run_comparison_body(
            selections, force_rerun, run_was_all, run_stamp, out_dir, log_path, tf)
    finally:
        stop_console_log()


def _run_comparison_body(selections, force_rerun, run_was_all, run_stamp, out_dir, log_path, tf):
    print(f"[config] console log -> {log_path}")
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"  ! could not set memory growth on {gpu}: {e}")

    with open(os.path.join(out_dir, 'selection_manifest.json'), 'w', encoding='utf-8') as fid:
        json.dump({
            'run_stamp': run_stamp, 'run_was_all': run_was_all, 'force_rerun': force_rerun,
            'selection_order': [{'index': i, 'tag': t, 'model_dir': d}
                                for i, (d, t) in enumerate(selections)],
        }, fid, indent=2)

    imdb_cache = {}
    all_results = {}
    summary_rows = []
    flat_by_tag = {}

    for i, (model_dir, tag) in enumerate(selections):
        print(f"\n=== [{i}] {tag}  ({model_dir}) ===")
        meta = read_run_meta(model_dir)
        flat_by_tag[tag] = meta['configs_flat']
        used_inference = False

        probs = labels = None
        source = None
        pred_path = None if force_rerun else find_saved_predictions(model_dir)
        if pred_path:
            try:
                probs, labels = read_predictions(pred_path)
                source = f"saved:{os.path.basename(pred_path)}"
                age_h = (time.time() - os.path.getmtime(pred_path)) / 3600.0
                print(f"  [reuse] {pred_path}  (modified {age_h:.1f} h ago, n={labels.size})")
                model_h5 = os.path.join(model_dir, 'model.h5')
                if os.path.isfile(model_h5) and os.path.getmtime(model_h5) > os.path.getmtime(pred_path) + 1:
                    print("  [WARNING] model.h5 is NEWER than the saved predictions -> they may be stale; "
                          "re-run with option [1] to be sure.")
            except Exception as e:
                print(f"  ! could not read saved predictions ({e}); falling back to inference.")
                probs = labels = None

        if probs is None:
            (probs, labels), _ = run_inference(model_dir, tag, meta['data_opts'], imdb_cache, out_dir)
            source = 'inference'
            used_inference = True

        sweep = sweep_thresholds(probs, labels)
        best = best_threshold_row(sweep, 'f1')
        print(f"  ORACLE best F1 (tuned on TEST, diagnostic only) = {best['f1']:.4f} "
              f"at t={best['threshold']:.2f}")
        plot_threshold_sweep(sweep, tag, out_dir)

        full_metrics = get_full_metrics(probs, labels, tag, out_dir)
        val_thr = read_val_threshold(model_dir)
        if val_thr:
            print(f"  [val threshold] {val_thr[0]:.4f}  (calibration.json key: {val_thr[1]})")

        all_results[tag] = {'sweep': sweep, 'full_metrics': full_metrics, 'model_dir': model_dir,
                            'probs': probs, 'labels': labels, 'val_thr': val_thr, 'meta': meta}
        summary_rows.append(build_summary_row(i, tag, source, probs, labels, sweep,
                                              full_metrics, meta, val_thr))

        if used_inference:
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"  [cleanup] cleared TF session after '{tag}'")

    groups = {}
    for r in summary_rows:
        groups.setdefault((r['n_windows'], r['test_fp']), []).append(r['idx'])
    if len(groups) > 1:
        print("\n[WARNING] the selected runs were NOT evaluated on the same test windows "
              "(different count or label sequence) -- acc/F1/AUC are NOT apples-to-apples:")
        for (n, fp), idxs in groups.items():
            print(f"    n_windows={n}  test_fp={fp}  runs={idxs}")

    print("\nLEGEND")
    for r in summary_rows:
        print(f"  [{r['idx']}] {r['tag']}   ({r['source']})")

    print_table(summary_rows,
                ['idx', 'n_windows', 'pos_rate', 'majority_acc', 'allpos_f1', 'auc',
                 'acc@0.5', 'f1@0.5', 'ece', 'nll'],
                "TEST @ threshold 0.5   (acc/F1 only mean something if above majority_acc/allpos_f1; AUC is the main metric)")
    print_table(summary_rows,
                ['idx', 'val_thr', 'acc@val_thr', 'f1@val_thr',
                 'oracle_thr(TEST-TUNED)', 'f1@oracle(TEST-TUNED)'],
                "THRESHOLD CHOICE   (val_thr = from calibration.json; oracle columns are tuned on TEST -> diagnostic only)")
    print_table(summary_rows,
                ['idx', 'epochs', 'best_epoch', 'monitor_key', 'best_monitor',
                 'train_auc_saved', 'val_auc_saved', 'test_auc_saved'],
                "TRAINING STATE   (from history.pkl and metrics_<split>.json; large train-val gap = overfit)")
    print_auc_deltas(summary_rows)

    diff = diff_configs(flat_by_tag)
    id_of = {r['tag']: r['idx'] for r in summary_rows}
    print(f"\nCONFIG DIFFERENCES  ({len(diff)} differing key(s); full list in config_diff.json)")
    for k in list(diff)[:60]:
        vals = ', '.join(f"[{id_of[t]}]={v}" for t, v in diff[k].items())
        print(f"  {k}: {vals}")
    if len(diff) > 60:
        print(f"  ... {len(diff) - 60} more in config_diff.json")

    with open(os.path.join(out_dir, 'config_diff.json'), 'w', encoding='utf-8') as fid:
        json.dump(diff, fid, indent=2, default=str)
    with open(os.path.join(out_dir, 'threshold_sweeps.json'), 'w', encoding='utf-8') as fid:
        json.dump({t: {'model_dir': r['model_dir'], 'sweep': r['sweep']}
                   for t, r in all_results.items()}, fid, indent=2)
    with open(os.path.join(out_dir, 'full_metrics_all.json'), 'w', encoding='utf-8') as fid:
        json.dump({t: {'model_dir': r['model_dir'], 'full_metrics': r['full_metrics']}
                   for t, r in all_results.items()}, fid, indent=2, default=float)

    if len(all_results) > 1:
        plot_comparison(all_results, summary_rows, out_dir)
        plot_all_models_overlay(all_results, summary_rows, out_dir)
    else:
        print("\nOnly one model selected -- skipping multi-model comparison plots.")

    final_rows = build_final_rows(summary_rows, all_results, diff, flat_by_tag)
    sentence = build_intro_sentence(summary_rows, diff)

    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    print(sentence)
    cfg_cols = [c for c in ordered_columns(final_rows) if c.startswith('cfg:')][:3]
    console_cols = [c for c in FINAL_CONSOLE_COLS if any(c in r for r in final_rows)] + cfg_cols
    print_table(final_rows, console_cols,
                "FINAL TABLE (compact view; the full table with every output column is in the saved file)",
                maxlen=40)
    save_final_table(final_rows, sentence, diff, id_of, out_dir)

    print(f"\nAll outputs saved under: {out_dir}")
    return {'out_dir': out_dir, 'rows': final_rows,
            'summary_rows': summary_rows, 'results': all_results}
