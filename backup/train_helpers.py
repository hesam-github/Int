"""
train_helpers.py

Helper utilities extracted from train_pie_intent_tensorflow_gpu.py, to keep
the main training script focused on the actual training flow.

Contains:
  - format_lr_tag: formats a learning rate as a short folder-name tag.
  - Interactive prompt helpers: interactive_pick_architecture,
    interactive_pick_new_run_name, interactive_pick_balance_mode,
    interactive_pick_monitor_metric,
    interactive_pick_data_split (plus their small private helpers
    _is_valid_name / _list_existing_run_dirs / _carve_validation_split).
  - build_split_sequences: wraps generate_data_trajectory_sequence() for
    all three splits, including the manual val-carve needed for kfold.
  - Tee: duplicates stdout/stderr writes to both the console and a log file.
  - compute_full_metrics: computes the full metrics dict (accuracy,
    precision, recall, f1, roc_auc, pr_auc, nll, ece, brier_score,
    confusion_matrix, etc.) at a given decision threshold.
  - print_full_history_table: pretty-prints the per-epoch history table
    from history.pkl.

NOTE: No logic was changed while moving these out of the main script --
only the location changed.

`focal_loss` / `weighted_binary_crossentropy` imports needed by
interactive_pick_loss_function are imported here directly from
focal_loss.py.
"""
import re
import os
import glob
import json
import numpy as np

from focal_loss import binary_focal_loss, weighted_binary_crossentropy
from log_mode import set_log_mode, phase

from augmentation import AUGMENTATION_CFG, _describe_aug_cfg


def find_kfold_run_dirs(experiments_root, framework, architecture):
    """
    Scans experiments_root/framework/architecture/ for run folders whose
    name contains a kfold tag ('kfold<N>_f<K>', produced by
    build_data_split_tag for data_split_type='kfold'), sorted by fold.

    :return: list of dicts: {'path', 'folder_name', 'num_folds', 'fold'}
    """
    architecture_dir = os.path.join(experiments_root, framework, architecture)
    if not os.path.isdir(architecture_dir):
        return []

    # BUGFIX: the previous pattern ended in \b, which never matches here --
    # '_' counts as a \w character in Python regex, so there is no word
    # boundary between the fold digit and the '_' that always follows it
    # in real run names (build_run_hyperparam_tag always appends more
    # tag segments right after split_tag). \d+ is already greedy, so
    # dropping \b does not risk misreading e.g. 'f10' as 'f1'.
    pattern = re.compile(r'kfold(\d+)_f(\d+)')
    matches = []
    for name in sorted(os.listdir(architecture_dir)):
        full_path = os.path.join(architecture_dir, name)
        if not os.path.isdir(full_path):
            continue
        m = pattern.search(name)
        if m is None:
            continue
        matches.append({
            'path': full_path, 'folder_name': name,
            'num_folds': int(m.group(1)), 'fold': int(m.group(2)),
        })

    matches.sort(key=lambda d: (d['fold'], d['path']))
    return matches


def find_all_kfold_run_dirs(experiments_root):
    """
    Discovers every kfold-tagged run folder anywhere under experiments_root,
    across ALL frameworks and ALL architecture folders -- nothing about
    which architecture to look in is hardcoded or passed in, since
    different kfold runs may legitimately live under different
    architecture folders (different hyperparameters -> different
    DEFAULT_ARCHITECTURE tag, or a manually-typed architecture name).

    Reuses find_kfold_run_dirs() per (framework, architecture) pair
    instead of re-implementing the matching logic separately.

    :return: list of dicts, each a find_kfold_run_dirs() entry plus
             'framework' and 'architecture' keys.
    """
    all_matches = []
    if not os.path.isdir(experiments_root):
        return all_matches

    for framework in sorted(os.listdir(experiments_root)):
        framework_dir = os.path.join(experiments_root, framework)
        if not os.path.isdir(framework_dir):
            continue
        for architecture in sorted(os.listdir(framework_dir)):
            architecture_dir = os.path.join(framework_dir, architecture)
            if not os.path.isdir(architecture_dir):
                continue
            runs = find_kfold_run_dirs(experiments_root, framework, architecture)
            for r in runs:
                r['framework'] = framework
                r['architecture'] = architecture
            all_matches.extend(runs)

    all_matches.sort(key=lambda d: (d['framework'], d['architecture'], d['fold'], d['path']))
    return all_matches

def interactive_pick_ensemble_folds(architecture_dir, name_filter=None):
    """
    Lists existing run folders under architecture_dir (optionally only
    those whose name contains name_filter, e.g. 'kfold') with a numeric
    index, and lets you pick MULTIPLE folds to ensemble -- instead of
    hardcoding fold_dirs by hand every time. Same numbered-list pattern
    as interactive_pick_balance_mode() / interactive_pick_clip_mode().

    Enter comma-separated indices ('0,1,2,3,4'), a range ('0-4'), or
    press Enter / type 'all' to select every listed folder (default).

    :return: list of full run_dir paths, in listed (oldest-first) order
    """
    existing = _list_existing_run_dirs(architecture_dir)
    if name_filter:
        existing = [d for d in existing if name_filter in os.path.basename(d)]
    if not existing:
        raise FileNotFoundError(
            f"No run folders found under {architecture_dir}"
            + (f" matching name_filter={name_filter!r}" if name_filter else ""))

    print(f"\n[ensemble] existing run folders under {architecture_dir}"
          + (f" (filtered: {name_filter!r} in name)" if name_filter else "") + ":")
    for i, d in enumerate(existing):
        print(f"  [{i}] {os.path.basename(d)}")

    while True:
        raw = input(
            "[ensemble] enter indices to include (comma-separated, e.g. "
            "'0,1,2,3,4'; a range like '0-4'; or 'all') [default: all]: "
        ).strip()

        if raw == '' or raw.lower() == 'all':
            selected = list(range(len(existing)))
        elif '-' in raw and ',' not in raw:
            try:
                lo, hi = raw.split('-')
                selected = list(range(int(lo), int(hi) + 1))
            except Exception:
                print("  ! invalid range, try again (e.g. '0-4').")
                continue
        else:
            try:
                selected = [int(x.strip()) for x in raw.split(',') if x.strip() != '']
            except ValueError:
                print("  ! invalid input -- enter numbers separated by commas, a range, or 'all'.")
                continue

        if not selected:
            print("  ! select at least one fold.")
            continue
        if any(i < 0 or i >= len(existing) for i in selected):
            print(f"  ! all indices must be in 0-{len(existing) - 1}.")
            continue

        chosen = [existing[i] for i in selected]
        print(f"[ensemble] selected {len(chosen)} run(s):")
        for d in chosen:
            print(f"    {os.path.basename(d)}")
        return chosen


def interactive_pick_kfold_aggregate(experiments_root):
    """
    Post-training (or standalone) interactive aggregator. Discovers EVERY
    kfold-tagged run folder anywhere under experiments_root -- no
    framework/architecture is passed in or hardcoded, see
    find_all_kfold_run_dirs() -- groups them by (framework, architecture),
    lets you pick which GROUP to aggregate, then within that group which
    specific fold run(s) to combine. Prints mean +/- std across
    train/val/test at both threshold=0.5 and the calibrated threshold.

    Reads only already-saved metrics_*.json files -- no retraining, no
    re-inference.

    Callable two ways:
      1. Right after a kfold run finishes (main script):
             if DATA_SPLIT_TYPE == 'kfold':
                 interactive_pick_kfold_aggregate(EXPERIMENTS_ROOT)
      2. Standalone, in a fresh notebook/session, once several folds
         (possibly under different architecture folders) already exist --
         only experiments_root is needed.
    """
    answer = input("\n[kfold-aggregate] Aggregate kfold runs (mean +/- std "
                    "across folds)? [y/N] ").strip().lower()
    if answer != 'y':
        return None

    print(f"[kfold-aggregate] scanning under {experiments_root} for "
          f"kfold-tagged run folders...")
    all_runs = find_all_kfold_run_dirs(experiments_root)
    if not all_runs:
        print(f"[kfold-aggregate] No kfold-tagged run folders found anywhere "
              f"under {experiments_root}.")
        return None

    groups = []
    for r in all_runs:
        key = (r['framework'], r['architecture'])
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(r)

    print(f"\n[kfold-aggregate] found {len(groups)} architecture group(s) "
          f"with kfold-tagged runs:")
    for i, ((framework, architecture), runs) in enumerate(groups):
        folds = sorted(r['fold'] for r in runs)
        print(f"  [{i}] {framework}/{architecture}  "
              f"({len(runs)} fold-run(s) found: {folds})")

    while True:
        raw = input(f"[kfold-aggregate] pick a group index [0-{len(groups)-1}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(groups):
            group_idx = int(raw)
            break
        print(f"  ! please enter a number 0-{len(groups)-1}.")

    (framework, architecture), runs = groups[group_idx]

    print(f"\n[kfold-aggregate] runs in {framework}/{architecture}:")
    for i, r in enumerate(runs):
        print(f"  [{i}] fold={r['fold']}/{r['num_folds']}  {r['folder_name']}")

    sel_raw = input("  Pick indices to combine, comma-separated "
                     "(e.g. 0,1,2,3,4) [default: all]: ").strip()
    indices = ([int(x.strip()) for x in sel_raw.split(',') if x.strip()]
               if sel_raw else list(range(len(runs))))

    bad = [i for i in indices if i < 0 or i >= len(runs)]
    assert not bad, f"index out of range: {bad} (valid range: 0..{len(runs)-1})"

    selected = [runs[i] for i in indices]
    selected_paths = [r['path'] for r in selected]
    fold_numbers = sorted(r['fold'] for r in selected)
    print(f"\n[kfold-aggregate] Combining {len(selected)} run(s) from "
          f"{framework}/{architecture}, folds={fold_numbers}")
    if len(set(fold_numbers)) != len(fold_numbers):
        print("[warning] the same fold number appears more than once in your "
              "selection -- make sure that's intentional (e.g. comparing two "
              "hyperparameter choices on the same fold), not an accidental duplicate.")

    results = {}
    for split in ('train', 'val', 'test'):
        for calibrated in (False, True):
            summary, n_used = aggregate_kfold_metrics(selected_paths, split, calibrated)
            print_kfold_summary_table(
                f"KFOLD AGGREGATE ({len(selected)} folds requested)",
                summary, n_used, len(selected), split, calibrated)
            results[(split, calibrated)] = summary
    return results

def _load_metrics_json(run_dir, split, calibrated=False):
    """Loads metrics_<split>[_calibrated].json. Returns None (with a
    warning) instead of raising, so one incomplete run doesn't crash the
    whole aggregation."""
    suffix = '_calibrated' if calibrated else ''
    path = os.path.join(run_dir, f'metrics_{split}{suffix}.json')
    if not os.path.isfile(path):
        print(f"[kfold-aggregate] WARNING: missing {path} -- skipped.")
        return None
    with open(path, 'r', encoding='utf-8') as fid:
        return json.load(fid)


_AGGREGATE_METRIC_KEYS = [
    'accuracy', 'majority_acc', 'precision', 'recall', 'f1',
    'roc_auc', 'pr_auc', 'nll', 'ece', 'brier_score',
]


def aggregate_kfold_metrics(run_dirs, split='test', calibrated=False):
    """
    Loads metrics_<split>[_calibrated].json from each run_dir, computes
    mean +/- sample std (ddof=1) across runs for every scalar metric.
    Runs with a missing file are skipped (not counted).

    :return: (summary_dict {metric: (mean, std)}, n_used)
    """
    per_run = [m for m in (_load_metrics_json(rd, split, calibrated) for rd in run_dirs)
               if m is not None]
    n_used = len(per_run)

    summary = {}
    for key in _AGGREGATE_METRIC_KEYS:
        values = [m[key] for m in per_run
                  if key in m and m[key] is not None
                  and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if not values:
            summary[key] = (float('nan'), float('nan'))
            continue
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = (mean, std)
    return summary, n_used


def print_kfold_summary_table(title, summary, n_used, n_requested, split, calibrated):
    """Prints a mean +/- std table, always showing how many runs actually
    contributed vs. how many were requested (never hides a partial result)."""
    tag = f"{split}{' (calibrated)' if calibrated else ' (threshold=0.5)'}"
    print("\n" + "=" * 70)
    print(f"{title} -- {tag}")
    print(f"[note] aggregated over {n_used}/{n_requested} requested runs "
          f"(mean +/- sample std, ddof=1).")
    if n_used < n_requested:
        print(f"[warning] {n_requested - n_used} run(s) skipped (missing "
              f"metrics file) -- this is NOT a complete k-fold result.")
    print("=" * 70)
    print(f"{'metric':>16} {'mean':>10} {'std':>10}")
    for key, (mean, std) in summary.items():
        print(f"{key:>16} {mean:>10.4f} {std:>10.4f}")
    print("=" * 70)

def interactive_pick_log_mode():
    LOG_MODE_OPTIONS = [
        ('clean', 'phase explanations + meaningful metrics only (recommended)'),
        ('full',  'everything, including raw config dumps (original behavior)'),
    ]
    default_idx = 0
    print("\n[log_mode] choose console verbosity for this run:", flush=True)
    for i, (value, desc) in enumerate(LOG_MODE_OPTIONS):
        marker = "  <-- default" if i == default_idx else ""
        print(f"  [{i}] {value}  ({desc}){marker}", flush=True)
    while True:
        raw = input(
            f"[log_mode] enter a number from the list above "
            f"[default: {default_idx} -> '{LOG_MODE_OPTIONS[default_idx][0]}']: "
        ).strip()
        if raw == "":
            mode = LOG_MODE_OPTIONS[default_idx][0]
        elif raw.isdigit() and 0 <= int(raw) < len(LOG_MODE_OPTIONS):
            mode = LOG_MODE_OPTIONS[int(raw)][0]
        else:
            print(f"  ! please enter a number 0-{len(LOG_MODE_OPTIONS) - 1}.", flush=True)
            continue
        set_log_mode(mode)
        return mode


# ------------------------------------------------------------------
# Phase narration messages -- one named constant per pipeline stage.
# Edit the wording HERE only; the training script just references the
# constant name, so it stays short and reviewable.
# ------------------------------------------------------------------
PHASE_BUILD_SEQUENCES = (
    "Building train/val/test sequences",
    "Loads or builds the (train, val, test) pedestrian tracks according "
    "to data_split_type -- cached to disk by content hash, so repeated "
    "runs with identical settings are instant."
)

PHASE_BALANCING = (
    "Applying balance strategy",
    "Controls whether the RAW class imbalance in train/val is kept "
    "as-is (relying on the loss function to compensate), fixed by "
    "discarding majority-class samples (undersampling), or fixed by "
    "generating synthetic minority-class copies (augmentation)."
)

PHASE_LOSS_FINALIZE = (
    "Finalizing the loss function",
    "For weighted_bce, pos_weight is computed HERE from the TRAIN "
    "split's real, final class counts (after any balancing above) -- "
    "never from val/test, and never a hardcoded guess."
)

PHASE_TRAINING = (
    "Training",
    "Runs model.fit() with early stopping and checkpointing on the "
    "chosen monitor metric. Per-epoch train/val/test metrics printed "
    "below use the SAME overlap as the training windows -- these "
    "differ from the final evaluation numbers below, which always use "
    "overlap=1 (the paper's fixed evaluation protocol)."
)

PHASE_FULL_EVAL = (
    "Full per-split evaluation (train, val, test)",
    "Re-runs the BEST checkpoint over each split at overlap=1, saving "
    "predictions + full metrics to disk. This is the authoritative "
    "number for the run, not the per-epoch numbers printed during "
    "training."
)

PHASE_CALIBRATION = (
    "Threshold calibration",
    "Sweeps decision thresholds 0.05-0.95 and picks the one that "
    "maximizes F1 -- using VAL labels ONLY, never test -- since "
    "picking a threshold on test would leak test information into "
    "the reported numbers."
)

PHASE_RECOMPUTE_CALIBRATED = (
    "Recomputing all splits at the calibrated threshold",
    "Applies the val-selected threshold to predictions already saved "
    "above -- no re-inference, just re-thresholding the same "
    "probabilities, so this step is fast."
)


def print_phase(phase_tuple):
    """Call with one of the PHASE_* constants above, e.g. print_phase(PHASE_TRAINING)."""
    title, explanation = phase_tuple
    phase(title, explanation)



def format_lr_tag(lr):
    """Formats a learning rate as a short tag for folder names.
    1e-4 -> '1e-4', 1e-5 -> '1e-5', 5e-4 -> '5e-4'."""
    import math
    exponent = int(math.floor(math.log10(abs(lr))))
    mantissa = lr / (10 ** exponent)
    mantissa = int(round(mantissa)) if abs(mantissa - round(mantissa)) < 1e-9 else round(mantissa, 2)
    return f"{mantissa}e{exponent}"


# ------------------------------------------------------------------
# Small private helpers (run naming / dir listing)
# ------------------------------------------------------------------
def _is_valid_name(name):
    """Rejects empty names and names containing filesystem path separators
    or other characters that are invalid in a folder name on Windows."""
    return bool(name) and not any(c in name for c in r'\/:*?"<>|')


def _list_existing_run_dirs(architecture_dir):
    """Returns existing run folders under architecture_dir, oldest first.
    Empty list if the architecture folder doesn't exist yet (first run)."""
    if not os.path.isdir(architecture_dir):
        return []
    candidates = [d for d in glob.glob(os.path.join(architecture_dir, '*'))
                 if os.path.isdir(d)]
    return sorted(candidates, key=os.path.getctime)


# ------------------------------------------------------------------
# Data split selection (interactive) -- verified against
# aras62/PIE/utilities/pie_data.py, see interactive_pick_data_split()
# docstring for details on each option.
# ------------------------------------------------------------------
def interactive_pick_data_split():
    """
    Chooses data_opts['data_split_type'] (+ its sub-params) interactively,
    same numbered-list pattern as interactive_pick_balance_mode() /
    interactive_pick_monitor_metric(). Verified directly against
    aras62/PIE/utilities/pie_data.py -- nothing invented here.

    Three options pie_data.py actually supports:

      'default' -- PIE._get_image_set_ids(): FIXED split by video SET
          (hardcoded in pie_data.py, cannot be changed via params):
              train = ['set01', 'set02', 'set04']
              val   = ['set05', 'set06']
              test  = ['set03']
          This is what aras62/PIEPredict/train_test.py uses. It is NOT
          the split described in the ICCV 2019 paper text (which reports
          a random 50/40/10 pedestrian-level split on ~1842 samples) --
          it's the split the released CODE uses instead.

      'random' -- PIE._get_random_pedestrian_ids(): pools all 6 sets,
          splits by PEDESTRIAN ID via sklearn.train_test_split. This is
          the protocol matching the PAPER TEXT's stated 50/40/10 split.
          Default ratios [0.5, 0.4, 0.1] = train/test/val (NOTE: PIE's
          own ratio list order is [train, test, val], not [train, val,
          test] -- easy to misread, so this function labels it clearly
          when prompting). Cached to
          <PIE_PATH>/data_cache/random_samples.pkl on first run and
          REUSED on every later run with the same ratios unless you pass
          regen_data=True -- so once you generate it, the identical split
          is reproducible run to run (good for a fair before/after
          comparison across experiments).

      'kfold' -- PIE._get_kfold_pedestrian_ids(): sklearn KFold on
          pedestrian ids. PIE ONLY SUPPORTS 'train'/'test' for this mode
          (see the assert in _get_kfold_pedestrian_ids) -- calling it with
          image_set='val' raises immediately. This function returns a
          flag ('needs_manual_val': True) so the caller knows to carve a
          val split OUT of the train fold itself, rather than asking PIE
          for a 'val' split under kfold (which would crash).

    :return: (data_split_type: str, split_opts: dict, needs_manual_val: bool)
        split_opts is meant to be merged straight into data_opts, e.g.:
            data_opts.update(split_opts)
            data_opts['data_split_type'] = data_split_type
    """
    SPLIT_OPTIONS = [
        ('default',
         "fixed video-SET split: train=[set01,02,04] val=[set05,06] test=[set03] "
         "-- matches aras62/PIEPredict/train_test.py CODE (not the paper's stated "
         "random 50/40/10 text)"),
        ('random',
         "random PEDESTRIAN-id split, ratios you choose -- matches the paper "
         "TEXT's described 50/40/10 protocol"),
        ('kfold',
         "k-fold pedestrian-id split -- PIE only gives train/test per fold, "
         "val must be carved out of train manually (handled automatically below)"),
    ]
    default_idx = 0

    print("\n[data_split_type] existing split strategies (verified against pie_data.py):", flush=True)
    for i, (value, desc) in enumerate(SPLIT_OPTIONS):
        marker = "  <-- default" if i == default_idx else ""
        print(f"  [{i}] {value}  ({desc}){marker}", flush=True)

    while True:
        raw = input(
            f"[data_split_type] enter a number from the list above "
            f"[default: {default_idx} -> '{SPLIT_OPTIONS[default_idx][0]}']: "
        ).strip()
        if raw == "":
            idx = default_idx
        elif raw.isdigit() and 0 <= int(raw) < len(SPLIT_OPTIONS):
            idx = int(raw)
        else:
            print(f"  ! please enter a number 0-{len(SPLIT_OPTIONS) - 1}.", flush=True)
            continue
        break

    split_type = SPLIT_OPTIONS[idx][0]

    if split_type == 'default':
        return 'default', {}, False

    if split_type == 'random':
        print("\n[random_params] PIE's internal ratio ORDER is "
              "[train, test, val] -- not [train, val, test]. "
              "Paper text default: train=0.5 test=0.4 val=0.1.", flush=True)
        raw_ratios = input(
            "  enter three ratios as 'train,test,val' summing to 1.0 "
            "[default: 0.5,0.4,0.1]: "
        ).strip()
        if raw_ratios == "":
            ratios = [0.5, 0.4, 0.1]
        else:
            try:
                ratios = [float(x) for x in raw_ratios.split(',')]
                assert len(ratios) == 3
                assert abs(sum(ratios) - 1.0) < 1e-6
            except Exception:
                print("  ! invalid input, falling back to default 0.5,0.4,0.1")
                ratios = [0.5, 0.4, 0.1]

        regen_raw = input(
            "  regenerate the random split even if a cached one exists at "
            "data_cache/random_samples.pkl? [y/N]: "
        ).strip().lower()
        regen_data = regen_raw == 'y'

        split_opts = {
            'random_params': {
                'ratios': ratios,
                'val_data': True,
                'regen_data': regen_data,
            }
        }
        print(f"[data_split_type] random split -> ratios(train,test,val)={ratios} "
              f"regen_data={regen_data}", flush=True)
        return 'random', split_opts, False

    # kfold
    nf_raw = input("  [kfold] num_folds [default: 5]: ").strip()
    num_folds = int(nf_raw) if nf_raw else 5
    f_raw = input(f"  [kfold] which fold to use, 1..{num_folds} [default: 1]: ").strip()
    fold = int(f_raw) if f_raw else 1
    assert 1 <= fold <= num_folds, f"fold must be in 1..{num_folds}"

    val_frac_raw = input(
        "  [kfold] fraction of the TRAIN fold to carve off as val "
        "(PIE gives no val for kfold, so this is done manually) "
        "[default: 0.15]: "
    ).strip()
    val_fraction = float(val_frac_raw) if val_frac_raw else 0.15

    split_opts = {
        'kfold_params': {'num_folds': num_folds, 'fold': fold},
        '_manual_val_fraction': val_fraction,  # consumed by build_split_sequences(), not by PIE itself
    }
    print(f"[data_split_type] kfold split -> num_folds={num_folds} fold={fold} "
          f"manual_val_fraction={val_fraction}", flush=True)
    return 'kfold', split_opts, True


def build_split_sequences(imdb, data_opts):
    """
    Wraps imdb.generate_data_trajectory_sequence() for all three splits,
    including the kfold case where PIE itself cannot produce a 'val' split
    (see interactive_pick_data_split() docstring). data_opts must already
    contain 'data_split_type' and whatever split_opts interactive_pick_
    data_split() returned merged in.

    :return: beh_seq_train, beh_seq_val, beh_seq_test
    """
    split_type = data_opts['data_split_type']

    if split_type != 'kfold':
        print("Building train/val sequences...")
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        beh_seq_val   = imdb.generate_data_trajectory_sequence('val',   **data_opts)
        print("Building test sequence...")
        beh_seq_test  = imdb.generate_data_trajectory_sequence('test',  **data_opts)
        return beh_seq_train, beh_seq_val, beh_seq_test

    # kfold: PIE only supports 'train' and 'test' for this split type.
    val_fraction = data_opts.pop('_manual_val_fraction', 0.15)
    print("Building train sequence (kfold)...")
    beh_seq_train_full = imdb.generate_data_trajectory_sequence('train', **data_opts)
    print("Building test sequence (kfold)...")
    beh_seq_test        = imdb.generate_data_trajectory_sequence('test',  **data_opts)

    beh_seq_train, beh_seq_val = _carve_validation_split(
        beh_seq_train_full, val_fraction=val_fraction, seed=42)
    return beh_seq_train, beh_seq_val, beh_seq_test



def _carve_validation_split(beh_seq, val_fraction=0.15, seed=42):
    """
    Index-splits every list-valued key in beh_seq (each key is a list
    aligned by TRACK index -- one entry per pedestrian track, since
    generate_data_trajectory_sequence() returns whole tracks, not yet
    windowed) into a (1 - val_fraction) train part and val_fraction val
    part. Used only for the kfold path, where PIE provides no val split
    of its own.
    """
    n = len(beh_seq['image'])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = set(idx[:n_val].tolist())

    train_seq, val_seq = {}, {}
    for k, v in beh_seq.items():
        if isinstance(v, list) and len(v) == n:
            train_seq[k] = [v[i] for i in range(n) if i not in val_idx]
            val_seq[k]   = [v[i] for i in range(n) if i in val_idx]
        else:
            train_seq[k] = v
            val_seq[k] = v
    print(f"[kfold->manual val] carved {len(val_seq['image'])} / {n} tracks "
          f"into val (train left with {len(train_seq['image'])}).")
    return train_seq, val_seq


# ------------------------------------------------------------------
# Interactive run naming helpers
# ------------------------------------------------------------------
def interactive_pick_architecture(experiments_root, framework, default_architecture):
    """
    Lists existing architecture folders under experiments_root/framework
    with a numeric index next to each one, so you can pick by number
    instead of typing the full name. Pressing Enter keeps
    default_architecture (derived from the hyperparameters above, not a
    hardcoded literal). Typing a name not in the list creates/uses that
    new architecture name (unchanged old behavior, still supported).
    """
    framework_dir = os.path.join(experiments_root, framework)
    existing = []
    if os.path.isdir(framework_dir):
        existing = sorted(
            d for d in os.listdir(framework_dir)
            if os.path.isdir(os.path.join(framework_dir, d))
        )

    print(f"\n[architecture] existing architecture folders under {framework_dir}:")
    if existing:
        for i, a in enumerate(existing):
            marker = "  <-- default (matches current hyperparameters)" if a == default_architecture else ""
            print(f"  [{i}] {a}{marker}")
    else:
        print("  (none yet)")

    while True:
        raw = input(
            f"[architecture] enter a number from the list above, "
            f"or type a new architecture name "
            f"[default: '{default_architecture}']: "
        ).strip()

        if raw == "":
            return default_architecture

        # Numeric input -> pick from the existing list by index.
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(existing):
                return existing[idx]
            print(f"  ! out of range, pick 0-{len(existing) - 1} or type a new name.")
            continue

        # Otherwise treat it as a literal (possibly new) architecture name.
        if _is_valid_name(raw):
            return raw
        print("  ! invalid characters in name, try again.")


def interactive_pick_new_run_name(architecture_dir, hyperparam_tag):
    """
    Lists existing run folders under architecture_dir purely for visibility.
    The final run name is ALWAYS built around hyperparam_tag (derived from
    the actual hyperparameters used in this run), so the folder name can
    never silently disagree with the model that was actually trained.

    - Enter (no input)  -> run name == hyperparam_tag
    - typed prefix 'foo' -> run name == f"{foo}_{hyperparam_tag}"
      (the prefix is PREPENDED, never a replacement, so the hyperparameter
      tag is never lost).
    """
    existing = _list_existing_run_dirs(architecture_dir)
    print(f"\n[run_name] existing run folders under:\n  {architecture_dir}")
    if existing:
        for i, d in enumerate(existing):
            print(f"  [{i}] {os.path.basename(d)}")
    else:
        print("  (none yet)")

    while True:
        raw = input(
            f"[run_name] enter an OPTIONAL prefix for this NEW run "
            f"(prepended to the auto hyperparameter tag "
            f"'{hyperparam_tag}') "
            f"[default: no prefix -> '{hyperparam_tag}']: "
        ).strip()

        name = hyperparam_tag if raw == "" else f"{raw}_{hyperparam_tag}"

        if _is_valid_name(name):
            return name
        print("  ! invalid characters in name, try again.")


def interactive_pick_balance_mode():
    """
    Lists balancing strategies with a numeric index, same interaction
    pattern as interactive_pick_architecture()/interactive_pick_new_run_name()
    -- a list of (value, label) pairs enumerated for display, so
    adding another strategy later only means appending to
    BALANCE_MODE_OPTIONS, not editing branching logic.

    'none' skips both undersampling and augmentation entirely -- train/val
    keep the raw, naturally imbalanced class distribution. This is the
    right choice when the loss function itself already compensates for
    imbalance (weighted_bce, focal_loss), since stacking data-level
    balancing on TOP of loss-level weighting double-compensates and can
    distort the effective class ratio the model actually trains against.
    """
    BALANCE_MODE_OPTIONS = [
                            ('augmentation',
                            'custom method -- geometric + photometric + temporal (shift/speed/occlusion) + micro-jitter'),
                            ('undersample_base_article',
                            'exactly matches the reference code aras62/PIEPredict'),
                            ('none',
                            'no balancing -- raw imbalanced train/val (use with weighted_bce/focal_loss)'),
                            ]
    default_idx = 0  # index into BALANCE_MODE_OPTIONS used when Enter is pressed

    print("\n[balance_mode] existing balancing strategies:", flush=True)
    for i, (value, desc) in enumerate(BALANCE_MODE_OPTIONS):
        marker = "  <-- default" if i == default_idx else ""
        print(f"  [{i}] {value}  ({desc}){marker}", flush=True)

    while True:
        raw = input(
            f"[balance_mode] enter a number from the list above "
            f"[default: {default_idx} -> '{BALANCE_MODE_OPTIONS[default_idx][0]}']: "
        ).strip()

        if raw == "":
            return BALANCE_MODE_OPTIONS[default_idx][0]

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(BALANCE_MODE_OPTIONS):
                return BALANCE_MODE_OPTIONS[idx][0]
            print(f"  ! out of range, pick 0-{len(BALANCE_MODE_OPTIONS) - 1}.", flush=True)
            continue

        print("  ! please enter a number (not text) from the list above.", flush=True)

def interactive_pick_val_augment_mode():
    """
    Only asked when BALANCE_MODE == 'augmentation'. Controls how val gets
    balanced in that mode:
      - 'augment_val': old behavior -- val also gets synthetic
        geometric + photometric + temporal (shift/speed/occlusion) + micro-jitter'.
      - 'undersample_val': val is balanced via real-sample undersampling
        only (matches the reference paper's train_test.py) -- no
        synthetic data ever enters val.
    """
    VAL_AUGMENT_MODE_OPTIONS = [
                                ('undersample_val',
                                'train augmented, val undersampled (real samples only) -- matches reference paper'),
                                ('augment_val',
                                'train AND val both get synthetic augmented copies (old behavior)'),
                                ('natural_val',
                                'val left completely untouched (raw, natural imbalance) -- matches test distribution'),
                            ]
    default_idx = 0

    print("\n[val_augment_mode] when BALANCE_MODE='augmentation', how should val be balanced?", flush=True)
    for i, (value, desc) in enumerate(VAL_AUGMENT_MODE_OPTIONS):
        marker = "  <-- default" if i == default_idx else ""
        print(f"  [{i}] {value}  ({desc}){marker}", flush=True)

    while True:
        raw = input(
            f"[val_augment_mode] enter a number from the list above "
            f"[default: {default_idx} -> '{VAL_AUGMENT_MODE_OPTIONS[default_idx][0]}']: "
        ).strip()

        if raw == "":
            return VAL_AUGMENT_MODE_OPTIONS[default_idx][0]

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(VAL_AUGMENT_MODE_OPTIONS):
                return VAL_AUGMENT_MODE_OPTIONS[idx][0]
            print(f"  ! out of range, pick 0-{len(VAL_AUGMENT_MODE_OPTIONS) - 1}.", flush=True)
            continue

        print("  ! please enter a number (not text) from the list above.", flush=True)


def interactive_pick_monitor_metric():
    """
    Chooses what EarlyStopping/ModelCheckpoint/ReduceLROnPlateau monitor
    during training. 'val_auc' is computed via sklearn's roc_auc_score
    inside EpochMetricsTable.on_epoch_end() (pie_intent.py) and injected
    into Keras' logs dict every epoch, so it is always available
    regardless of which option is chosen here. Same numbered-list
    interaction pattern as interactive_pick_balance_mode().
    """
    MONITOR_OPTIONS = [
        ('val_auc', 'max', 'validation ROC-AUC (recommended -- robust to class imbalance)'),
        ('val_loss', 'min', 'validation loss (classic, sensitive to imbalance/calibration)'),
    ]
    default_idx = 0

    print("\n[monitor_metric] choose what EarlyStopping/Checkpoint/Plateau monitor:", flush=True)
    for i, (metric, mode, desc) in enumerate(MONITOR_OPTIONS):
        marker = "  <-- default" if i == default_idx else ""
        print(f"  [{i}] {metric} ({mode})  -- {desc}{marker}", flush=True)

    while True:
        raw = input(
            f"[monitor_metric] enter a number from the list above "
            f"[default: {default_idx} -> '{MONITOR_OPTIONS[default_idx][0]}']: "
        ).strip()

        if raw == "":
            metric, mode, _ = MONITOR_OPTIONS[default_idx]
            return metric, mode

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(MONITOR_OPTIONS):
                metric, mode, _ = MONITOR_OPTIONS[idx]
                return metric, mode
            print(f"  ! out of range, pick 0-{len(MONITOR_OPTIONS) - 1}.", flush=True)
            continue

        print("  ! please enter a number (not text) from the list above.", flush=True)



def interactive_pick_loss_choice():
    """
    Chooses the loss FAMILY. For 'weighted_bce', pos_weight is NOT asked
    here -- it is impossible to know the correct value at this point in
    the script, since it depends on the TRAIN split's actual class counts
    AFTER balancing/augmentation (if any) has been applied. Asking for it
    here risks a stale/guessed value and possible double-compensation
    with data-level balancing. The caller must resolve it later via
    train_helpers.compute_pos_weight_from_train(beh_seq_train) once
    beh_seq_train is final, then build the actual loss closure with
    weighted_binary_crossentropy(pos_weight=...).

    Returns one of:
      ('binary_crossentropy', 'binary_crossentropy')          -- ready to use
      (name, closure)                                          -- focal_loss, ready to use
      ('weighted_bce', None)                                   -- resolve later
    """
    print("\n[loss_function] choose the training loss function:", flush=True)
    print("  [0] binary_crossentropy  (standard BCE loss)  <-- default", flush=True)
    print("  [1] focal_loss           (focal loss for class imbalance)", flush=True)
    print("  [2] weighted_bce         (pos_weight computed AUTOMATICALLY "
          "from the train split's real class counts, after balancing/"
          "augmentation is applied)", flush=True)

    while True:
        raw = input("[loss_function] enter a number from the list above [default: 0]: ").strip()

        if raw == "" or raw == "0":
            return "binary_crossentropy", "binary_crossentropy"

        if raw == "1":
            gamma_raw = input("  [focal_loss] gamma (focusing param) [default: 1.0]: ").strip()
            alpha_raw = input("  [focal_loss] alpha (positive-class weight) [default: 0.25]: ").strip()
            gamma = float(gamma_raw) if gamma_raw else 1.0
            alpha = float(alpha_raw) if alpha_raw else 0.25
            name = f"focal_loss_g{gamma}_a{alpha}"
            return name, binary_focal_loss(gamma=gamma, alpha=alpha)

        if raw == "2":
            return "weighted_bce", None  # resolved later, once train data is final

        print("  ! please enter 0, 1, or 2.", flush=True)


def compute_pos_weight_from_train(beh_seq_train):
    """
    Computes pos_weight = neg_count / pos_count from the TRAIN split's
    actual class counts, AFTER any balancing/augmentation has already
    been applied to beh_seq_train (so this reflects exactly what the
    model will actually be trained on -- never val/test, and never the
    pre-balancing raw counts).

    pos_weight=1.0 (the previous hardcoded default) is mathematically a
    no-op in weighted_binary_crossentropy -- it does not compensate for
    any imbalance at all. This function exists so that never happens by
    accident again.
    """
    pos = sum(1 for lbl in beh_seq_train['intention_binary'] if lbl[0][0] == 1)
    neg = len(beh_seq_train['intention_binary']) - pos
    if pos == 0:
        raise ValueError("compute_pos_weight_from_train: train split has zero "
                          "positive samples -- cannot compute pos_weight.")
    pos_weight = neg / pos
    print(f"[loss] pos_weight computed from TRAIN split ONLY: "
          f"pos={pos} neg={neg} -> pos_weight={pos_weight:.4f}")
    return pos_weight


# ------------------------------------------------------------------
# Console logging helper
# ------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------
def compute_full_metrics(y_true, y_prob, save_path=None, threshold=0.5):
    """
    :param threshold: decision threshold used to convert raw probabilities
        into binary labels. Defaults to 0.5 (original behavior, unchanged).
        Whenever a calibrated threshold is available (e.g. loaded from
        calibration.json), pass it here so all metrics are computed at
        that threshold instead of the fixed 0.5 default.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    )

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    # Previously: y_pred = np.round(y_prob).astype(int)  -- always threshold=0.5
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    positive_rate_gt = float(y_true.mean())
    majority_acc = max(positive_rate_gt, 1 - positive_rate_gt)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')

    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    nll = log_loss(y_true, y_prob_clipped, labels=[0, 1])

    def expected_calibration_error(y_true, y_prob, n_bins=15):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            bin_conf = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += (mask.sum() / len(y_prob)) * abs(bin_conf - bin_acc)
        return float(ece)

    ece = expected_calibration_error(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    metrics = {
        'accuracy': float(accuracy), 'majority_acc': float(majority_acc),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc),
        'nll': float(nll), 'ece': float(ece), 'brier_score': brier_score,
        'confusion_matrix': cm, 'n_test_samples': int(n),
        'positive_rate_gt': positive_rate_gt,
        'threshold_used': float(threshold),   # NEW: record which threshold produced these metrics
    }
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as fid:
            json.dump(metrics, fid, indent=2)
    return metrics


def print_full_epoch_metrics_table(epoch_metrics_path):
    """
    Unlike `print_full_history_table` (which only includes the columns present in
   `history.pkl`—i.e., the raw Keras History output: 
   loss/accuracy/val_loss/val_accuracy/val_auc), this function reads directly
     from `epoch_metrics.json`—the same file that
       `EpochMetricsTable._save_history()` in `pie_intent.py` writes at each epoch,
         containing all metrics (precision/recall/f1/auc/nll/ece/...) for all three
           sets: train, val, and test.
    """
    if not os.path.exists(epoch_metrics_path):
        print(f"[warning] epoch metrics file not found at {epoch_metrics_path} -- nothing to print.")
        return

    with open(epoch_metrics_path, 'r', encoding='utf-8') as fid:
        records = json.load(fid)

    if not records:
        print("[warning] epoch_metrics.json is empty -- nothing to print.")
        return

    METRIC_NAMES = ['loss', 'accuracy', 'majority_acc', 'precision', 'recall',
                    'f1', 'roc_auc', 'pr_auc', 'nll', 'ece', 'brier_score']
    splits = ['train', 'val', 'test']
    col_width = 12

    header = "epoch".rjust(6)
    for split in splits:
        for m in METRIC_NAMES:
            header += f"{split[:2]}_{m}"[:col_width].rjust(col_width)

    print("\n" + "=" * len(header))
    print("FULL PER-EPOCH METRICS -- TRAIN / VAL / TEST (from epoch_metrics.json)")
    print("=" * len(header))
    print(header)

    for rec in records:
        row = str(rec['epoch']).rjust(6)
        for split in splits:
            for m in METRIC_NAMES:
                key = f'{split}_{m}'
                v = rec.get(key)
                row += f"{v:>{col_width}.4f}" if isinstance(v, (int, float)) else f"{'--':>{col_width}}"
        print(row)
    print("=" * len(header))

def print_full_history_table(history_dict):

    if not history_dict:
        print("[warning] history dict is empty -- nothing to print.")
        return

    metric_keys = ['loss', 'val_loss'] + [
        k for k in history_dict.keys()
        if k not in ('loss', 'val_loss')
    ]
    n_epochs_ran = len(history_dict['loss'])

    col_width = 14
    header = "epoch".rjust(6) + "".join(k.rjust(col_width) for k in metric_keys)
    print("\n" + "=" * len(header))
    print("FULL PER-EPOCH METRICS (from history.pkl -- Keras' own History callback)")
    print("=" * len(header))
    print(header)
    for e in range(n_epochs_ran):
        row = str(e + 1).rjust(6)
        for k in metric_keys:
            v = history_dict[k][e]
            if isinstance(v, (int, float)):
                row += f"{v:>{col_width}.4f}"
            else:
                row += f"{str(v):>{col_width}}"
        print(row)
    print("=" * len(header))

# ------------------------------------------------------------------
# Balance-mode preset resolution
# ------------------------------------------------------------------
BALANCE_MODE_DISPLAY = {
    'none': 'noBalance',
    'undersample_base_article': 'undersample',
    'augmentation': 'augmentation',
}

def resolve_balance_mode_preset(balance_mode, base_clip_type, base_clip_amount, base_lr,
                                batch_size=128, normalize_decoder_bbox=True):
    """Returns (clip_type, clip_amount, batch_size, normalize_decoder_bbox)."""
    if balance_mode == "undersample_base_article":
        clip_type, clip_amount = 'none', None
    else:
        clip_type, clip_amount = base_clip_type, base_clip_amount

    print(f"[config] preset={balance_mode}: lr={base_lr} clip_type={clip_type} "
          f"clip_amount={clip_amount} batch_size={batch_size} "
          f"normalize_decoder_bbox={normalize_decoder_bbox}")
    return clip_type, clip_amount, batch_size, normalize_decoder_bbox

    
# ------------------------------------------------------------------
# Balancing strategy application
# ------------------------------------------------------------------
def _print_class_counts(beh_seq, split_name, balance_mode_tag):
    pos = sum(1 for lbl in beh_seq['intention_binary'] if lbl[0][0] == 1)
    neg = len(beh_seq['intention_binary']) - pos
    print(f"[{balance_mode_tag}] {split_name} class counts: "
          f"positive={pos} negative={neg} (positive_rate={pos / (pos + neg):.4f})")
    return pos, neg




def apply_balancing_strategy(imdb, beh_seq_train, beh_seq_val, balance_mode,
                              augment_fn, val_augment_mode='undersample_val'):
    """
    Applies one of the three balancing strategies to train/val, exactly
    matching the previous inline logic in train_pie_intent_tensorflow_gpu.py.

    :param augment_fn: augment_pie_tracks from augmentation.py, passed in
        rather than imported here to avoid a circular/heavy import at
        module load time.
    :return: (beh_seq_train, beh_seq_val) -- possibly rebuilt in place
    """
    if balance_mode == "undersample_base_article":
        print("Balancing train/val via UNDERSAMPLING (base-article method)...")
        beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')
        beh_seq_val   = imdb.balance_samples_count(beh_seq_val,   label_type='intention_binary')

    elif balance_mode == "none":
        print("Skipping balancing entirely -- train/val kept at raw, "
              "naturally imbalanced class distribution.")
        _print_class_counts(beh_seq_train, 'train', 'balance_mode=none')
        _print_class_counts(beh_seq_val,   'val',   'balance_mode=none')

    else:  # augmentation
        print("Balancing train/val via AUGMENTATION (custom method)...")
        pos, neg = _print_class_counts(beh_seq_train, 'train', 'balance_mode=augmentation')
        minority_label = 0 if neg < pos else 1
        num_augments = max(1, -(-(max(pos, neg) - min(pos, neg)) // max(min(pos, neg), 1)))
        print(f"[augmentation] train: minority_label={minority_label} "
              f"(pos={pos} neg={neg}) num_augments={num_augments} -- "
              f"{_describe_aug_cfg(AUGMENTATION_CFG)} seed=42")
        
        beh_seq_train = augment_fn(beh_seq_train, num_augments=num_augments,
                                    augment_only_label=minority_label, seed=42,
                                    **AUGMENTATION_CFG)
        
        beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

        if val_augment_mode == 'augment_val':
            val_pos, val_neg = _print_class_counts(beh_seq_val, 'val', 'balance_mode=augmentation')
            val_minority_label = 0 if val_neg < val_pos else 1
            val_num_augments = max(1, -(-(max(val_pos, val_neg) - min(val_pos, val_neg)) // max(min(val_pos, val_neg), 1)))
            print(f"[augmentation] val: minority_label={val_minority_label} "
                  f"(pos={val_pos} neg={val_neg}) num_augments={val_num_augments} -- "
                  f"{_describe_aug_cfg(AUGMENTATION_CFG)} seed=43")
            beh_seq_val = augment_fn(beh_seq_val, num_augments=val_num_augments,
                                      augment_only_label=val_minority_label, seed=43,
                                      **AUGMENTATION_CFG)
            beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

        elif val_augment_mode == 'natural_val':
            _print_class_counts(beh_seq_val, 'val',
                'balance_mode=augmentation (val left natural/raw -- matches test distribution)')
            # beh_seq_val unchanged

        else:  # 'undersample_val'
            print("[augmentation] val: balancing via UNDERSAMPLING (real samples only "
                  "-- matches reference paper's train_test.py)")
            beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

    return beh_seq_train, beh_seq_val


# ------------------------------------------------------------------
# Run hyperparameter tag
# ------------------------------------------------------------------
def build_run_hyperparam_tag(REGULARIZER_VAL, clip_type, clip_amount, balance_mode, lr, lstm_dropout,
                              lstm_recurrent_dropout, loss_function_name, monitor_metric,
                              split_tag, weight_decay=0.0,
                              early_stop_min_delta=0.0005, early_stop_patience=10,
                              seed=None):
    """Builds the RUN_HYPERPARAM_TAG string used for both the folder name
    and the console-visible [config] RUN_NAME line."""
    balance_mode_tag = BALANCE_MODE_DISPLAY.get(balance_mode, balance_mode)
    clip_tag = f"{clip_type}{clip_amount}" if clip_type != 'none' else "noClip"
    wd_tag = f"_wd{weight_decay}" if weight_decay else ""
    seed_tag = f"_seed{seed}" if seed is not None else ""
    return (
        f"{split_tag}_{clip_tag}_{balance_mode_tag}_Reg{REGULARIZER_VAL}_LR{format_lr_tag(lr)}"
        f"_lsdr{lstm_dropout}_lsre{lstm_recurrent_dropout}"
        f"{wd_tag}"
        f"_esmd{early_stop_min_delta}_esp{early_stop_patience}"
        f"_{loss_function_name}_{monitor_metric}"
        f"{seed_tag}"
    )

def build_data_split_tag(data_split_type, data_split_opts):
    """
    Builds a short, filesystem-safe tag summarizing the interactively-chosen
    data_split_type (+ its sub-params), for inclusion in the run folder name.
    Every branch of interactive_pick_data_split() must be representable here.

      'default' -> 'defaultSplit'
      'random'  -> 'randomSplit_r0.5-0.4-0.1' (ratios joined with '-', no
                    spaces/commas -- both are invalid in Windows folder names)
      'kfold'   -> 'kfold5_f1' (num_folds=5, fold=1)
    """
    if data_split_type == 'default':
        return 'defaultSplit'

    if data_split_type == 'random':
        ratios = data_split_opts.get('random_params', {}).get('ratios', [0.5, 0.4, 0.1])
        ratios_tag = '-'.join(str(r) for r in ratios)
        return f'randomSplit_r{ratios_tag}'

    if data_split_type == 'kfold':
        kfold_params = data_split_opts.get('kfold_params', {})
        num_folds = kfold_params.get('num_folds', 5)
        fold = kfold_params.get('fold', 1)
        return f'kfold{num_folds}_f{fold}'

    return data_split_type 
# ------------------------------------------------------------------
# Weight decay (interactive)
# ------------------------------------------------------------------
def interactive_pick_weight_decay(default_amount=0.0):
    """
    Chooses the weight_decay value forwarded to pie_intent.py's
    optimizer_params, which passes it to RMSprop ONLY when non-zero
    (pie_intent.py, ~lines 590-594).

    IMPORTANT (this project's actual environment: TF 2.10.1 / Keras
    2.10.0 -- NOT Keras 3): tf.keras.optimizers.RMSprop in Keras 2.10.0
    only accepts {lr, decay, clipnorm, clipvalue, global_clipnorm} --
    there is no 'weight_decay' argument at all. Any non-zero value
    crashes at optimizer construction. Verified empirically on this
    system (set_HP diagnostic: "RMSprop(weight_decay=...) supported:
    False"). This function previously suggested a range (1e-5 - 1e-4)
    and referenced "Keras 3" in its docstring -- both were wrong for
    this environment and are fixed here.

    Real L2 regularization in this project comes from REGULARIZER_VAL
    on the model's kernel/recurrent/bias regularizers -- a separate,
    working mechanism, unaffected by this optimizer kwarg.

    :return: weight_decay: float -- always 0.0 in this environment.
    """
    print("\n[weight_decay] RMSprop weight_decay in TF 2.10.1 / Keras 2.10.0:", flush=True)
    print("  NOT SUPPORTED. Keras 2.10.0's RMSprop has no 'weight_decay' "
          "argument -- any non-zero value crashes at optimizer construction.", flush=True)
    print("  L2 regularization in this project comes from REGULARIZER_VAL "
          "(kernel/recurrent/bias regularizers), which is unaffected by this.", flush=True)

    raw = input(
        f"[weight_decay] enter a value [default: {default_amount}] "
        f"(non-zero will be rejected in this environment): "
    ).strip()
    weight_decay = float(raw) if raw else default_amount

    if weight_decay != 0.0:
        print(f"[weight_decay] REJECTED: {weight_decay} is non-zero and would crash "
              f"RMSprop in Keras 2.10.0 (no weight_decay support). Forcing 0.0.")
        weight_decay = 0.0

    print(f"[weight_decay] weight_decay = {weight_decay}")
    return weight_decay

# ------------------------------------------------------------------
# Pickle loading + threshold calibration + summary printing
# ------------------------------------------------------------------
def load_ped_intents_pkl(saved_files_path, split_name):
    """Loads ped_intents_{split_name}.pkl and returns (y_prob, y_true) as
    flat lists -- the same extraction repeated in evaluate_split_full_metrics,
    the calibration block, and recompute_with_calibrated_threshold."""
    import pickle as _pickle
    path = os.path.join(saved_files_path, f'ped_intents_{split_name}.pkl')
    with open(path, 'rb') as fid:
        ped_intents = _pickle.load(fid)
    y_prob = [np.asarray(r).ravel()[0] for r in ped_intents['results']]
    y_true = ped_intents['gt']
    return y_prob, y_true


def find_best_threshold_by_f1(val_prob, val_true, lo=0.05, hi=0.96, step=0.01):
    """
    Sweeps decision thresholds [lo, hi) and returns (best_threshold,
    best_f1) that maximizes F1 on the given (val_prob, val_true) --
    ALWAYS pass val-split predictions/labels here, never test, since
    picking a threshold on test would leak test information into the
    reported numbers.
    """
    from sklearn.metrics import f1_score
    val_prob = np.asarray(val_prob)
    val_true = np.asarray(val_true).astype(int).ravel()
    best_thresh, best_f1 = None, -1.0
    for thresh in np.arange(lo, hi, step):
        y_pred = (val_prob >= thresh).astype(int)
        score = f1_score(val_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh
    return float(best_thresh), float(best_f1)


def evaluate_split_full_metrics(model, data_split, data_opts, saved_files_path,
                                 paths, split_name, threshold=0.5):
    """
    Reloads the BEST checkpoint and runs model.test_chunk() once over the
    given split at overlap=1 (the paper's fixed evaluation protocol),
    then computes the full metrics dict at `threshold`.

    NOTE: overlap=1 here is independent of data_opts['seq_overlap_rate']
    used during training -- window count/composition differ from the
    per-epoch tables printed during training, so numbers will differ even
    for the identical checkpoint weights. See PHASE_FULL_EVAL explanation.
    """
    acc, f1 = model.test_chunk(
        data_split, data_opts, saved_files_path, visualize=False,
        results_suffix=f'_{split_name}', vis_num_correct=30, vis_num_incorrect=30,
    )
    print(f"[note] '{split_name}' below uses overlap=1 (fixed evaluation "
          f"protocol), NOT seq_overlap_rate={data_opts['seq_overlap_rate']} "
          f"used during training -- expect different sample counts/values "
          f"than the per-epoch table above.")
    print(f"[{split_name}] Paper metrics (threshold=0.5) -- accuracy: {acc:.4f}  F1: {f1:.4f}")

    y_prob, y_true = load_ped_intents_pkl(saved_files_path, split_name)
    split_metrics_path = paths['metrics_path'].replace('.json', f'_{split_name}.json')
    metrics = compute_full_metrics(y_true, y_prob, save_path=split_metrics_path, threshold=threshold)
    print(f"[{split_name}] full metrics (threshold={threshold:.2f}) saved to: {split_metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def recompute_split_metrics_at_threshold(saved_files_path, paths, split_name, threshold):
    """Re-thresholds already-saved predictions (no re-inference) at the
    given calibrated threshold and saves/returns the resulting metrics."""
    y_prob, y_true = load_ped_intents_pkl(saved_files_path, split_name)
    calibrated_path = paths['metrics_path'].replace('.json', f'_{split_name}_calibrated.json')
    metrics = compute_full_metrics(y_true, y_prob, save_path=calibrated_path, threshold=threshold)
    print(f"[{split_name}] CALIBRATED metrics (threshold={threshold:.2f}) saved to: {calibrated_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def print_summary_table(title, note, train_metrics, val_metrics, test_metrics):
    """Prints the two-line title/note header + aligned train/val/test
    table, shared between the threshold=0.5 SUMMARY and the calibrated
    SUMMARY blocks."""
    print("\n" + "=" * 70)
    print(title)
    if note:
        print(note)
    print("=" * 70)
    header = f"{'metric':>18s}  {'train':>10s}  {'val':>10s}  {'test':>10s}"
    print(header)
    for k in test_metrics:
        if isinstance(test_metrics[k], float):
            print(f"{k:>18s}  {train_metrics[k]:>10.4f}  {val_metrics[k]:>10.4f}  {test_metrics[k]:>10.4f}")

# ------------------------------------------------------------------
# Gradient clipping mode (interactive)
# ------------------------------------------------------------------
def interactive_pick_clip_mode(default_amount=0.5):
    """
    Chooses the gradient-clipping strategy passed to RMSprop.

      'clipvalue'      -- element-wise clipping (Keras' clipvalue). Each
          gradient COMPONENT is independently clamped to
          [-amount, +amount], regardless of the other components. Cheap,
          but can distort the gradient's DIRECTION (only the components
          that exceed the threshold are touched).

      'clipnorm'       -- per-VARIABLE norm clipping (Keras' clipnorm).
          For EACH weight tensor separately, if its L2 norm exceeds
          `amount`, the WHOLE tensor is rescaled by a single factor so
          its norm becomes exactly `amount`. Direction is preserved
          within each tensor, but there is no single norm computed
          across the whole model.

      'global_clipnorm' -- GLOBAL norm clipping (Keras' global_clipnorm),
          the direct equivalent of PyTorch's torch.nn.utils.
          clip_grad_norm_(). ONE L2 norm is computed across ALL
          trainable variables together; if it exceeds `amount`, every
          gradient tensor is rescaled by the SAME shared factor.
          Direction of the full gradient vector is preserved exactly.

      'none'           -- no gradient clipping at all.

    :return: (clip_type: str, clip_amount: float or None)
        clip_amount is None when clip_type == 'none'.
    """
    CLIP_OPTIONS = [
        ('clipvalue',
         "element-wise clipping -- each gradient component clamped "
         "independently (can distort gradient direction, cheapest)"),
        ('clipnorm',
         "per-variable norm clipping -- each weight tensor's OWN norm "
         "clamped separately (direction preserved per-tensor)"),
        ('global_clipnorm',
         "global norm clipping -- ONE norm across ALL trainable "
         "variables, same as torch.nn.utils.clip_grad_norm_ "
         "(direction of the full gradient preserved)"),
        ('none',
         "no gradient clipping"),
    ]
    default_idx = 0

    print("\n[clip_mode] choose the gradient-clipping strategy:", flush=True)
    for i, (value, desc) in enumerate(CLIP_OPTIONS):
        marker = "  <-- default" if i == default_idx else ""
        print(f"  [{i}] {value}  ({desc}){marker}", flush=True)

    while True:
        raw = input(
            f"[clip_mode] enter a number from the list above "
            f"[default: {default_idx} -> '{CLIP_OPTIONS[default_idx][0]}']: "
        ).strip()
        if raw == "":
            idx = default_idx
        elif raw.isdigit() and 0 <= int(raw) < len(CLIP_OPTIONS):
            idx = int(raw)
        else:
            print(f"  ! please enter a number 0-{len(CLIP_OPTIONS) - 1}.", flush=True)
            continue
        break

    clip_type = CLIP_OPTIONS[idx][0]
    if clip_type == 'none':
        print("[clip_mode] gradient clipping disabled.")
        return 'none', None

    amt_raw = input(
        f"  [{clip_type}] clip amount (threshold/max_norm) "
        f"[default: {default_amount}]: "
    ).strip()
    clip_amount = float(amt_raw) if amt_raw else default_amount

    print(f"[clip_mode] {clip_type} = {clip_amount}")
    return clip_type, clip_amount

def build_calibration_dict(threshold, metric, val_true, val_prob, pos_weight_used=None):
    """
    Builds the calibration.json payload with ONE canonical schema, shared by
    BOTH producers of this file:
      - train_pie_intent_tensorflow_gpu.py (inline, right after training)
      - select_calibration_threshold.py (standalone, re-run on a finished run)
    Previously each built its own dict with different key names
    (val_f1_at_threshold vs val_score_at_threshold, presence/absence of
    val_metrics_at_threshold / pos_weight_used) -- any downstream reader
    of calibration.json had to guess which schema it would get.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    val_true = np.asarray(val_true).astype(int).ravel()
    val_prob = np.asarray(val_prob).astype(float).ravel()

    def _metrics_at(thresh):
        y_pred = (val_prob >= thresh).astype(int)
        return {
            'threshold': float(thresh),
            'accuracy': float(accuracy_score(val_true, y_pred)),
            'f1': float(f1_score(val_true, y_pred, zero_division=0)),
            'precision': float(precision_score(val_true, y_pred, zero_division=0)),
            'recall': float(recall_score(val_true, y_pred, zero_division=0)),
        }

    metrics_at_thresh = _metrics_at(threshold)
    val_score = metrics_at_thresh.get(metric, metrics_at_thresh['f1'])

    return {
        'threshold': float(threshold),
        'selected_on': 'val',
        'metric_optimized': metric,
        'val_score_at_threshold': val_score,
        'val_metrics_at_threshold': metrics_at_thresh,
        'val_metrics_at_default_0.5': _metrics_at(0.5),
        'pos_weight_used': pos_weight_used,
        'note': 'chosen using val labels only; safe to apply as-is to test',
    }

# Maps the MONITOR_METRIC value used by train()/interactive_pick_monitor_metric()
# to the actual key name stored in epoch_metrics.json by EpochMetricsTable.
_MONITOR_METRIC_TO_JSON_KEY = {
    'val_auc': 'val_roc_auc',
    'val_loss': 'val_loss',
}

# ------------------------------------------------------------------
# Ensemble / k-fold-CV reporting
# ------------------------------------------------------------------
def _flatten_pid(p):
    """Fully unwraps PIE's nested ped_id structure ([[pid], [pid], ...])
    down to a plain scalar id."""
    while isinstance(p, (list, tuple)):
        p = p[0]
    return p


def load_ped_intents_full(run_dir, split_name='test'):
    """Like load_ped_intents_pkl, but also returns ped_id (flattened) --
    needed to detect whether several runs share the same test set."""
    import pickle as _pickle
    path = os.path.join(run_dir, f'ped_intents_{split_name}.pkl')
    with open(path, 'rb') as fid:
        ped_intents = _pickle.load(fid)
    y_prob = [np.asarray(r).ravel()[0] for r in ped_intents['results']]
    y_true = ped_intents['gt']
    ped_id = [_flatten_pid(p) for p in ped_intents['ped_id']]
    return y_prob, y_true, ped_id


ENSEMBLE_METRIC_ORDER = ['accuracy', 'majority_acc', 'precision', 'recall', 'f1',
                          'roc_auc', 'pr_auc', 'nll', 'ece', 'brier_score']


def print_metrics_table(rows, metric_order=ENSEMBLE_METRIC_ORDER, label_width=42, col_width=11):
    """rows: list of (label, metrics_dict) -- metrics_dict has the same
    keys compute_full_metrics() returns."""
    header = "".ljust(label_width) + "".join(k[:col_width].rjust(col_width) for k in metric_order)
    print(header)
    print("-" * len(header))
    for label, metrics in rows:
        row = label[:label_width - 1].ljust(label_width)
        row += "".join(f"{metrics[k]:>{col_width}.4f}" for k in metric_order)
        print(row)


def evaluate_runs_ensemble_or_cv(fold_dirs, split_name='test'):
    """
    Loads test-set predictions for every run in fold_dirs and either:
    - averages them into a genuine ensemble and reports full metrics
      for each single run, the ensemble, and the delta vs. the best
      single run -- IF every run shares the same test pedestrians
      (true for data_split_type='default', and for 'random' as long as
      random_samples.pkl was never regenerated between the selected
      runs); or
    - reports the standard k-fold cross-validation summary (per-fold
      metrics + mean/std) instead -- IF the runs do NOT share a test
      set (true for data_split_type='kfold', where PIE rotates test
      pedestrians per fold by construction, or for 'random' if
      regen_data=True was used partway through).

    Runs missing ped_intents_{split_name}.pkl (e.g. training crashed
    before PHASE_FULL_EVAL) are skipped with a warning, not fatal.

    :return: dict with 'same_test_set', 'fold_dirs' (the ones actually
        used), and either 'single_metrics'/'ensemble_metrics' or
        'fold_metrics'/'mean_metrics'/'std_metrics'.
    """
    probs, trues, ped_ids, valid_fold_dirs = [], [], [], []
    for run_dir in fold_dirs:
        pkl_path = os.path.join(run_dir, f'ped_intents_{split_name}.pkl')
        if not os.path.exists(pkl_path):
            print(f"[skip] {os.path.basename(run_dir)}: no ped_intents_{split_name}.pkl -- "
                  f"this run likely never reached PHASE_FULL_EVAL. Check history.pkl / "
                  f"epoch_metrics.json in that folder to confirm.")
            continue
        p, t, pid = load_ped_intents_full(run_dir, split_name)
        probs.append(p)
        trues.append(t)
        ped_ids.append(set(pid))
        valid_fold_dirs.append(run_dir)

    if len(valid_fold_dirs) < 2:
        raise RuntimeError(
            f"Only {len(valid_fold_dirs)} run(s) have usable predictions -- "
            "need at least 2 to compare/ensemble. See [skip] warnings above.")

    fold_dirs = valid_fold_dirs
    same_test_set = all(ped_ids[0] == s for s in ped_ids[1:])

    print(f"\n[check] {len(fold_dirs)} runs, test-set sizes: {[len(p) for p in probs]}")
    print(f"[check] all runs share identical test pedestrians: {same_test_set}")
    if not same_test_set:
        overlap_01 = len(ped_ids[0] & ped_ids[1]) if len(ped_ids) > 1 else None
        print(f"[check] overlap between run 0 and run 1's test ped_ids: {overlap_01} "
              f"(near 0 confirms disjoint k-fold test folds, not a shared holdout)")
    print()

    result = {'same_test_set': same_test_set, 'fold_dirs': fold_dirs}

    if same_test_set:
        test_true = trues[0]
        ensemble_prob = np.mean(probs, axis=0)
        single_metrics = [compute_full_metrics(test_true, p, threshold=0.5) for p in probs]
        ensemble_metrics = compute_full_metrics(test_true, ensemble_prob, threshold=0.5)

        print("Runs share a common test set -- full metrics per run, then the ensemble:\n")
        rows = list(zip([os.path.basename(d) for d in fold_dirs], single_metrics))
        rows.append((f"ENSEMBLE ({len(fold_dirs)} runs)", ensemble_metrics))
        print_metrics_table(rows)

        best_idx = max(range(len(single_metrics)), key=lambda i: single_metrics[i]['roc_auc'])
        best_single = single_metrics[best_idx]
        print(f"\nChange vs. best single run (by roc_auc: {os.path.basename(fold_dirs[best_idx])}):\n")
        for k in ENSEMBLE_METRIC_ORDER:
            delta = ensemble_metrics[k] - best_single[k]
            sign = '+' if delta >= 0 else ''
            print(f"  {k:<14} best_single={best_single[k]:>8.4f}   ensemble={ensemble_metrics[k]:>8.4f}   delta={sign}{delta:.4f}")

        result['single_metrics'] = single_metrics
        result['ensemble_metrics'] = ensemble_metrics

    else:
        print("Runs do NOT share a common test set -- averaging probabilities across")
        print("them is not meaningful. Reporting full per-fold metrics instead.\n")

        fold_metrics = [compute_full_metrics(t, p, threshold=0.5) for t, p in zip(trues, probs)]
        rows = list(zip([os.path.basename(d) for d in fold_dirs], fold_metrics))
        print_metrics_table(rows)

        mean_row = {k: float(np.mean([m[k] for m in fold_metrics])) for k in ENSEMBLE_METRIC_ORDER}
        std_row = {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1
           else 0.0 for k in ENSEMBLE_METRIC_ORDER}
        print()
        print_metrics_table([("MEAN across folds", mean_row), ("STD across folds", std_row)])

        print("\nFor a genuine prediction-averaging ensemble, use data_split_type='default'")
        print("(or 'random' without regenerating the split) for every member, varying")
        print("seed/hyperparameters instead.")

        result['fold_metrics'] = fold_metrics
        result['mean_metrics'] = mean_row
        result['std_metrics'] = std_row

    return result

def print_best_epoch_summary(epoch_metrics_path, monitor_metric, monitor_mode, top_n=5):
    """Prints the single epoch that maximizes/minimizes monitor_metric
    (whichever MONITOR_METRIC/MONITOR_MODE were actually used for this
    run's EarlyStopping/ModelCheckpoint), plus the top_n epochs for
    context -- so you don't need to scroll through hundreds of
    per-epoch rows to find the run that matters.

    monitor_metric: the value passed to model.train(monitor_metric=...),
                     e.g. 'val_auc' or 'val_loss'.
    monitor_mode:   the matching mode, 'max' for val_auc, 'min' for val_loss.
    """
    monitor_key = _MONITOR_METRIC_TO_JSON_KEY.get(monitor_metric, monitor_metric)
    if monitor_mode not in ('max', 'min'):
        raise ValueError(f"monitor_mode must be 'max' or 'min', got {monitor_mode!r}")

    with open(epoch_metrics_path, 'r', encoding='utf-8') as f:
        epoch_hist = json.load(f)

    if not epoch_hist:
        print(f"[best epoch] {epoch_metrics_path} is empty.")
        return

    scored = [r for r in epoch_hist if r.get(monitor_key) is not None]
    if not scored:
        available = sorted(k for k in epoch_hist[0].keys() if k.startswith('val_'))
        print(f"[best epoch] no epoch has a value for '{monitor_key}'. "
              f"Available val_* keys: {available}")
        return

    reverse = (monitor_mode == 'max')
    ranked = sorted(scored, key=lambda r: r[monitor_key], reverse=reverse)
    best = ranked[0]

    def _fmt_row(r):
        parts = [f"epoch={r['epoch']}", f"{monitor_key}={r[monitor_key]:.4f}"]
        for split in ('train', 'val', 'test'):
            acc = r.get(f'{split}_accuracy')
            auc = r.get(f'{split}_roc_auc')
            f1 = r.get(f'{split}_f1')
            if acc is not None:
                parts.append(f"{split}[acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}]")
        return "  ".join(parts)

    print(f"\n[best epoch] ranked by '{monitor_key}' (mode={monitor_mode}) -- this IS "
          f"the epoch whose weights were saved to model.h5 by ModelCheckpoint:")
    print("  " + _fmt_row(best))

    print(f"\n[best epoch] top {min(top_n, len(ranked))} epochs by '{monitor_key}':")
    for r in ranked[:top_n]:
        print("  " + _fmt_row(r))

    # Best val_accuracy can land on a DIFFERENT epoch than the monitored
    # metric -- shown for context only; the checkpoint on disk is
    # selected by monitor_key, not accuracy.
    acc_scored = [r for r in epoch_hist if r.get('val_accuracy') is not None]
    if acc_scored:
        best_acc_epoch = max(acc_scored, key=lambda r: r['val_accuracy'])
        if best_acc_epoch['epoch'] != best['epoch']:
            print(f"\n[best epoch] NOTE: best val_accuracy occurred at a DIFFERENT "
                  f"epoch than the saved checkpoint (checkpoint is selected by "
                  f"'{monitor_key}', mode={monitor_mode}, not accuracy):")
            print("  " + _fmt_row(best_acc_epoch))


def interactive_pick_lstm_recurrent_activation(default='sigmoid'):
    """Asks which recurrent_activation the decoder LSTM should use.

    'sigmoid'      -> Keras>=2.3 default (what the current code trains with).
    'hard_sigmoid' -> what the official model.h5 uses (old Keras default);
                      disables the cuDNN kernel, so training is slower.
    """
    options = {'1': 'sigmoid', '2': 'hard_sigmoid'}
    print("\nDecoder LSTM recurrent_activation:")
    print("  1) sigmoid       (current code)")
    print("  2) hard_sigmoid  (official model.h5; slower, no cuDNN)")
    while True:
        raw = input(f"Choose [1/2] (Enter = {default}): ").strip()
        if raw == '':
            return default
        if raw in options:
            return options[raw]
        print(f"  invalid choice: {raw!r}")