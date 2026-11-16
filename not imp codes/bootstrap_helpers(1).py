"""
bootstrap_helpers.py

Notebook entry for bootstrap_compare.py.
Paired cluster bootstrap over pedestrians for runs scored on the same
overlap=1 test windows (ped_intents_test.pkl, else ped_intents.pkl).

Set os.environ['PIE_PATH'] before importing this module if the dataset
is not at the default path below.

The pasted source ended mid-sentence ("until it"). That sentence is closed
here, and the printed report is written under evaluation_comparisons.
No CSV writer was in the pasted text, so none was added.
"""

import os
import glob
import pickle
import time

import numpy as np

PIE_PATH = os.environ.get('PIE_PATH', r'D:\PIE codes\PIE_dataset')
EXPERIMENTS_ROOT = os.path.join(PIE_PATH, 'experiments')
FRAMEWORK = 'tensorflow'
OFFICIAL_MODEL_DIR = r'D:\PIE codes\PIE\runs\downloaded'
OUTPUT_ROOT = os.path.join(PIE_PATH, 'evaluation_comparisons')

PRED_CANDIDATES = ['ped_intents_test.pkl', 'ped_intents.pkl']  # preferred first; not *_ov05
THRESHOLD = 0.5          # same rule as compute_full_metrics: y_prob >= threshold
N_BOOT = 2000
SEED = 0
CI_PCT = (2.5, 97.5)
MIN_OVERLAP = 0.99       # fraction of reference windows that every run must share
EPS = 1e-7               # same clipping as compute_full_metrics (NLL)

HIGHER_BETTER = {'auc': True, 'acc': True, 'f1': True, 'nll': False,
                 'acc_margin': True, 'f1_margin': True, 'nll_margin': False}

_CONFIGURABLE = {
    'OFFICIAL_MODEL_DIR', 'THRESHOLD', 'N_BOOT', 'SEED',
    'CI_PCT', 'MIN_OVERLAP', 'PRED_CANDIDATES', 'FRAMEWORK',
}
_LINES = []


def configure(**kwargs):
    """Override module constants from the notebook before run_bootstrap()."""
    unknown = sorted(set(kwargs) - _CONFIGURABLE)
    if unknown:
        raise KeyError(f"not configurable: {unknown}. Allowed: {sorted(_CONFIGURABLE)}")
    globals().update(kwargs)


def emit(s=''):
    print(s)
    _LINES.append(s)


def _flatten_pid(p):
    while isinstance(p, (list, tuple, np.ndarray)):
        p = p[0]
    return p


def find_pred_file(run_dir):
    for name in PRED_CANDIDATES:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def load_run_predictions(run_dir):
    path = find_pred_file(run_dir)
    if path is None:
        raise FileNotFoundError(f"no {PRED_CANDIDATES} in {run_dir}")
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
    for key in ('results', 'gt', 'ped_id'):
        if key not in data:
            raise KeyError(f"{path} has no key '{key}' (keys: {list(data)})")
    prob = np.array([np.asarray(r).ravel()[0] for r in data['results']], dtype=float)
    y = np.asarray(data['gt']).astype(int).ravel()
    pid = np.array([str(_flatten_pid(p)) for p in data['ped_id']])
    if not (len(prob) == len(y) == len(pid)):
        raise ValueError(
            f"{path}: lengths differ -- probs={len(prob)} gt={len(y)} ped_id={len(pid)}. "
            f"ped_id may not be per-window; inspect the pkl before trusting this script.")
    return prob, y, pid, path


def window_keys(pid):
    seen, keys = {}, []
    for p in pid:
        k = seen.get(p, 0)
        seen[p] = k + 1
        keys.append((p, k))
    return keys


def drop_incompatible(runs):
    """Keeps runs[0] (reference) and every run scored on (almost) the same windows with the same labels.
    Others are skipped with a printed reason, so 'compare with all' never aborts on one bad run."""
    ref_keys = window_keys(runs[0]['pid'])
    ref_label = dict(zip(ref_keys, runs[0]['y']))
    kept = [runs[0]]
    for run in runs[1:]:
        shared = mismatch = 0
        for key, yy in zip(window_keys(run['pid']), run['y']):
            if key in ref_label:
                shared += 1
                mismatch += int(ref_label[key] != yy)
        frac = shared / len(ref_label)
        if frac < MIN_OVERLAP:
            print(f"[skip] {run['tag']}: shares only {frac:.1%} of the reference windows "
                  f"(need >= {MIN_OVERLAP:.0%}) -- other split/fold or different test set.")
        elif mismatch:
            print(f"[skip] {run['tag']}: {mismatch} labels differ from the reference on shared windows.")
        else:
            kept.append(run)
    return kept


def align_runs(runs):
    """Returns (list of prob arrays, y, pid, kept_fraction) on the windows shared by all runs."""
    ref = runs[0]
    if all(len(r['pid']) == len(ref['pid']) and np.array_equal(r['pid'], ref['pid'])
           and np.array_equal(r['y'], ref['y']) for r in runs[1:]):
        return [r['prob'] for r in runs], ref['y'], ref['pid'], 1.0

    keyed = [window_keys(r['pid']) for r in runs]
    common = set(keyed[0])
    for keys in keyed[1:]:
        common &= set(keys)
    order = [key for key in keyed[0] if key in common]
    kept = len(order) / len(keyed[0])
    if kept < MIN_OVERLAP:
        raise RuntimeError(
            f"the kept runs share only {kept:.1%} of the reference windows (need >= {MIN_OVERLAP:.0%}). "
            f"Each run overlaps the reference enough, but their missing windows differ. "
            f"Pick a smaller, more homogeneous set of runs.")
    probs, ys = [], []
    for run, keys in zip(runs, keyed):
        pos = {key: i for i, key in enumerate(keys)}
        idx = np.array([pos[key] for key in order])
        probs.append(run['prob'][idx])
        ys.append(run['y'][idx])
    if not all(np.array_equal(ys[0], y) for y in ys[1:]):
        raise RuntimeError("labels differ on the shared windows -- runs are not comparable.")
    return probs, ys[0], np.array([key[0] for key in order]), kept


class RunStats:
    def __init__(self, prob, y):
        self.prob = prob
        self.yf = y.astype(float)
        self.pred = (prob >= THRESHOLD).astype(float)
        _, self.inv = np.unique(prob, return_inverse=True)   # tie groups for AUC
        self.ng = int(self.inv.max()) + 1
        pc = np.clip(prob, EPS, 1 - EPS)
        self.ll = -(self.yf * np.log(pc) + (1 - self.yf) * np.log(1 - pc))

    def metrics(self, w):
        total = w.sum()
        wy = w * self.yf
        n_pos = wy.sum()
        n_neg = total - n_pos
        pos_g = np.bincount(self.inv, weights=wy, minlength=self.ng)
        neg_g = np.bincount(self.inv, weights=w - wy, minlength=self.ng)
        cum_neg = np.cumsum(neg_g) - neg_g
        auc = ((pos_g * (cum_neg + 0.5 * neg_g)).sum() / (n_pos * n_neg)
               if n_pos > 0 and n_neg > 0 else np.nan)

        tp = (wy * self.pred).sum()
        fp = ((w - wy) * self.pred).sum()
        fn = (wy * (1 - self.pred)).sum()
        tn = total - tp - fp - fn
        acc = (tp + tn) / total
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        nll = (w * self.ll).sum() / total

        rate = n_pos / total
        base_acc = max(rate, 1 - rate)
        base_f1 = 2 * rate / (1 + rate) if rate > 0 else 0.0
        base_nll = (-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))
                    if 0 < rate < 1 else 0.0)
        return {'auc': auc, 'acc': acc, 'f1': f1, 'nll': nll,
                'acc_margin': acc - base_acc, 'f1_margin': f1 - base_f1,
                'nll_margin': nll - base_nll}


def parse_index_list(raw, max_index):
    out = []
    for tok in raw.replace(',', ' ').split():
        if '-' in tok:
            lo, hi = tok.split('-', 1)
            items = range(int(lo), int(hi) + 1)
        else:
            items = [int(tok)]
        for i in items:
            if i < 0 or i > max_index:
                raise ValueError(f"index {i} out of range 0-{max_index}")
            if i not in out:
                out.append(i)
    if not out:
        raise ValueError("no indices given")
    return out


def gather_candidates():
    items = []
    if os.path.isdir(OFFICIAL_MODEL_DIR) and find_pred_file(OFFICIAL_MODEL_DIR):
        items.append((OFFICIAL_MODEL_DIR, 'official'))
    framework_dir = os.path.join(EXPERIMENTS_ROOT, FRAMEWORK)
    if os.path.isdir(framework_dir):
        for arch in sorted(os.listdir(framework_dir)):
            arch_dir = os.path.join(framework_dir, arch)
            if not os.path.isdir(arch_dir):
                continue
            dirs = [d for d in glob.glob(os.path.join(arch_dir, '*'))
                    if os.path.isdir(d) and find_pred_file(d)]
            dirs.sort(key=os.path.getctime, reverse=True)
            items += [(d, f"{arch}/{os.path.basename(d)}") for d in dirs]
    return items


def interactive_pick_runs():
    items = gather_candidates()
    if len(items) < 2:
        raise FileNotFoundError("need at least 2 runs with saved predictions.")
    n = len(items)
    print("\n[runs] runs with saved predictions (ped_intents_test.pkl / ped_intents.pkl):")
    for i, (path, tag) in enumerate(items):
        ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getctime(path)))
        print(f"  [{i}] {tag}   (created {ts})")

    while True:
        raw = input(f"\n[reference] index of the REFERENCE run (e.g. the official model) "
                    f"[0-{n - 1}, default: 0]: ").strip()
        try:
            ref = 0 if raw == '' else int(raw)
        except ValueError:
            print("  ! please enter a number.")
            continue
        if 0 <= ref < n:
            break
        print(f"  ! out of range, pick 0-{n - 1}.")

    while True:
        raw = input("[compare] 'all' = every other run (runs scored on a different split/fold are skipped "
                    "automatically), or indices like '3 5-8' [default: all]: ").strip().lower()
        if raw in ('', 'a', 'all'):
            idxs = [i for i in range(n) if i != ref]
            break
        try:
            idxs = [i for i in parse_index_list(raw, n - 1) if i != ref]
        except ValueError as exc:
            print(f"  ! {exc}")
            continue
        if not idxs:
            print("  ! pick at least one run other than the reference.")
            continue
        break
    return [items[ref]] + [items[i] for i in idxs]


def cell(pt, lo, hi):
    return f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def verdict(metric, lo, hi):
    higher_better = HIGHER_BETTER[metric]
    if (higher_better and lo > 0) or ((not higher_better) and hi < 0):
        return 'BETTER'
    if (higher_better and hi < 0) or ((not higher_better) and lo > 0):
        return 'WORSE'
    return 'inconclusive'


def print_rows(title, header, rows):
    emit(f"\n{title}")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    emit('  ' + ' | '.join(str(h).ljust(w) for h, w in zip(header, widths)))
    emit('  ' + '-+-'.join('-' * w for w in widths))
    for row in rows:
        emit('  ' + ' | '.join(str(c).ljust(w) for c, w in zip(row, widths)))


def _save_report():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, f"bootstrap_{time.strftime('%d%b%Y-%Hh%Mm%Ss')}.txt")
    with open(path, 'w', encoding='utf-8') as fid:
        fid.write('\n'.join(_LINES) + '\n')
    print(f"[bootstrap] report -> {path}")
    return path


def run_bootstrap(preselected=None):
    """Interactive unless preselected=[(run_dir, tag), ...] with the reference first."""
    _LINES.clear()
    selection = preselected if preselected else interactive_pick_runs()
    runs = []
    for pos, (run_dir, tag) in enumerate(selection):
        try:
            prob, y, pid, path = load_run_predictions(run_dir)
        except Exception as exc:
            if pos == 0:
                raise
            print(f"[skip] {tag}: {type(exc).__name__}: {exc}")
            continue
        runs.append({'dir': run_dir, 'tag': tag, 'prob': prob, 'y': y, 'pid': pid, 'path': path})
        print(f"[load] {tag}: {len(y)} windows from {os.path.basename(path)}")

    runs = drop_incompatible(runs)
    if len(runs) < 2:
        raise RuntimeError("no run left to compare with the reference after skipping incompatible ones.")

    probs, y, pid, kept = align_runs(runs)
    uniq_ped, ped_idx = np.unique(pid, return_inverse=True)
    n_ped, n_win = len(uniq_ped), len(y)
    stats = [RunStats(p, y) for p in probs]

    emit("=" * 100)
    emit("PAIRED CLUSTER BOOTSTRAP OVER PEDESTRIANS")
    emit("=" * 100)
    emit(f"{len(runs)} runs ({len(runs) - 1} compared with the reference), reference = [0] {runs[0]['tag']}")
    emit(f"shared test windows: {n_win}  ({kept:.1%} of the reference's)   pedestrians (clusters): {n_ped}   "
         f"pos_rate={y.mean():.4f}   threshold={THRESHOLD}   N_BOOT={N_BOOT}   seed={SEED}")
    for i, run in enumerate(runs):
        emit(f"  [{i}] {run['tag']}   ({os.path.basename(run['path'])})")

    ones = np.ones(n_win)
    point = [s.metrics(ones) for s in stats]
    try:
        from sklearn.metrics import roc_auc_score
        worst = max(abs(pt['auc'] - roc_auc_score(y, s.prob)) for pt, s in zip(point, stats))
        emit(f"[check] max |AUC(bootstrap code) - sklearn AUC| on the full data = {worst:.2e}"
             + ("" if worst < 1e-9 else "   <-- WARNING: mismatch, do not trust the AUC rows"))
    except ImportError:
        emit("[check] sklearn not available -- AUC cross-check skipped.")

    rng = np.random.default_rng(SEED)
    metric_names = list(point[0].keys())
    boot = {name: np.empty((len(runs), N_BOOT)) for name in metric_names}
    t0 = time.time()
    for b in range(N_BOOT):
        counts = np.bincount(rng.integers(0, n_ped, n_ped), minlength=n_ped)
        weights = counts[ped_idx].astype(float)
        for k, s in enumerate(stats):
            measured = s.metrics(weights)
            for name in metric_names:
                boot[name][k, b] = measured[name]
        if (b + 1) % max(1, N_BOOT // 10) == 0:
            print(f"  [bootstrap] {b + 1}/{N_BOOT}  ({time.time() - t0:.0f}s)", flush=True)

    def interval(arr):
        lo, hi = np.nanpercentile(arr, CI_PCT)
        return float(lo), float(hi)

    rows = []
    for k, _run in enumerate(runs):
        row = [f"[{k}]"]
        for name in ('auc', 'acc', 'f1', 'nll'):
            lo, hi = interval(boot[name][k])
            row.append(f"{point[k][name]:.4f} [{lo:.4f}, {hi:.4f}]")
        rows.append(row)
    print_rows("A) METRICS @ threshold with 95% CI (pedestrian-level bootstrap)",
               ['run', 'AUC', 'accuracy', 'F1', 'NLL'], rows)

    rows = []
    for k, _run in enumerate(runs):
        row = [f"[{k}]"]
        for name in ('acc_margin', 'f1_margin', 'nll_margin'):
            lo, hi = interval(boot[name][k])
            if HIGHER_BETTER[name]:
                p_better = float(np.mean(boot[name][k] > 0))
            else:
                p_better = float(np.mean(boot[name][k] < 0))
            row.append(f"{cell(point[k][name], lo, hi)}  P={p_better:.2f}  {verdict(name, lo, hi)}")
        rows.append(row)
    print_rows("B) MARGIN vs TRIVIAL PREDICTOR (acc: majority, F1: all-positive, NLL: constant prior; "
               "NLL margin < 0 is better)",
               ['run', 'acc - majority_acc', 'F1 - allpos_F1', 'NLL - const_NLL'], rows)

    order = sorted(range(1, len(runs)), key=lambda k: point[k]['auc'] - point[0]['auc'], reverse=True)
    rows = []
    for k in order:
        row = [f"[{k}] vs [0]"]
        for name in ('auc', 'acc', 'f1', 'nll'):
            delta = boot[name][k] - boot[name][0]
            lo, hi = interval(delta)
            pt = point[k][name] - point[0][name]
            if HIGHER_BETTER[name]:
                p_better = float(np.mean(delta > 0))
            else:
                p_better = float(np.mean(delta < 0))
            row.append(f"{cell(pt, lo, hi)}  P={p_better:.2f}  {verdict(name, lo, hi)}")
        rows.append(row)
    print_rows("C) PAIRED DELTA vs REFERENCE [0], sorted by dAUC (NLL delta < 0 is better). "
               "P = fraction of bootstrap samples in which the run is better.",
               ['comparison', 'dAUC', 'dAccuracy', 'dF1', 'dNLL'], rows)

    emit("\nREAD THIS: a verdict BETTER/WORSE means the 95% CI of the difference excludes 0 for THIS test set's "
         "pedestrians. It says nothing about training-seed variance (~0.010 AUC std in the project notes) and it "
         "is optimistic if the run was picked as the best of many after looking at test. 'inconclusive' = the "
         "difference is within pedestrian-sampling noise.")
    n_cmp = len(runs) - 1
    if n_cmp > 1:
        emit(f"MULTIPLE COMPARISONS: {n_cmp} runs were compared with the reference at 95% each, so about "
             f"{0.05 * n_cmp:.1f} of them are expected to look BETTER/WORSE by chance alone. A lone BETTER "
             f"among many is not a finding until it shows up again on another training seed.")

    report_path = _save_report()
    return {'report_path': report_path, 'lines': list(_LINES), 'n_runs': len(runs)}


if __name__ == '__main__':
    run_bootstrap()
