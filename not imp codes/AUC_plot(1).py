import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

PIE_PATH = os.environ.get("PIE_PATH", r"D:\PIE codes\PIE_dataset")
EXPERIMENTS_ROOT = os.path.join(PIE_PATH, "experiments")
DB_PATH = os.path.join(PIE_PATH, "data_cache", "pie_database.pkl")
AUG_MARKER = "__aug"


# ---------------------------------------------------------------- menus
def _list_dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def pick_one(title, items, default=0):
    print(f"\n{title}")
    for i, it in enumerate(items):
        print(f"  [{i}] {it}{'  <-- default' if i == default else ''}")
    while True:
        raw = input(f"choose [default: {default}]: ").strip()
        idx = default if raw == "" else (int(raw) if raw.isdigit() else -1)
        if 0 <= idx < len(items):
            return items[idx]
        print(f"  ! enter a number 0-{len(items) - 1}.")


def pick_many(title, items):
    """Accepts '0,2', '0-3' or 'all' (default)."""
    print(f"\n{title}")
    for i, it in enumerate(items):
        print(f"  [{i}] {it}")
    while True:
        raw = input("indices (e.g. 0,2 or 0-3 or all) [default: all]: ").strip().lower()
        try:
            if raw in ("", "all"):
                sel = list(range(len(items)))
            elif "-" in raw and "," not in raw:
                lo, hi = raw.split("-")
                sel = list(range(int(lo), int(hi) + 1))
            else:
                sel = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("  ! invalid input.")
            continue
        if sel and all(0 <= i < len(items) for i in sel):
            return [items[i] for i in dict.fromkeys(sel)]
        print(f"  ! indices must be in 0-{len(items) - 1}.")


def available_splits(run_dir):
    names = []
    for f in os.listdir(run_dir):
        m = re.fullmatch(r"ped_intents_(.+)\.pkl", f)
        if m:
            names.append(m.group(1))
    return sorted(names)


def interactive_select():
    frameworks = _list_dirs(EXPERIMENTS_ROOT)
    if not frameworks:
        raise FileNotFoundError(f"no framework folders under {EXPERIMENTS_ROOT}")
    framework = frameworks[0] if len(frameworks) == 1 else pick_one("[framework]", frameworks)

    arch_root = os.path.join(EXPERIMENTS_ROOT, framework)
    architecture = pick_one("[architecture folder]", _list_dirs(arch_root))

    arch_dir = os.path.join(arch_root, architecture)
    # keep only runs that already have at least one ped_intents file
    runs = [r for r in _list_dirs(arch_dir) if available_splits(os.path.join(arch_dir, r))]
    if not runs:
        raise FileNotFoundError(f"no run with ped_intents_*.pkl under {arch_dir}")
    runs = sorted(runs, key=lambda r: os.path.getctime(os.path.join(arch_dir, r)), reverse=True)
    chosen_runs = pick_many("[runs] (newest first)", runs)

    splits = sorted(set.intersection(*[set(available_splits(os.path.join(arch_dir, r)))
                                       for r in chosen_runs]))
    if not splits:
        raise RuntimeError("the selected runs share no common ped_intents split.")
    split = pick_one("[split] ('val' = overlap 1, 'val_ov05' = overlap 0.5)", splits,
                     default=splits.index("val") if "val" in splits else 0)

    thr_raw = input("\n[threshold] decision threshold [default: 0.5]: ").strip()
    threshold = float(thr_raw) if thr_raw else 0.5
    return [os.path.join(arch_dir, r) for r in chosen_runs], split, threshold


# ---------------------------------------------------------------- data
def _flatten_pid(p):
    while isinstance(p, (list, tuple, np.ndarray)):
        p = p[0]
    return str(p)


def load_window_predictions(run_dir, split):
    with open(os.path.join(run_dir, f"ped_intents_{split}.pkl"), "rb") as fid:
        d = pickle.load(fid)
    prob = np.array([np.asarray(r).ravel()[0] for r in d["results"]], dtype=float)
    y = np.asarray(d["gt"]).astype(int).ravel()
    pid = np.array([_flatten_pid(p) for p in d["ped_id"]])   # window -> first frame's ped_id
    return prob, y, pid


_HUMAN_PROB = None


def human_intention_prob():
    """pid -> intention_prob (mean human rating), read once from pie_database.pkl."""
    global _HUMAN_PROB
    if _HUMAN_PROB is None:
        with open(DB_PATH, "rb") as fid:
            try:
                db = pickle.load(fid)
            except Exception:
                fid.seek(0)
                db = pickle.load(fid, encoding="bytes")
        _HUMAN_PROB = {}
        for sid in db:
            for vid in db[sid]:
                for pid, ann in db[sid][vid].get("ped_annotations", {}).items():
                    _HUMAN_PROB[str(pid)] = float(ann["attributes"]["intention_prob"])
    return _HUMAN_PROB


def per_track(prob, y, pid, include_synthetic=False):
    """Aggregate windows -> one row per ped_id (mean predicted probability)."""
    lut = human_intention_prob()
    rows = {}
    for p, yy, i in zip(prob, y, pid):
        rows.setdefault(i, {"p": [], "y": yy})["p"].append(p)
    ids, mean_p, y_bin, human, n_win = [], [], [], [], []
    n_synth = n_missing = 0
    for i, v in rows.items():
        if AUG_MARKER in i:
            n_synth += 1
            if not include_synthetic:
                continue
        orig = i.split(AUG_MARKER)[0]
        if orig not in lut:
            n_missing += 1
            continue
        ids.append(i); mean_p.append(np.mean(v["p"])); y_bin.append(v["y"])
        human.append(lut[orig]); n_win.append(len(v["p"]))
    if n_synth:
        print(f"  [info] synthetic tracks: {n_synth} "
              f"({'kept' if include_synthetic else 'dropped from scatter'})")
    if n_missing:
        print(f"  [warning] {n_missing} tracks have no intention_prob in the database.")
    return (np.array(ids), np.array(mean_p), np.array(y_bin),
            np.array(human), np.array(n_win))


# ---------------------------------------------------------------- plot
def plot_run(run_dir, split, threshold=0.5, bins=30, include_synthetic=False):
    prob, y, pid = load_window_predictions(run_dir, split)
    edges = np.linspace(0.0, 1.0, bins + 1)            # same bins for both classes
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    auc_win = roc_auc_score(y, prob) if n_neg and n_pos else float("nan")

    ids, tp, ty, human, n_win = per_track(prob, y, pid, include_synthetic)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    ax = axes[0]
    ax.hist(prob[y == 0], bins=edges, alpha=0.6, label=f"Negative (n={n_neg})")
    ax.hist(prob[y == 1], bins=edges, alpha=0.6, label=f"Positive (n={n_pos})")
    ax.axvline(threshold, ls="--", color="k")
    ax.set(xlabel="Predicted probability", ylabel="Count (windows)", title="Counts")
    ax.legend(); ax.grid(alpha=0.2)

    ax = axes[1]
    ax.hist(prob[y == 0], bins=edges, alpha=0.6, density=True, label="Negative")
    ax.hist(prob[y == 1], bins=edges, alpha=0.6, density=True, label="Positive")
    ax.axvline(threshold, ls="--", color="k")
    ax.set(xlabel="Predicted probability", ylabel="Density (per class)",
           title="Per-class normalized")
    ax.legend(); ax.grid(alpha=0.2)

    ax = axes[2]
    rho = pr = auc_trk = float("nan")
    if len(ids) > 2:
        sizes = 10 + 25 * (n_win / n_win.max())
        for lab, name in ((0, "binary GT = 0"), (1, "binary GT = 1")):
            m = ty == lab
            ax.scatter(human[m], tp[m], s=sizes[m], alpha=0.55, label=f"{name} (n={m.sum()})")
        ax.axvline(0.5, ls=":", color="gray")
        ax.axhline(threshold, ls="--", color="k")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        rho = spearmanr(human, tp)[0]
        pr = pearsonr(human, tp)[0]
        auc_trk = roc_auc_score(ty, tp) if len(np.unique(ty)) > 1 else float("nan")
        ax.set_title(f"Per track: Spearman={rho:.3f}  Pearson={pr:.3f}")
    ax.set(xlabel="Human intention_prob (ground truth)",
           ylabel="Mean predicted probability (per track)")
    ax.legend(); ax.grid(alpha=0.2)

    run_name = os.path.basename(run_dir)
    fig.suptitle(f"{run_name[:90]}\n{split}: window AUC={auc_win:.4f}  "
                 f"track AUC={auc_trk:.4f}  pos_rate(windows)={n_pos / len(y):.3f}", fontsize=10)
    fig.tight_layout()

    out_dir = os.path.join(run_dir, "metrics_plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"prob_distribution_{split}.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")
    plt.show()

    # numeric summaries
    print(f"\n{run_name}  [{split}]")
    for name, v in (("negative", prob[y == 0]), ("positive", prob[y == 1])):
        print(f"  {name:9s} mean={v.mean():.3f} std={v.std():.3f} "
              f"p10={np.percentile(v, 10):.3f} median={np.median(v):.3f} "
              f"p90={np.percentile(v, 90):.3f} frac>={threshold}: {(v >= threshold).mean():.3f}")
    if len(ids) > 2:
        print("  per human-intention_prob bin (tracks):")
        print(f"  {'bin':>10} {'n':>5} {'mean_pred':>10} {'frac>=thr':>10}")
        for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]:
            m = (human >= lo) & (human < hi)
            if m.any():
                print(f"  {lo:>4.1f}-{min(hi, 1.0):<4.1f} {m.sum():>5d} "
                      f"{tp[m].mean():>10.3f} {(tp[m] >= threshold).mean():>10.3f}")


if __name__ == "__main__":
    run_dirs, split, threshold = interactive_select()
    for rd in run_dirs:
        plot_run(rd, split, threshold)
        
        