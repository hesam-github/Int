"""check_grid_consistency.py -- read-only check of grid results.

Lists the grid_results.jsonl record and every run folder on disk for one
(condition, split, seed=42), then compares config files and per-epoch
curves between the runs. Does not load TensorFlow or any model.

Run from D:\\PIE codes\\PIE:
    python check_grid_consistency.py vis_only default
    python check_grid_consistency.py full_norm random
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

PIE_PATH = r"D:\PIE codes\PIE_dataset"
RUNS_ROOT = os.path.join(PIE_PATH, "experiments", "tensorflow", "pie_convlstm")
RESULTS = os.path.join(PIE_PATH, "experiments", "grid", "grid_results.jsonl")
SEED = 42


def read_jsonl(cond, split):
    if not os.path.isfile(RESULTS):
        return []
    out = []
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["cond"] == cond and r["split"] == split and r["seed"] == SEED:
                out.append(r)
    return out


def find_run_dirs(cond, split):
    prefix = f"grid_{cond}_{split}_s{SEED}"
    return sorted(d for d in glob.glob(os.path.join(RUNS_ROOT, prefix + "*"))
                  if os.path.isdir(d))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = json.dumps(obj, sort_keys=True, default=str)
    return out


def diff_dicts(a, b, skip=("run_name", "run_dir")):
    fa, fb = flatten(a), flatten(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if any(k == s or k.startswith(s + ".") for s in skip):
            continue
        if fa.get(k) != fb.get(k):
            rows.append((k, fa.get(k), fb.get(k)))
    return rows


def load_history(run_dir):
    p = os.path.join(run_dir, "history.pkl")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def main(cond, split):
    print(f"=== condition={cond}  split={split}  seed={SEED} ===")

    print("\n[1] records in grid_results.jsonl:")
    recs = read_jsonl(cond, split)
    if not recs:
        print("   (none)")
    for r in recs:
        print(f"   run_dir   = {r['run_dir']}")
        print(f"   test_auc  = {r['test_auc']:.4f}   val_auc = {r['val_auc']:.4f}   "
              f"best_epoch = {r['best_epoch']}   epochs_ran = {r['epochs_ran']}")

    print("\n[2] run folders on disk with this prefix:")
    dirs = find_run_dirs(cond, split)
    if not dirs:
        print("   (none)")
        return
    jsonl_dirs = {os.path.normpath(r["run_dir"]) for r in recs}
    for d in dirs:
        mt = os.path.join(d, "metrics_test.json")
        auc = load_json(mt)["roc_auc"] if os.path.isfile(mt) else float("nan")
        listed = os.path.normpath(d) in jsonl_dirs
        print(f"   {os.path.basename(d)}")
        print(f"      metrics_test roc_auc = {auc:.4f}   listed_in_jsonl = {listed}")

    if len(dirs) < 2:
        print("\n(only one run folder: nothing to compare)")
        return

    base = dirs[0]
    for other in dirs[1:]:
        print(f"\n[3] config diff: {os.path.basename(base)}  vs  {os.path.basename(other)}")
        for name in ("hyperparams.json", "choices.json"):
            pa, pb = os.path.join(base, name), os.path.join(other, name)
            if not (os.path.isfile(pa) and os.path.isfile(pb)):
                print(f"   {name}: missing in one of the runs")
                continue
            rows = diff_dicts(load_json(pa), load_json(pb))
            if not rows:
                print(f"   {name}: identical (ignoring run_name)")
            for k, va, vb in rows:
                print(f"   {name} | {k}: {va}  ->  {vb}")

        print("\n[4] per-epoch curves (first divergence between the two runs):")
        ha, hb = load_history(base), load_history(other)
        if ha is None or hb is None:
            print("   history.pkl missing in one of the runs")
            continue
        print(f"   epochs: {len(ha['loss'])} vs {len(hb['loss'])}")
        for key in ("loss", "val_auc"):
            a = np.asarray(ha[key], dtype=float)
            b = np.asarray(hb[key], dtype=float)
            n = min(len(a), len(b))
            print(f"   {key} first 3 epochs A: {np.round(a[:3], 6).tolist()}")
            print(f"   {key} first 3 epochs B: {np.round(b[:3], 6).tolist()}")
            d = np.abs(a[:n] - b[:n])
            idx = np.flatnonzero(d > 1e-6)
            if idx.size == 0:
                print(f"   {key}: no divergence over the {n} common epochs")
            else:
                print(f"   {key}: first divergence at epoch {idx[0] + 1}, |diff| = {d[idx[0]]:.3g}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python check_grid_consistency.py <cond> <split>")
    main(sys.argv[1], sys.argv[2])