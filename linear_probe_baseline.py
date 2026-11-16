"""linear_probe_baseline.py -- logistic regression on the frozen VGG16 features.

Question: does the ConvLSTM over 15 frames add information beyond a linear
model on the SAME frozen per-frame VGG16 features plus the bbox?

Identical to the grid runs:
  - split: data_split_type='default' (train set01,02,04 / val set05,06 / test set03)
  - DATA_OPTS, and the track-level undersampling of train and val (same call)
  - features: read from the per-frame cache the ConvLSTM reads (no re-extraction)
  - evaluation windows: overlap=1 on val and test
Different by design (printed as NOTES at the end):
  - the probe is linear and sees only the LAST frame (plain or 2x2-pooled)

Run (env pie_tf, from D:\\PIE codes\\PIE):
    python linear_probe_baseline.py
"""
import json
import os
import pickle
import sys
import time

PIE_PATH = r"D:\PIE codes\PIE_dataset"
CODE_DIR = r"D:\PIE codes\PIE"
os.environ["PIE_PATH"] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from pie_data_Hesam import PIE, get_tracks

SEQ_LEN = 15
FEATURE_ROOT = os.path.join(PIE_PATH, "data", "data", "pie")   # <root>/<subset>/features_context_pad_resize/vgg16_none_prep1/
CACHE_DIR = os.path.join(PIE_PATH, "experiments", "linear_probe")
GRID_RESULTS = os.path.join(PIE_PATH, "experiments", "grid", "grid_results.jsonl")
C_GRID = (0.001, 0.01, 0.1, 1.0)
N_BOOT = 500

DATA_OPTS = {
    "fstride": 1, "sample_type": "all", "height_rng": [0, float("inf")],
    "squarify_ratio": 0, "data_split_type": "default", "seq_type": "intention",
    "min_track_size": 15, "max_size_observe": 15, "max_size_predict": 5,
    "seq_overlap_rate": 0.5, "balance": True,
    "crop_type": "context", "crop_mode": "pad_resize",
    "encoder_input_type": [], "decoder_input_type": ["bbox"],
    "output_type": ["intention_binary"],
}


def ask_choice(prompt, options, default):
    print(f"[probe] {prompt}")
    for i, (name, _) in enumerate(options):
        mark = "  <-- default" if i == default else ""
        print(f"  [{i}] {name}{mark}")
    raw = input(f"[probe] select [0-{len(options) - 1}, default {default}]: ").strip()
    if raw == "":
        return options[default][1]
    if raw.isdigit() and int(raw) < len(options):
        return options[int(raw)][1]
    print("  ! invalid, using default")
    return options[default][1]


def frame_feature_path(subset, img_path, pid):
    parts = os.path.normpath(img_path).split(os.sep)
    set_id, vid_id = parts[-3], parts[-2]
    frame = os.path.splitext(parts[-1])[0]
    return os.path.join(FEATURE_ROOT, subset, "features_context_pad_resize",
                        "vgg16_none_prep1", set_id, vid_id, f"{frame}_{pid}.pkl")


def pool_grid(feat, g):
    """feat: (7, 7, C) -> (g*g*C,). g=1 is the global mean."""
    if g == 1:
        return feat.mean(axis=(0, 1))
    parts = []
    for rows in np.array_split(np.arange(feat.shape[0]), g):
        for cols in np.array_split(np.arange(feat.shape[1]), g):
            parts.append(feat[np.ix_(rows, cols)].mean(axis=(0, 1)))
    return np.concatenate(parts)


def load_split(split, overlap, grid):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{split}_ov{overlap}_g{grid}.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        print(f"[probe] {split}: loaded cache {os.path.basename(cache)} (windows={len(z['y'])})")
        return {k: z[k] for k in ("vis", "box", "y", "pid")}

    imdb = PIE(data_path=PIE_PATH)
    raw = imdb.generate_data_trajectory_sequence(split, **DATA_OPTS)
    if split in ("train", "val"):
        raw = imdb.balance_samples_count(raw, label_type="intention_binary")

    dtype = {"encoder_input_type": [], "decoder_input_type": ["bbox"],
             "output_type": ["intention_binary"]}
    d, images, bboxes, pids = get_tracks(raw, dtype, SEQ_LEN, overlap)
    labels = d["intention_binary"]

    vis, box, y, pid_list = [], [], [], []
    missing = 0
    t0 = time.time()
    for i in range(len(images)):
        pid = str(pids[i][-1][0])
        path = frame_feature_path(split, images[i][-1], pid)
        if not os.path.isfile(path):
            missing += 1
            continue
        with open(path, "rb") as fh:
            feat = np.squeeze(pickle.load(fh))
        vis.append(pool_grid(feat, grid).astype(np.float32))
        box.append(np.asarray(bboxes[i][-1], dtype=np.float64))
        y.append(int(np.asarray(labels[i][-1]).ravel()[0]))
        pid_list.append(pid)
        if (i + 1) % 5000 == 0:
            print(f"[probe]   {split}: {i + 1}/{len(images)} windows read")

    if not y:
        raise RuntimeError(f"{split}: no usable windows. Check FEATURE_ROOT={FEATURE_ROOT}")
    y = np.asarray(y, dtype=int)
    print(f"[probe] {split}: windows={len(images)} used={len(y)} missing_features={missing} "
          f"positive_rate={y.mean():.4f} time={time.time() - t0:.0f}s")
    out = dict(vis=np.stack(vis), box=np.stack(box), y=y, pid=np.array(pid_list))
    np.savez(cache, **out)
    return out


def make_X(D, use_vis, use_box):
    parts = []
    if use_vis:
        parts.append(D["vis"].astype(np.float64))
    if use_box:
        parts.append(D["box"])
    return np.concatenate(parts, axis=1)


def cluster_bootstrap_ci(y, p, groups, seed=SEED_BOOT if False else 0):
    """95% CI of ROC-AUC, resampling whole pedestrians (ped_id clusters)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(N_BOOT):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_probe(Xtr, ytr, Xva, yva, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(Xtr_s, ytr)
        va_auc = roc_auc_score(yva, clf.predict_proba(Xva_s)[:, 1])
        print(f"    C={C:<7} val AUC = {va_auc:.4f}")
        if best is None or va_auc > best[0]:
            best = (va_auc, C, clf)
    va_auc, C, clf = best             # C chosen on VAL only, never on test
    p_test = clf.predict_proba(Xte_s)[:, 1]
    return dict(C=C, val_auc=va_auc, p_test=p_test)


def load_grid_default():
    out = {}
    if not os.path.isfile(GRID_RESULTS):
        return out
    with open(GRID_RESULTS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["split"] == "default" and r["seed"] == 42:
                out[r["cond"]] = r
    return out


def print_verdict(results, grid_rec):
    if "vis_only" not in grid_rec:
        print("\n[probe] VERDICT skipped: no vis_only record in grid_results.jsonl")
        return
    cl = grid_rec["vis_only"]
    lr = results["VGG only"]
    gap = cl["test_auc"] - lr["test_auc"]
    cl_lo, cl_hi = cl["test_ci"]
    ci_overlap = not (lr["ci"][1] < cl_lo or cl_hi < lr["ci"][0])
    print("\n[probe] VERDICT (VGG-only probe vs ConvLSTM with bbox zeroed, default split):")
    print(f"   ConvLSTM test AUC - probe test AUC = {gap:+.4f}")
    if abs(gap) < 0.01:
        print("   -> the probe matches the ConvLSTM encoder: no measurable gain from the "
              "spatio-temporal encoder on this split.")
    elif gap > 0:
        print("   -> the ConvLSTM is ahead. Claim a gain only if its CI clears the probe CI.")
    else:
        print("   -> the probe is ahead: the ConvLSTM encoder is hurting on this split. "
              "Report this explicitly.")
    if ci_overlap:
        print("   -> the 95% intervals overlap: one split and one seed do not resolve this.")


def main():
    print("=== linear probe baseline (frozen VGG16 + bbox, last frame) ===")
    grid = ask_choice("pooling of the 7x7 VGG map",
                      [("mean over 7x7 -> 512 features", 1),
                       ("2x2 grid pool -> 2048 features", 2)], 0)
    overlap = ask_choice("train window overlap (val and test are always 1.0)",
                         [("0.5 (as in training)", 0.5), ("1.0 (faster)", 1.0)], 0)

    t0 = time.time()
    tr = load_split("train", overlap, grid)
    va = load_split("val", 1.0, grid)
    te = load_split("test", 1.0, grid)
    print(f"\n[probe] VGG feature dim = {tr['vis'].shape[1]}; load time {time.time() - t0:.0f}s")

    feature_sets = [("bbox only", False, True),
                    ("VGG only", True, False),
                    ("VGG + bbox", True, True)]
    results = {}
    for name, use_vis, use_box in feature_sets:
        print(f"\n[probe] --- {name} ---")
        res = fit_probe(make_X(tr, use_vis, use_box), tr["y"],
                        make_X(va, use_vis, use_box), va["y"],
                        make_X(te, use_vis, use_box))
        test_auc = roc_auc_score(te["y"], res["p_test"])
        lo, hi = cluster_bootstrap_ci(te["y"], res["p_test"], te["pid"])
        acc = accuracy_score(te["y"], res["p_test"] >= 0.5)
        results[name] = dict(val_auc=res["val_auc"], test_auc=test_auc, ci=(lo, hi), C=res["C"])
        print(f"[probe] {name}: C={res['C']} val AUC={res['val_auc']:.4f} "
              f"test AUC={test_auc:.4f} [{lo:.4f}, {hi:.4f}] test acc@0.5={acc:.4f}")

    print("\n[probe] SUMMARY (test, overlap=1, cluster bootstrap over ped_id)")
    print(f"   {'feature set':<14}{'val AUC':>9}{'test AUC':>10}   95% CI")
    for name, r in results.items():
        print(f"   {name:<14}{r['val_auc']:>9.4f}{r['test_auc']:>10.4f}   "
              f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")

    grid_rec = load_grid_default()
    if grid_rec:
        print("\n[probe] ConvLSTM grid, default split (from grid_results.jsonl):")
        for cond in ("bbox_raw", "bbox_norm", "vis_only", "full_raw", "full_norm"):
            if cond in grid_rec:
                r = grid_rec[cond]
                lo, hi = r["test_ci"]
                print(f"   {cond:<11} test AUC = {r['test_auc']:.4f}  CI [{lo:.3f}, {hi:.3f}]")
        print("   mapping: bbox_* <-> 'bbox only';  vis_only <-> 'VGG only' (bbox zeroed);  "
              "full_* <-> 'VGG + bbox'")

    print_verdict(results, grid_rec)

    print("\n[probe] NOTES")
    print("   1. The probe sees only the LAST frame. A temporal-mean probe needs 15 reads per "
          "window and is not included here.")
    print("   2. Train and val are undersampled as in training. Test is natural (not balanced).")
    print("   3. One split and one seed: differences below about 0.02 AUC are within CI width.")


if __name__ == "__main__":
    main()