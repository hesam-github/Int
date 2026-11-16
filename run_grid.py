"""run_grid.py -- one fixed protocol for every condition of the context/bbox question.

  python run_grid.py                        run every missing (split, condition, seed)
  python run_grid.py --report               print the tables only
  python run_grid.py --one COND SPLIT SEED  single run (the loop calls this in a subprocess)
"""
import builtins
import json
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

PIE_PATH = r"D:\PIE codes\PIE_dataset"
os.environ["PIE_PATH"] = PIE_PATH
HERE = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.join(PIE_PATH, "experiments", "grid")
RESULTS = os.path.join(GRID_DIR, "grid_results.jsonl")

SPLITS = ["default", "random"]
SEEDS = [42]
CONDITIONS = [
    "full_raw", "full_norm", "vis_only",
    "full_raw_vnorm", "full_norm_vnorm", "vis_only_vnorm",
    "bbox_raw", "bbox_norm",
]

COND = {
    "full_raw":        dict(visual=True,  zero_bbox=False, norm_bbox=False, vnorm=False),
    "full_norm":       dict(visual=True,  zero_bbox=False, norm_bbox=True,  vnorm=False),
    "vis_only":        dict(visual=True,  zero_bbox=True,  norm_bbox=False, vnorm=False),
    "full_raw_vnorm":  dict(visual=True,  zero_bbox=False, norm_bbox=False, vnorm=True),
    "full_norm_vnorm": dict(visual=True,  zero_bbox=False, norm_bbox=True,  vnorm=True),
    "vis_only_vnorm":  dict(visual=True,  zero_bbox=True,  norm_bbox=False, vnorm=True),
    "bbox_raw":        dict(visual=False, zero_bbox=False, norm_bbox=False, vnorm=False),
    "bbox_norm":       dict(visual=False, zero_bbox=False, norm_bbox=True,  vnorm=False),
}

DATA_OPTS = {
    "fstride": 1, "sample_type": "all", "height_rng": [0, float("inf")],
    "squarify_ratio": 0, "data_split_type": "default", "seq_type": "intention",
    "min_track_size": 15, "max_size_observe": 15, "max_size_predict": 5,
    "seq_overlap_rate": 0.5, "balance": True,
    "crop_type": "context", "crop_mode": "pad_resize",
    "encoder_input_type": [], "decoder_input_type": ["bbox"],
    "output_type": ["intention_binary"],
}


def make_hp(seed):
    return dict(
        framework="tensorflow", lr=1e-5, clipvalue=None, batch_size=128,
        hidden_units=128, reg=0.001, lstm_dropout=0.4, lstm_recurrent_dropout=0.2,
        convlstm_filters=64, convlstm_kernel=2,
        es_min_delta=1e-3, es_patience=15,
        plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7, plateau_min_delta=1e-3,
        n_epochs=400, seed=seed, deterministic_ops=True,
        log_mode="clean", threshold_space="raw", decision_threshold=0.5)


def build_answers(c, split, prefix):
    a = [("[offline_features]", "n"),
         ("[use_visual]", "y" if c["visual"] else "n")]
    if c["visual"]:
        a.append(("[bbox_mode]", "1" if c["zero_bbox"] else "0"))
        a.append(("[visual_norm]", "y" if c["vnorm"] else "n"))
    a += [("[log_mode]", "0"),
          ("[balance_mode]", "1"),
          ("[monitor_metric]", "1"),
          ("[loss_function]", "0"),
          ("[weight_decay]", ""),
          ("Choose [1/2]", "2"),
          ("[normalize_bbox]", "1" if c["norm_bbox"] else "0"),
          ("[plateau]", ""),
          ("[eval_every]", "4"),
          ("[tte]", "")]
    a += [("[seed]", ""), ("[seed]", "")]  # train seed, data seed: keep defaults
    if split == "default":
        a.append(("[data_split_type]", "0"))
        n_seed = 2
    else:
        a += [("[data_split_type]", "1"), ("three ratios", ""),
              ("regenerate the random split", "n")]
        n_seed = 3
    a += [("[seed]", "")] * n_seed
    a += [("[decoder_features]", "0"), ("[extra_norm]", "0")]
    
    if c["visual"]:
        a.append(("(default pie_convlstm)", "1"))
    a.append(("[run_name]", prefix))
    return a


def install_bbox_check(c):
    import pie_intent
    orig = pie_intent.PIEIntent.get_train_val_data

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        dec = np.asarray(out["decoder_input"])
        if c["zero_bbox"]:
            assert not dec.any(), "decoder bbox is NOT zero in a zero-bbox condition"
        else:
            assert dec.any(), "decoder input is all zero in a non-zero condition"
        return out

    pie_intent.PIEIntent.get_train_val_data = wrapped


def summarize(res, cond, split, seed, minutes):
    from train_helpers import load_ped_intents_full, bootstrap_auc_by_group
    c = COND[cond]
    hist = res.get("history") or {}
    val_auc = list(hist.get("val_auc", []))
    run_dir = res["saved_files_path"]
    rec = dict(cond=cond, split=split, seed=seed, run_dir=run_dir,
               vnorm=c["vnorm"], epochs_ran=len(val_auc),
               best_epoch=(int(np.argmax(val_auc)) + 1) if val_auc else None,
               minutes=round(minutes, 1))
    for s in ("train", "val", "test"):
        rec[f"{s}_auc"] = float(res["metrics"][s]["roc_auc"])
        rec[f"{s}_n"] = int(res["metrics"][s]["n_test_samples"])
    for s in ("val", "test"):
        p, y, pid = load_ped_intents_full(run_dir, s)
        _, lo, hi = bootstrap_auc_by_group(y, p, pid, n_boot=1000)
        rec[f"{s}_ci"] = [lo, hi]
        rec[f"{s}_n_peds"] = len(set(pid))
    return rec


def run_one(cond, split, seed):
    c = COND[cond]
    prefix = f"grid_{cond}_{split}_s{seed}"
    queue = build_answers(c, split, prefix)

    def fake_input(prompt=""):
        if not queue:
            raise RuntimeError(f"Unexpected prompt, no answers left: {prompt!r}")
        keyword, ans = queue.pop(0)
        if keyword not in prompt:
            raise RuntimeError(f"Menu order mismatch: expected {keyword!r}, got {prompt!r}")
        print(f"{prompt}{ans}")
        return ans

    builtins.input = fake_input
    install_bbox_check(c)
    from train_pipeline import run_training_pipeline
    t0 = time.time()
    res = run_training_pipeline(make_hp(seed), DATA_OPTS, PIE_PATH)
    assert not queue, f"unused answers: {queue}"

    o = res["opts"]
    assert bool(o["use_visual"]) == c["visual"], o["use_visual"]
    assert bool(o.get("zero_decoder_bbox")) == c["zero_bbox"], o.get("zero_decoder_bbox")
    assert bool(o["normalize_bbox"]) == c["norm_bbox"], o["normalize_bbox"]
    assert bool(o.get("visual_norm")) == c["vnorm"], o.get("visual_norm")
    assert o["batch_size"] == 128 and o["balance_mode"] == "undersample_base_article"
    assert o["split_type"] == split and o["monitor_metric"] == "val_auc"

    has_layer = any(l.name == "visual_standardize" for l in res["model"].train_model.layers)
    assert has_layer == c["vnorm"], "visual_standardize layer presence does not match the condition"

    rec = summarize(res, cond, split, seed, (time.time() - t0) / 60.0)
    os.makedirs(GRID_DIR, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("[grid] recorded:", rec)


def load_results():
    if not os.path.isfile(RESULTS):
        return []
    with open(RESULTS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dir_of(cond, split, seed=42):
    matches = [r for r in load_results()
               if (r["cond"], r["split"], r["seed"]) == (cond, split, seed)]
    return matches[-1]["run_dir"] if matches else None


def paired_auc_diff(dir_a, dir_b, part="test", n_boot=1000, seed=0):
    from train_helpers import load_ped_intents_full
    pa, ya, ida = load_ped_intents_full(dir_a, part)
    pb, yb, idb = load_ped_intents_full(dir_b, part)
    pa = np.asarray(pa, float)
    pb = np.asarray(pb, float)
    ya = np.asarray(ya).ravel().astype(int)
    yb = np.asarray(yb).ravel().astype(int)
    ida = np.asarray(ida)
    idb = np.asarray(idb)
    if not (len(ya) == len(yb) and np.array_equal(ya, yb) and np.array_equal(ida, idb)):
        raise ValueError("windows differ between the two runs; the paired comparison is invalid")
    if len(np.unique(ya)) < 2:
        return None

    point = roc_auc_score(ya, pa) - roc_auc_score(ya, pb)
    uniq = np.unique(ida)
    idx_by = {g: np.where(ida == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in picked])
        if len(np.unique(ya[idx])) < 2:
            continue
        diffs.append(roc_auc_score(ya[idx], pa[idx]) - roc_auc_score(ya[idx], pb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bbox_gain_report():
    pairs = [
        ("bbox gain | raw bbox,  VGG raw ", "full_raw",       "vis_only"),
        ("bbox gain | norm bbox, VGG raw ", "full_norm",      "vis_only"),
        ("bbox gain | raw bbox,  VGG norm", "full_raw_vnorm", "vis_only_vnorm"),
        ("bbox gain | norm bbox, VGG norm", "full_norm_vnorm", "vis_only_vnorm"),
        ("VGG-norm effect | visual only  ", "vis_only_vnorm", "vis_only"),
        ("VGG-norm effect | full, norm bb", "full_norm_vnorm", "full_norm"),
    ]
    print("\nPaired AUC differences on TEST (cluster bootstrap by ped_id, 95% CI)")
    print(f"{'split':<9}{'comparison':<36}{'diff':>9}   95% CI")
    for split in SPLITS:
        for label, a, b in pairs:
            da, db = run_dir_of(a, split), run_dir_of(b, split)
            if da is None or db is None:
                print(f"{split:<9}{label:<36}   (missing run)")
                continue
            out = paired_auc_diff(da, db)
            if out is None:
                print(f"{split:<9}{label:<36}   (single class in test)")
                continue
            point, lo, hi = out
            print(f"{split:<9}{label:<36}{point:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]")


def report():
    rows = load_results()
    if not rows:
        print("no results yet")
        return
    rows.sort(key=lambda r: (r["split"], r["cond"], r["seed"]))
    print(f"\n{'split':<9}{'cond':<17}{'seed':>5}{'best/ran':>10}{'train':>8}{'val':>8}"
          f"{'test':>8}   test 95% CI (ped)      gap(tr-val)")
    for r in rows:
        lo, hi = r["test_ci"]
        print(f"{r['split']:<9}{r['cond']:<17}{r['seed']:>5}"
              f"{str(r['best_epoch']) + '/' + str(r['epochs_ran']):>10}"
              f"{r['train_auc']:>8.4f}{r['val_auc']:>8.4f}{r['test_auc']:>8.4f}"
              f"   [{lo:.3f}, {hi:.3f}]     {r['train_auc'] - r['val_auc']:+.3f}")
    print("\nAUC is threshold-free. train/val are balanced, test is natural. "
          "Checkpoint = best val_auc, overlap=1.")
    try:
        bbox_gain_report()
    except Exception as exc:
        print(f"[bbox gain report skipped] {type(exc).__name__}: {exc}")


def main():
    os.makedirs(GRID_DIR, exist_ok=True)
    done = {(r["cond"], r["split"], r["seed"]) for r in load_results()}
    for split in SPLITS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                if (cond, split, seed) in done:
                    print(f"[grid] skip (already done): {cond} {split} seed={seed}")
                    continue
                print(f"\n[grid] ===== {cond} | {split} | seed={seed} =====", flush=True)
                rc = subprocess.call(
                    [sys.executable, os.path.abspath(__file__), "--one", cond, split, str(seed)],
                    cwd=HERE)
                if rc != 0:
                    print(f"[grid] FAILED (exit {rc}): {cond} {split} seed={seed} -- continuing")
    report()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--one"]:
        run_one(args[1], args[2], int(args[3]))
    elif args[:1] == ["--report"]:
        report()
    else:
        main()