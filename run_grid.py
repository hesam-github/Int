"""run_grid.py -- one fixed protocol for every condition of the context/bbox question.

  python run_grid.py                        run every missing (split, condition, seed)
  python run_grid.py --report               print the table only
  python run_grid.py --one COND SPLIT SEED  single run (the loop calls this in a subprocess)
"""
import builtins
import json
import os
import subprocess
import sys
import time

import numpy as np

PIE_PATH = r"D:\PIE codes\PIE_dataset"
os.environ["PIE_PATH"] = PIE_PATH
HERE = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.join(PIE_PATH, "experiments", "grid")
RESULTS = os.path.join(GRID_DIR, "grid_results.jsonl")

SPLITS = ["default", "random"]
SEEDS = [42]                      # add 43, 44 only if stage-1 differences are ambiguous
CONDITIONS = ["full_raw", "full_norm", "vis_only", "bbox_raw", "bbox_norm"]
# "permute" and "gap" can be appended ONLY after the patch is in train_pipeline.py / feature_routing.py

COND = {
    "full_raw":  dict(visual=True,  zero_bbox=False, norm_bbox=False),   # paper reference
    "full_norm": dict(visual=True,  zero_bbox=False, norm_bbox=True),
    "vis_only":  dict(visual=True,  zero_bbox=True,  norm_bbox=False),
    "bbox_raw":  dict(visual=False, zero_bbox=False, norm_bbox=False),
    "bbox_norm": dict(visual=False, zero_bbox=False, norm_bbox=True),
    "permute":   dict(visual=True,  zero_bbox=True,  norm_bbox=False, permute=True),
    "gap":       dict(visual=True,  zero_bbox=True,  norm_bbox=False, gap=True),
}

DATA_OPTS = {
    "fstride": 1, "sample_type": "all", "height_rng": [0, float("inf")],
    "squarify_ratio": 0, "data_split_type": "default", "seq_type": "intention",
    "min_track_size": 15, "max_size_observe": 15, "max_size_predict": 5,
    "seq_overlap_rate": 0.5, "balance": True,            # dead key, kept for compatibility
    "crop_type": "context", "crop_mode": "pad_resize",
    "encoder_input_type": [], "decoder_input_type": ["bbox"],
    "output_type": ["intention_binary"],
}


def make_hp(seed, c):
    hp = dict(
        framework="tensorflow", lr=1e-5, clipvalue=None, batch_size=128,
        hidden_units=128, reg=0.001, lstm_dropout=0.4, lstm_recurrent_dropout=0.2,
        convlstm_filters=64, convlstm_kernel=2,
        es_min_delta=1e-3, es_patience=30,
        plateau_patience=8, plateau_factor=0.5, plateau_min_lr=1e-7, plateau_min_delta=1e-3,
        n_epochs=400, seed=seed, deterministic_ops=True,
        log_mode="clean", threshold_space="raw", decision_threshold=0.5)
    if c.get("permute") or c.get("gap"):
        hp["permute_train_labels"] = bool(c.get("permute"))
        hp["spatial_pool"] = bool(c.get("gap"))
    return hp


def build_answers(c, split, prefix):
    """(keyword, answer) in the exact order of pick_run_options + choose_run_identity."""
    a = [("[offline_features]", "n"),
         ("[use_visual]", "y" if c["visual"] else "n")]
    if c["visual"]:                                   # not asked when use_visual=False
        a.append(("[bbox_mode]", "1" if c["zero_bbox"] else "0"))
    a += [("[log_mode]", "0"),
          ("[balance_mode]", "1"),                    # undersample_base_article (no clip prompt)
          ("[monitor_metric]", "1"),                  # val_auc
          ("[loss_function]", "0"),                   # BCE
          ("[weight_decay]", ""),
          ("Choose [1/2]", "2"),                      # hard_sigmoid
          ("[normalize_bbox]", "1" if c["norm_bbox"] else "0"),
          ("[plateau]", ""),
          ("[eval_every]", "4"),                      # 4 = never (final eval gives train/val/test)
          ("[tte]", "")]
    if split == "default":
        a.append(("[data_split_type]", "0"))
    else:
        a += [("[data_split_type]", "1"), ("three ratios", ""),
              ("regenerate the random split", "n")]
    a += [("[decoder_features]", "0"), ("[extra_norm]", "0")]
    if c["visual"]:                                   # architecture menu only exists with VGG
        a.append(("(default pie_convlstm)", "1"))
    a.append(("[run_name]", prefix))
    return a


def install_bbox_check(c):
    """Fails at once if the decoder matrix is not what the condition says."""
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
    hist = res.get("history") or {}
    val_auc = list(hist.get("val_auc", []))
    rec = dict(cond=cond, split=split, seed=seed, run_dir=res["saved_files_path"],
               epochs_ran=len(val_auc),
               best_epoch=(int(np.argmax(val_auc)) + 1) if val_auc else None,
               minutes=round(minutes, 1))
    for s in ("train", "val", "test"):
        rec[f"{s}_auc"] = float(res["metrics"][s]["roc_auc"])
        rec[f"{s}_n"] = int(res["metrics"][s]["n_test_samples"])
    for s in ("val", "test"):
        p, y, pid = load_ped_intents_full(rec["run_dir"], s)
        _, lo, hi = bootstrap_auc_by_group(y, p, pid, n_boot=1000)
        rec[f"{s}_ci"] = [lo, hi]
        rec[f"{s}_n_peds"] = len(set(pid))
    return rec


def run_one(cond, split, seed):
    c = COND[cond]
    if c.get("permute") or c.get("gap"):
        import inspect
        import train_pipeline
        src = inspect.getsource(train_pipeline)
        assert "permute_train_labels" in src and "spatial_pool" in src, (
            "this train_pipeline.py has no permute/gap patch: the run would silently be a normal one")
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
    res = run_training_pipeline(make_hp(seed, c), DATA_OPTS, PIE_PATH)
    assert not queue, f"unused answers: {queue}"

    o = res["opts"]
    assert o["use_visual"] == c["visual"], o["use_visual"]
    assert o["zero_decoder_bbox"] == c["zero_bbox"], o["zero_decoder_bbox"]
    assert o["normalize_bbox"] == c["norm_bbox"], o["normalize_bbox"]
    assert o["batch_size"] == 128 and o["balance_mode"] == "undersample_base_article"
    assert o["split_type"] == split and o["monitor_metric"] == "val_auc"

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


def video_prior_report(rows):
    """AUC of a predictor that knows only the video id (train positive rate per video)."""
    from sklearn.metrics import roc_auc_score
    from train_helpers import load_ped_intents_full

    def vkey(pid):
        return "_".join(str(pid).split("_")[:2])

    print("\nvideo-id-only prior (approximate: built from the balanced train windows)")
    for split in sorted({r["split"] for r in rows}):
        run_dir = next((r["run_dir"] for r in rows
                        if r["split"] == split and r["cond"] != "permute"), None)
        if run_dir is None:
            continue
        _, y_tr, id_tr = load_ped_intents_full(run_dir, "train")
        y_tr = np.asarray(y_tr).ravel()
        acc = {}
        for v, y in zip(map(vkey, id_tr), y_tr):
            s = acc.setdefault(v, [0.0, 0])
            s[0] += float(y)
            s[1] += 1
        glob = float(y_tr.mean())
        for name in ("val", "test"):
            _, y, ids = load_ped_intents_full(run_dir, name)
            y = np.asarray(y).ravel().astype(int)
            keys = [vkey(i) for i in ids]
            prior = np.array([(acc[k][0] + 1) / (acc[k][1] + 2) if k in acc else glob
                              for k in keys])
            seen = float(np.mean([k in acc for k in keys]))
            auc = roc_auc_score(y, prior) if len(set(y)) == 2 else float("nan")
            print(f"  split={split:<8}{name:<5} AUC={auc:.4f}  windows whose video is in train: {seen:.2f}")


def report():
    rows = load_results()
    if not rows:
        print("no results yet")
        return
    rows.sort(key=lambda r: (r["split"], r["cond"], r["seed"]))
    print(f"\n{'split':<9}{'cond':<11}{'seed':>5}{'best/ran':>10}{'train':>8}{'val':>8}"
          f"{'test':>8}   test 95% CI (by ped_id)   gap(train-val)")
    for r in rows:
        lo, hi = r["test_ci"]
        print(f"{r['split']:<9}{r['cond']:<11}{r['seed']:>5}"
              f"{str(r['best_epoch']) + '/' + str(r['epochs_ran']):>10}"
              f"{r['train_auc']:>8.4f}{r['val_auc']:>8.4f}{r['test_auc']:>8.4f}"
              f"   [{lo:.3f}, {hi:.3f}]         {r['train_auc'] - r['val_auc']:+.3f}")
    print("\nAUC is threshold-free; train/val are balanced, test is natural. "
          "train/val/test = best val_auc checkpoint, overlap=1.")
    try:
        video_prior_report(rows)
    except Exception as exc:                       # report must never hide the main table
        print(f"[video prior skipped] {type(exc).__name__}: {exc}")


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