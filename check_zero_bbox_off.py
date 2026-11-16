"""check_zero_bbox_off.py

Recomputes the TEST ROC-AUC of one finished run in two conditions:
  zero_bbox=True   decoder bbox forced to 0 (the training condition)
  zero_bbox=False  decoder bbox left as stored

For each condition it also reports the fraction of nonzero decoder-bbox
values produced by the data path BEFORE any explicit zeroing in this script.
That fraction shows whether the zeroing happens inside get_test_data().

Run from D:\\PIE codes\\PIE (conda env pie_tf), one line:
    python check_zero_bbox_off.py "<run_dir without trailing backslash>"

Changes to existing files: none. Imports gate_saturation.py read-only
for its test_chunks() helper; that module sets PIE_PATH itself.
"""
import json
import os
import pickle
import sys

CODE_DIR = r"D:\PIE codes\PIE"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import gate_saturation as gs  # sets PIE_PATH before project imports

import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score

from feature_routing import CUSTOM_OBJECTS, restore_routing
from pie_intent_experiment import PIEIntentExperiment


def load_run(run_dir):
    with open(os.path.join(run_dir, "configs.pkl"), "rb") as f:
        configs = pickle.load(f)
    with open(os.path.join(run_dir, "hyperparams.json"), encoding="utf-8") as f:
        hp = json.load(f)

    exp = PIEIntentExperiment(framework="tensorflow", architecture="analysis",
                              run_name="analysis", run_dir=run_dir)
    exp.load_model_config(configs[0])
    restore_routing(exp, hp)  # also sets exp.zero_decoder_bbox from hyperparams.json
    exp.normalize_decoder_bbox = bool(hp.get("normalize_decoder_bbox", False))

    model = load_model(os.path.join(run_dir, "model.h5"), compile=False,
                       custom_objects=CUSTOM_OBJECTS)
    data_opts = dict(configs[2])
    if "random_params" in data_opts:
        data_opts["random_params"] = {**data_opts["random_params"], "regen_data": False}
    return exp, model, configs, data_opts


def auc_with_bbox(run_dir, zero_bbox):
    exp, model, configs, data_opts = load_run(run_dir)
    # Must be set BEFORE test_chunks(): get_test_data() reads this attribute.
    exp.zero_decoder_bbox = zero_bbox

    probs, labels = [], []
    nonzero, total = 0, 0
    for enc, dec, y in gs.test_chunks(exp, configs, data_opts):
        if len(y) == 0:
            continue
        # Measure what the data path produced, before any explicit zeroing here.
        nonzero += int(np.count_nonzero(dec))
        total += dec.size
        if zero_bbox:
            dec = np.zeros_like(dec)
        for s in range(0, len(y), gs.WINDOWS_PER_BATCH):
            e = min(s + gs.WINDOWS_PER_BATCH, len(y))
            probs.append(model.predict([enc[s:e], dec[s:e]], batch_size=e - s,
                                       verbose=0).ravel())
        labels.append(y)

    p = np.concatenate(probs)
    t = np.concatenate(labels).astype(int)
    return roc_auc_score(t, p), nonzero / max(total, 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('usage: python check_zero_bbox_off.py "<run_dir>"')
    run_dir = os.path.normpath(sys.argv[1].strip().strip('"'))
    with open(os.path.join(run_dir, "metrics_test.json"), encoding="utf-8") as f:
        saved = json.load(f)["roc_auc"]

    auc_on, frac_on = auc_with_bbox(run_dir, zero_bbox=True)
    auc_off, frac_off = auc_with_bbox(run_dir, zero_bbox=False)

    print(f"\nrun: {os.path.basename(run_dir)}")
    print(f"saved metrics_test.json roc_auc         : {saved:.4f}")
    print(f"zero_bbox=True  recomputed roc_auc      : {auc_on:.4f}")
    print(f"zero_bbox=True  nonzero bbox fraction   : {frac_on:.4f}   (from data path, before explicit zeroing)")
    print(f"zero_bbox=False recomputed roc_auc      : {auc_off:.4f}")
    print(f"zero_bbox=False nonzero bbox fraction   : {frac_off:.4f}")
    print(f"difference (off - on)                   : {auc_off - auc_on:+.4f}")