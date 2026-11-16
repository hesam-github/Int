"""bbox_sensitivity.py -- does a trained model use the bbox, and its temporal dynamics?

Test windows only (overlap=1, same windows as final evaluation). Variants of the
decoder bbox input, applied AFTER the model's own normalization:

  orig  bbox as given
  zero  bbox set to 0                            -> any use of bbox at all
  mean  bbox replaced by its within-window mean  -> removes motion, keeps position/size
  rev   bbox sequence reversed in time           -> uses the direction of motion
  shuf  bbox of window i replaced by the bbox of a random window in the same chunk
                                                 -> breaks the pairing with the image

Reported per variant: AUC, dAUC vs orig, mean and max |dp| vs orig,
Spearman rho vs orig, and the fraction of decisions at 0.5 that flip.

Usage (pie_tf, from D:\\PIE codes\\PIE). Pass each run_dir WITHOUT a trailing backslash:
    python bbox_sensitivity.py "<run_dir>" ["<run_dir2>" ...]
"""
import json
import os
import pickle
import sys

PIE_PATH = r"D:\PIE codes\PIE_dataset"
CODE_DIR = r"D:\PIE codes\PIE"
os.environ["PIE_PATH"] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
from keras.models import load_model
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from feature_routing import CUSTOM_OBJECTS, restore_routing
from gate_saturation import read_zero_flag, test_chunks
from pie_intent_experiment import PIEIntentExperiment

BATCH = 64
SEED = 0
TOL = 1e-3
VARIANTS = ['orig', 'zero', 'mean', 'rev', 'shuf']


def make_variants(dec, rng):
    n, T, _ = dec.shape
    return {
        'orig': dec,
        'zero': np.zeros_like(dec),
        'mean': np.repeat(dec.mean(axis=1, keepdims=True), T, axis=1),
        'rev': dec[:, ::-1, :].copy(),
        'shuf': dec[rng.permutation(n)],
    }


def analyse(run_dir):
    print(f"\n========== {os.path.basename(run_dir)} ==========")
    with open(os.path.join(run_dir, 'hyperparams.json'), encoding='utf-8') as f:
        hp = json.load(f)
    if not hp.get('use_visual', True):
        print("[skip] use_visual=False: no encoder input in this model; bbox-only sensitivity not run here.")
        return None
    if read_zero_flag(run_dir):
        print("[skip] zero_decoder_bbox=True: this run was trained without bbox; sensitivity is undefined.")
        return None

    with open(os.path.join(run_dir, 'configs.pkl'), 'rb') as f:
        configs = pickle.load(f)
    data_opts = dict(configs[2])
    if 'random_params' in data_opts:
        data_opts['random_params'] = {**data_opts['random_params'], 'regen_data': False}

    exp = PIEIntentExperiment(framework='tensorflow', architecture='analysis',
                              run_name='analysis', run_dir=run_dir)
    exp.load_model_config(configs[0])
    restore_routing(exp, hp)
    exp.normalize_decoder_bbox = bool(hp.get('normalize_decoder_bbox', False))
    model = load_model(os.path.join(run_dir, 'model.h5'), compile=False,
                       custom_objects=CUSTOM_OBJECTS)

    rng = np.random.default_rng(SEED)
    preds = {k: [] for k in VARIANTS}
    ys = []
    for enc, dec, y in test_chunks(exp, configs, data_opts):
        if len(y) == 0:
            continue
        assert dec.shape[-1] == 4, f"decoder width {dec.shape[-1]} != 4: not a bbox-only decoder"
        for name, d in make_variants(dec, rng).items():
            for s in range(0, len(y), BATCH):
                e = min(s + BATCH, len(y))
                preds[name].append(
                    model.predict([enc[s:e], d[s:e]], batch_size=e - s, verbose=0).ravel())
        ys.append(y)

    y = np.concatenate(ys)
    P = {k: np.concatenate(v) for k, v in preds.items()}
    base = P['orig']
    auc_orig = roc_auc_score(y, base)

    saved_path = os.path.join(run_dir, 'metrics_test.json')
    if os.path.isfile(saved_path):
        with open(saved_path, encoding='utf-8') as f:
            saved = json.load(f)['roc_auc']
        print(f"[check] AUC orig recomputed={auc_orig:.4f} saved={saved:.4f}")
        if abs(auc_orig - saved) > TOL:
            raise RuntimeError("AUC differs from metrics_test.json: windows or inputs differ")

    print(f"windows={len(y)}")
    print(f"{'variant':<8}{'AUC':>8}{'dAUC':>9}{'mean|dp|':>10}{'max|dp|':>9}{'rho':>7}{'flip%':>8}")
    rows = {}
    for k in VARIANTS:
        p = P[k]
        auc = roc_auc_score(y, p)
        d = np.abs(p - base)
        rho = spearmanr(p, base).correlation
        flip = float(np.mean((p >= 0.5) != (base >= 0.5)))
        rows[k] = dict(auc=auc, dauc=auc - auc_orig, mean_dp=float(d.mean()),
                       max_dp=float(d.max()), rho=float(rho), flip=flip)
        print(f"{k:<8}{auc:>8.4f}{auc - auc_orig:>+9.4f}{d.mean():>10.4f}{d.max():>9.4f}"
              f"{rho:>7.3f}{100 * flip:>7.1f}%")
    return rows


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python bbox_sensitivity.py "<run_dir>" ["<run_dir2>" ...]')
    for rd in sys.argv[1:]:
        analyse(os.path.normpath(rd.strip().strip('"')))