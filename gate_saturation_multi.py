"""gate_saturation_multi.py -- measure ConvLSTM and decoder gate saturation
and decoder pre-activation decomposition on any existing run directory.

Works with baseline (2304-d) and capacity-reduced models (e.g. Cap_E8 -> 288-d).

Usage (from D:\\PIE codes\\PIE, conda activate pie_tf):
    python gate_saturation_multi.py "<run_dir1>" ["<run_dir2>" ...]
"""
import gc
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
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model, load_model

from feature_routing import CUSTOM_OBJECTS, restore_routing
from pie_intent_experiment import PIEIntentExperiment
from visual_norm import apply_visual_ops_np, visual_op_order

MAX_WINDOWS_PER_CHUNK = 256
WINDOWS_PER_BATCH = 32
TOL = 1e-3
FORCE_CPU = False
GATES = ["i", "f", "c", "o"]


def saturation_threshold(act_name):
    if act_name == "hard_sigmoid":
        return 2.501
    if act_name == "tanh":
        return 1.819
    if act_name == "sigmoid":
        return 5.0
    return 3.0


class GateStats:
    def __init__(self, T, thr):
        self.thr = thr
        self.n = 0
        self.sat = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.a = 0.0
        self.sat_t = np.zeros(T)
        self.cnt_t = np.zeros(T)

    def add(self, z, t):
        z = np.asarray(z, np.float64)
        s = np.abs(z) > self.thr
        self.n += z.size
        self.sat += int(s.sum())
        self.s1 += float(z.sum())
        self.s2 += float((z * z).sum())
        self.a += float(np.abs(z).sum())
        self.sat_t[t] += s.sum()
        self.cnt_t[t] += z.size

    def report(self):
        mean = self.s1 / self.n
        return dict(thr=self.thr, sat=self.sat / self.n,
                    mean_abs=self.a / self.n,
                    std=float(np.sqrt(max(self.s2 / self.n - mean * mean, 0.0))),
                    sat_t=self.sat_t / np.maximum(self.cnt_t, 1))


class Decomposition:
    def __init__(self, units):
        self.u = units
        self.t_sum = np.zeros((3, 4))
        self.t_cnt = 0
        self.v1 = np.zeros((4, units))
        self.v2 = np.zeros((4, units))
        self.v_cnt = 0

    def add(self, Z, B, R, V):
        for g in range(4):
            sl = slice(g * self.u, (g + 1) * self.u)
            for row, arr in enumerate((Z, B, R)):
                self.t_sum[row, g] += np.std(arr[:, :, sl], axis=1).sum()
            v = V[:, sl].astype(np.float64)
            self.v1[g] += v.sum(axis=0)
            self.v2[g] += (v * v).sum(axis=0)
        self.t_cnt += Z.shape[0] * self.u
        self.v_cnt += Z.shape[0]

    def report(self):
        t_std = self.t_sum / max(self.t_cnt, 1)
        mean = self.v1 / max(self.v_cnt, 1)
        v_std = np.sqrt(np.maximum(self.v2 / max(self.v_cnt, 1) - mean * mean, 0.0)).mean(axis=1)
        return t_std, v_std


def np_activation(name, x):
    if name == "hard_sigmoid":
        return np.clip(0.2 * x + 0.5, 0.0, 1.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(name)


def convlstm_pass(x_np, K, R, b, rec_name, act_name, gs):
    n, T = x_np.shape[0], x_np.shape[1]
    x = tf.constant(x_np)
    kt, rt, bt = tf.constant(K), tf.constant(R), tf.constant(b)
    hw = x_np.shape[2] - K.shape[0] + 1
    filters = K.shape[-1] // 4
    rec = tf.keras.activations.get(rec_name)
    act = tf.keras.activations.get(act_name)
    h = tf.zeros((n, hw, hw, filters), tf.float32)
    c = tf.zeros_like(h)
    for t in range(T):
        z = (tf.nn.conv2d(x[:, t], kt, strides=[1, 1, 1, 1], padding="VALID") + bt
             + tf.nn.conv2d(h, rt, strides=[1, 1, 1, 1], padding="SAME"))
        zi, zf, zc, zo = tf.split(z, 4, axis=-1)
        for k, part in enumerate((zi, zf, zc, zo)):
            gs[k].add(part.numpy(), t)
        c = rec(zf) * c + rec(zi) * act(zc)
        h = rec(zo) * act(c)
    return h.numpy()


def decoder_pass(dec_seq, visual, Wd, Ud, bd, units, rec_name, act_name, gs, decomp):
    n, T, _ = dec_seq.shape
    V = visual @ Wd[:-4]
    Wb = Wd[-4:]
    h = np.zeros((n, units))
    c = np.zeros((n, units))
    Z = np.empty((n, T, 4 * units))
    B = np.empty_like(Z)
    R = np.empty_like(Z)
    for t in range(T):
        B[:, t] = dec_seq[:, t] @ Wb
        R[:, t] = h @ Ud
        Z[:, t] = V + B[:, t] + R[:, t] + bd
        zi, zf, zc, zo = np.split(Z[:, t], 4, axis=1)
        for k, zk in enumerate((zi, zf, zc, zo)):
            gs[k].add(zk, t)
        i = np_activation(rec_name, zi)
        f = np_activation(rec_name, zf)
        g = np_activation(act_name, zc)
        o = np_activation(rec_name, zo)
        c = f * c + i * g
        h = o * np_activation(act_name, c)
    decomp.add(Z, B, R, V)
    return h


def test_chunks(exp, configs, data_opts):
    from pie_data_Hesam import PIE
    imdb = PIE(data_path=PIE_PATH)
    data = imdb.generate_data_trajectory_sequence("test", **data_opts)
    n = len(data["image"])
    L = exp._sequence_length
    start = 0
    while start < n:
        stop, budget = start, 0
        while stop < n:
            n_win = max(0, len(data["bbox"][stop]) - L + 1)
            if stop > start and budget + n_win > MAX_WINDOWS_PER_CHUNK:
                break
            budget += n_win
            stop += 1
        sub = {k: v[start:stop] for k, v in data.items()
               if isinstance(v, list) and len(v) == n}
        (enc, dec), y = exp.get_test_data(sub, configs[1], L, 1, data_subset="test")
        yield (np.asarray(enc, dtype=np.float32),
               np.asarray(dec, dtype=np.float32),
               np.asarray(y).ravel())
        getattr(exp, "_feat_buffers", {}).clear()
        gc.collect()
        start = stop


def read_zero_flag(run_dir):
    path = os.path.join(run_dir, "choices.json")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        return bool(json.load(f)["opts"].get("zero_decoder_bbox", False))


def find_visual_extractor(model, expected_dim):
    """Return a Model that outputs the exact visual vector fed to the decoder."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            out_dim = layer.output.shape[-1]
            if out_dim == expected_dim:
                return Model(model.inputs[0], layer.output)
    return None


def analyse(run_dir):
    print(f"\n========== {os.path.basename(run_dir)} ==========")
    with open(os.path.join(run_dir, "configs.pkl"), "rb") as f:
        configs = pickle.load(f)
    with open(os.path.join(run_dir, "hyperparams.json"), encoding="utf-8") as f:
        hp = json.load(f)
    zero_bbox = read_zero_flag(run_dir)
    print(f"[config] zero_decoder_bbox={zero_bbox} "
          f"normalize_decoder_bbox={hp.get('normalize_decoder_bbox')} "
          f"visual_norm={'yes' if hp.get('visual_norm') else 'no'}")

    data_opts = dict(configs[2])
    if "random_params" in data_opts:
        data_opts["random_params"] = {**data_opts["random_params"], "regen_data": False}

    exp = PIEIntentExperiment(framework="tensorflow", architecture="analysis",
                              run_name="analysis", run_dir=run_dir)
    exp.load_model_config(configs[0])
    restore_routing(exp, hp)
    exp.normalize_decoder_bbox = bool(hp.get("normalize_decoder_bbox", False))
    T = exp._sequence_length

    if FORCE_CPU:
        tf.config.set_visible_devices([], "GPU")

    model = load_model(os.path.join(run_dir, "model.h5"), compile=False,
                       custom_objects=CUSTOM_OBJECTS)
    conv = model.get_layer("conv_lstm2d")
    dec_layer = model.get_layer("decoder_network")
    dense = model.get_layer("decoder_dense")
    K, R_c, b_c = conv.get_weights()
    Wd, U_d, b_d = dec_layer.get_weights()
    Wo, bo = dense.get_weights()
    cc = conv.get_config()
    dc = dec_layer.get_config()
    units = int(dc["units"])

    expected_vis_dim = Wd.shape[0] - 4
    print(f"[config] decoder expects visual dim = {expected_vis_dim}")

    conv_acts = [cc["recurrent_activation"], cc["recurrent_activation"],
                 cc["activation"], cc["recurrent_activation"]]
    dec_acts = [dc["recurrent_activation"], dc["recurrent_activation"],
                dc["activation"], dc["recurrent_activation"]]
    thr = {n: saturation_threshold(n) for n in set(conv_acts + dec_acts)}
    print(f"[config] ConvLSTM acts={cc['recurrent_activation']}/{cc['activation']}  "
          f"decoder acts={dc['recurrent_activation']}/{dc['activation']}  "
          f"thresholds={ {k: round(v, 3) for k, v in thr.items()} }")

    gs_c = [GateStats(T, thr[n]) for n in conv_acts]
    gs_d = [GateStats(T, thr[n]) for n in dec_acts]
    decomp = Decomposition(units)

    vn = hp.get("visual_norm")
    print(f"[config] visual ops order: {visual_op_order(vn, hp.get('capacity')) or 'raw'}")
    v_mean = np.asarray(vn["mean"], np.float32) if vn else None
    v_std = np.asarray(vn["std"], np.float32) if vn else None
    conv_sub = Model(model.inputs[0], conv.output)

    visual_extractor = find_visual_extractor(model, expected_vis_dim)
    if visual_extractor is None:
        print("[info] using manual ConvLSTM flatten (baseline)")
    else:
        print(f"[info] using model Flatten output (dim={expected_vis_dim})")

    y_all, p_model_all, p_manual_all = [], [], []
    checked = False
    for enc, dec_in, y in test_chunks(exp, configs, data_opts):
        if len(y) == 0:
            continue
        if zero_bbox:
            dec_in = np.zeros_like(dec_in)

        for s in range(0, len(y), WINDOWS_PER_BATCH):
            e = min(s + WINDOWS_PER_BATCH, len(y))
            enc_b = enc[s:e]
            dec_b = dec_in[s:e]

            p_model_all.append(
                model.predict([enc_b, dec_b], batch_size=len(enc_b), verbose=0).ravel())

            # Manual ConvLSTM (always done for verification + gate stats)
            enc_n = apply_visual_ops_np(enc_b, vn, hp.get('capacity'))
            visual_raw = convlstm_pass(enc_n, K, R_c, b_c,
                                       cc["recurrent_activation"], cc["activation"], gs_c)
            if not checked:
                ref = conv_sub.predict(enc_b[:8], verbose=0)
                diff_c = float(np.abs(ref - visual_raw[:len(ref)]).max())
                print(f"[check] ConvLSTM manual vs model: max|diff|={diff_c:.2e}")
                if diff_c > TOL:
                    raise RuntimeError("manual ConvLSTM does not match the model")
                checked = True

            # Get the visual vector that actually enters the decoder
            if visual_extractor is not None:
                visual = visual_extractor.predict(enc_b, verbose=0)
                if visual.ndim == 3:
                    visual = visual[:, 0]
            else:
                visual = visual_raw.reshape(e - s, -1)

            if visual.shape[-1] != expected_vis_dim:
                raise RuntimeError(f"visual dim {visual.shape[-1]} != expected {expected_vis_dim}")

            h_dec = decoder_pass(dec_b, visual, Wd, U_d, b_d, units,
                                 dc["recurrent_activation"], dc["activation"], gs_d, decomp)
            logit = h_dec @ Wo + bo
            p_manual_all.append(1.0 / (1.0 + np.exp(-logit.ravel())))

        y_all.append(y)

    y = np.concatenate(y_all)
    p_model = np.concatenate(p_model_all)
    p_manual = np.concatenate(p_manual_all)
    diff_p = float(np.abs(p_model - p_manual).max())
    print(f"[check] decoder manual vs model.predict: max|diff|={diff_p:.2e}")
    if diff_p > TOL:
        raise RuntimeError("manual forward pass does not match model.predict")

    auc = roc_auc_score(y, p_model)
    saved_path = os.path.join(run_dir, "metrics_test.json")
    if os.path.isfile(saved_path):
        with open(saved_path, encoding="utf-8") as f:
            saved = json.load(f)["roc_auc"]
        print(f"[check] test AUC recomputed={auc:.4f} saved={saved:.4f}")

    return report(gs_c, gs_d, decomp)


def report(gs_c, gs_d, decomp):
    for title, gs in (("ConvLSTM", gs_c), ("decoder", gs_d)):
        print(f"\n[{title} gates] saturation = |z| > thr")
        print(f"{'gate':<6}{'thr':>7}{'sat%':>8}{'mean|z|':>9}{'std z':>8}{'sat t0':>8}{'sat tN':>8}")
        for k, g in enumerate(GATES):
            r = gs[k].report()
            print(f"{g:<6}{r['thr']:>7.2f}{100 * r['sat']:>7.1f}%{r['mean_abs']:>9.2f}"
                  f"{r['std']:>8.2f}{100 * r['sat_t'][0]:>7.1f}%{100 * r['sat_t'][-1]:>7.1f}%")

    t_std, v_std = decomp.report()
    print("\n[decoder decomposition] temporal std of Z,B,R; across-window std of V")
    print(f"{'gate':<6}{'Z_t':>9}{'B_t':>9}{'R_t':>9}{'V_win':>9}{'B_t/Z_t':>9}{'V/Z_t':>9}")
    for k, g in enumerate(GATES):
        z, b, rr = t_std[:, k]
        zz = max(z, 1e-12)
        print(f"{g:<6}{z:>9.3f}{b:>9.3f}{rr:>9.3f}{v_std[k]:>9.3f}{b / zz:>9.3f}{v_std[k] / zz:>9.3f}")

    zz = np.maximum(t_std[0], 1e-12)
    return dict(
        conv_sat=float(np.mean([g.report()["sat"] for g in gs_c])),
        dec_sat=float(np.mean([g.report()["sat"] for g in gs_d])),
        B_over_Z=float(np.mean(t_std[1] / zz)),
        V_over_Z=float(np.mean(v_std / zz)),
    )


def main(run_dirs):
    run_dirs = [os.path.normpath(rd.strip().strip('"')) for rd in run_dirs]
    rows = [(os.path.basename(rd), analyse(rd)) for rd in run_dirs]
    print("\n=== summary ===")
    print(f"{'run':<60}{'conv sat%':>11}{'dec sat%':>10}{'B_t/Z_t':>9}{'V/Z_t':>8}")
    for name, s in rows:
        print(f"{name[:58]:<60}{100 * s['conv_sat']:>10.1f}%{100 * s['dec_sat']:>9.1f}%"
              f"{s['B_over_Z']:>9.3f}{s['V_over_Z']:>8.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python gate_saturation_multi.py "<run_dir>" ["<run_dir2>" ...]')
    main(sys.argv[1:])