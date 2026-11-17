"""capacity_budget.py -- interactive capacity presets with an analytic parameter budget.

Pure Python by default: the budget comes from the layer formulas, so the
parameter cost is visible BEFORE training. --verify also builds the Keras
graph (needs TensorFlow, env pie_tf) and compares count_params() with the formula.

Run (from D:\\PIE codes\\PIE):
    python capacity_budget.py            interactive menu + budget table
    python capacity_budget.py --verify   same, plus Keras cross-check

Used by train_pipeline.py through interactive_pick_capacity() and capacity_tag().
Every prompt starts with "[capacity]" so run_grid.py can answer it by keyword.
"""
import sys

SEQ_LEN = 15
CONV_CH = 512        # VGG16 block5_pool output channels
SPATIAL = 7          # 224 / 32
KERNEL = 2           # convlstm_kernel_size used by the project
BBOX_DIM = 4
BASELINE_TOTAL = 1837953   # model.summary() of the current pie_convlstm run (use_visual=True)

PRESETS = [
    ("baseline (paper design)",
     dict(encoder_reduce=0, input_reduce=0, hidden_units=128)),
    ("A: 1x1 bottleneck after ConvLSTM (64 -> 8)",
     dict(encoder_reduce=8, input_reduce=0, hidden_units=128)),
    ("B: 1x1 input projection (512 -> 64)",
     dict(encoder_reduce=0, input_reduce=64, hidden_units=128)),
    ("A+B",
     dict(encoder_reduce=8, input_reduce=64, hidden_units=128)),
    ("A+B with decoder hidden units 64",
     dict(encoder_reduce=8, input_reduce=64, hidden_units=64)),
    ("custom (asked step by step)", None),
]


# ----------------------------------------------------------------------
# Analytic parameter formulas (Keras conventions)
# ----------------------------------------------------------------------
def convlstm_params(cin, filters):
    # input kernel + recurrent kernel + bias, for 4 gates
    return (KERNEL * KERNEL * cin * 4 * filters
            + KERNEL * KERNEL * filters * 4 * filters
            + 4 * filters)


def conv1x1_params(cin, cout):
    return cin * cout + cout


def lstm_params(d_in, units):
    return (d_in + units) * 4 * units + 4 * units


def dense_params(d_in, units=1):
    return d_in * units + units


def budget(spec):
    """Returns (rows, total, flat) for the visual branch + decoder."""
    rows = []
    cin = CONV_CH
    if spec.get("log1p"):
        rows.append(("log1p on VGG features (no params)", 0))
    r_in = spec.get("input_reduce", 0)
    if r_in:
        rows.append((f"input 1x1 conv {CONV_CH} -> {r_in}", conv1x1_params(CONV_CH, r_in)))
        cin = r_in
    filters = spec.get("convlstm_filters", 64)
    rows.append((f"ConvLSTM2D {cin} -> {filters}, kernel {KERNEL}, valid",
                 convlstm_params(cin, filters)))
    side = SPATIAL - KERNEL + 1            # valid padding: 7 -> 6
    cout = filters
    r_enc = spec.get("encoder_reduce", 0)
    if r_enc:
        rows.append((f"encoder 1x1 conv {filters} -> {r_enc}", conv1x1_params(filters, r_enc)))
        cout = r_enc
    flat = side * side * cout
    hidden = spec["hidden_units"]
    rows.append((f"decoder LSTM in={flat + BBOX_DIM} (flat {flat} + bbox {BBOX_DIM}), units={hidden}",
                 lstm_params(flat + BBOX_DIM, hidden)))
    rows.append((f"decoder Dense {hidden} -> 1", dense_params(hidden)))
    total = sum(p for _, p in rows)
    return rows, total, flat


def print_budget(spec, rows, total, flat):
    print("\n[capacity] parameter budget (visual branch + decoder):")
    width = max(len(name) for name, _ in rows) + 2
    for name, p in rows:
        print(f"  {name:<{width}}{p:>12,}")
    print("  " + "-" * (width + 12))
    print(f"  {'TOTAL':<{width}}{total:>12,}")
    print(f"  {'share of baseline':<{width}}{100.0 * total / BASELINE_TOTAL:>11.1f}%")
    print(f"[capacity] decoder LSTM input width = {flat + BBOX_DIM}")
    print_advice(spec, total, flat)


def print_advice(spec, total, flat):
    is_baseline = (spec.get("encoder_reduce", 0) == 0 and spec.get("input_reduce", 0) == 0
                   and spec.get("hidden_units", 128) == 128)
    if is_baseline and not spec.get("log1p"):
        print("[capacity] baseline selected: this is the paper design, no capacity change.")
    if total > 0.5 * BASELINE_TOTAL and not is_baseline:
        print("[capacity] WARNING: still above 50% of the baseline. An ablation at this size "
              "may not isolate capacity from information content.")
    if spec.get("encoder_reduce") and not spec.get("input_reduce"):
        print("[capacity] note: the 512-channel ConvLSTM input kernel (~590k) still dominates "
              "the encoder. Add B to test the input projection separately.")
    if spec.get("input_reduce") and not spec.get("encoder_reduce"):
        print(f"[capacity] note: the decoder LSTM still reads a {flat}-wide flatten. "
              "Most parameters remain. Add A to reduce it.")
    if spec.get("log1p"):
        print("[capacity] log1p changes the feature distribution, not only the size. "
              "Judge it with gate_saturation, not with AUC alone.")
    print("[capacity] reminder: any reduction deviates from the paper. Report it as an ablation "
          "and compare with a parameter-matched bbox-only control.")


def capacity_tag(spec):
    """Folder-name fragment. Empty string for the baseline."""
    if not spec:
        return ""
    parts = []
    if spec.get("input_reduce"):
        parts.append(f"I{spec['input_reduce']}")
    if spec.get("encoder_reduce"):
        parts.append(f"E{spec['encoder_reduce']}")
    if spec.get("hidden_units", 128) != 128:
        parts.append(f"H{spec['hidden_units']}")
    if spec.get("log1p"):
        parts.append("log1p")
    if spec.get("percentile"):
        parts.append("P99")
    if spec.get("return_sequences"):
        parts.append("RS")
        
    if not parts:
        return ""
    return "Cap_" + "_".join(parts) + "_"


# ----------------------------------------------------------------------
# Interactive prompts (all start with "[capacity]")
# ----------------------------------------------------------------------
def _ask_index(prompt, n, default):
    while True:
        raw = input(f"[capacity] {prompt} [0-{n - 1}, default {default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and int(raw) < n:
            return int(raw)
        print("  ! invalid choice, try again.")


def _ask_option(prompt, options, default):
    while True:
        raw = input(f"[capacity] {prompt} {options} [default {default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and int(raw) in options:
            return int(raw)
        print(f"  ! choose one of {options}.")


def _ask_yes_no(prompt, default=False):
    hint = "Y/n" if default else "y/N"
    raw = input(f"[capacity] {prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def interactive_pick_capacity():
    print("\n[capacity] presets for the visual branch "
          "(every choice except the baseline deviates from the paper):")
    for i, (name, _) in enumerate(PRESETS):
        mark = "  <-- default" if i == 0 else ""
        print(f"  [{i}] {name}{mark}")
    idx = _ask_index("select preset", len(PRESETS), 0)
    name, preset = PRESETS[idx]

    spec = {"convlstm_filters": 64, "log1p": False}
    if preset is None:
        print("[capacity] custom: one value per question, Enter keeps the default shown.")
        spec["encoder_reduce"] = _ask_option("encoder 1x1 channels after ConvLSTM (0 = none)",
                                             [0, 4, 8, 16, 32], 0)
        spec["input_reduce"] = _ask_option("input 1x1 channels before ConvLSTM (0 = none)",
                                           [0, 32, 64, 128], 0)
        spec["hidden_units"] = _ask_option("decoder LSTM hidden units", [32, 64, 128], 128)
    else:
        spec.update(preset)

    spec["log1p"] = _ask_yes_no("add log1p to VGG features before the ConvLSTM (tail test)?",
                                default=False)
    spec["return_sequences"] = _ask_yes_no(
        "ConvLSTM return_sequences=True? (sequence of maps instead of last step only)",
        default=False)

    rows, total, flat = budget(spec)
    print_budget(spec, rows, total, flat)
    return spec


# ----------------------------------------------------------------------
# Keras cross-check (optional, needs TensorFlow)
# ----------------------------------------------------------------------
def build_keras(spec):
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Conv2D, TimeDistributed, ConvLSTM2D,
                                         Flatten, RepeatVector, Concatenate, LSTM, Dense)
    from tensorflow.keras.models import Model

    maps = Input(shape=(SEQ_LEN, SPATIAL, SPATIAL, CONV_CH), name="encoder_input")
    x = maps
    if spec.get("log1p"):
        x = tf.math.log1p(tf.maximum(x, 0.0))
    if spec.get("input_reduce"):
        x = TimeDistributed(Conv2D(spec["input_reduce"], 1), name="input_reduce")(x)
    x = ConvLSTM2D(filters=spec.get("convlstm_filters", 64), kernel_size=KERNEL,
                   return_sequences=False, name="conv_lstm2d")(x)
    if spec.get("encoder_reduce"):
        x = Conv2D(spec["encoder_reduce"], 1, name="encoder_reduce")(x)
    x = Flatten(name="encoder_flatten")(x)
    dec = Input(shape=(SEQ_LEN, BBOX_DIM), name="decoder_input")
    x = RepeatVector(SEQ_LEN)(x)
    x = Concatenate(axis=2)([x, dec])
    x = LSTM(spec["hidden_units"], name="decoder_network")(x)
    out = Dense(1, activation="sigmoid", name="decoder_dense")(x)
    return Model([maps, dec], out)


def verify(spec, analytic_total):
    model = build_keras(spec)
    keras_total = model.count_params()
    print(f"\n[verify] analytic total = {analytic_total:,}")
    print(f"[verify] keras count    = {keras_total:,}")
    if keras_total == analytic_total:
        print("[verify] OK: formula matches the Keras graph.")
    else:
        print("[verify] MISMATCH: check the formulas before trusting the budget.")
    base_model = build_keras(PRESETS[0][1] | {"convlstm_filters": 64, "log1p": False})
    print(f"[verify] baseline keras count = {base_model.count_params():,} "
          f"(expected {BASELINE_TOTAL:,})")


if __name__ == "__main__":
    spec = interactive_pick_capacity()
    print(f"\n[capacity] spec = {spec}")
    print(f"[capacity] folder tag = '{capacity_tag(spec) or '(none, baseline)'}'")
    if "--verify" in sys.argv:
        _, total, _ = budget(spec)
        verify(spec, total)