"""
test_pie_intent_tensorflow.py

Runs ONLY test_chunk() against an already-trained model, completely
independent from train_pie_intent_tensorflow.py. No training epochs,
no GPU memory pressure from a training graph still resident, no need
to re-run the 400-epoch training loop just to debug a test-time error.

Usage: set RUN_DIR below to the exact run folder produced by the
training script (the one containing model.h5 and configs.pkl), then run:
    py -3.10 test_pie_intent_tensorflow.py
"""

import os
import sys
import time

PIE_PATH = r'D:\PIE codes\PIE_dataset'
os.environ['PIE_PATH'] = PIE_PATH  # must match the value used at train time

# ============ set this to the run produced by the training script ============
# e.g. r'D:\PIE codes\PIE_dataset\experiments\tensorflow\convlstm_encdec_h128_f64_k2\baseline_10Jul2026-14h30m00s'
RUN_DIR = r'D:\PIE codes\PIE\data\pie\intention\10Jul2026-15h24m02s'
# ===============================================================================

import tensorflow as tf


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(
            "No GPU detected by TensorFlow. Run:\n"
            "    python -c \"import tensorflow as tf; "
            "print(tf.config.list_physical_devices('GPU'))\"\n"
            "and confirm it returns a non-empty list before retrying."
        )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not set memory growth for {gpu}: {e}")
    print(f"GPU configured and visible to TensorFlow: {gpus}")


configure_gpu()

from pie_data_Hesam import PIE
from pie_intent_experiment import PIEIntentExperiment
from intent_metrics import compute_extra_metrics

# Must match extraction / training exactly -- only what's needed to build
# beh_seq_test; the model's own hyperparameters (seq_length, decoder type,
# etc.) come from configs.pkl inside RUN_DIR via load_model_config, NOT
# from this dict.
data_opts = {
    'fstride': 1,
    'sample_type': 'all',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'data_split_type': 'default',
    'seq_type': 'intention',
    'min_track_size': 0,
    'max_size_observe': 15,
    'max_size_predict': 5,
    'seq_overlap_rate': 0.5,
    'balance': True,
    'crop_type': 'context',
    'crop_mode': 'pad_resize',
    'encoder_input_type': [],
    'decoder_input_type': ['bbox'],
    'output_type': ['intention_binary'],
}

if __name__ == '__main__':
    assert RUN_DIR != r'PUT_YOUR_TRAINED_RUN_DIR_HERE', \
        "Set RUN_DIR to the actual trained run folder before running."
    assert os.path.exists(os.path.join(RUN_DIR, 'model.h5')), \
        f"No model.h5 found in {RUN_DIR} -- check the path."
    assert os.path.exists(os.path.join(RUN_DIR, 'configs.pkl')), \
        f"No configs.pkl found in {RUN_DIR} -- check the path."

    imdb = PIE(data_path=PIE_PATH)

    print("Building test sequences (overlap=1, no balancing, matches extraction)...")
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

    model = PIEIntentExperiment(
        framework='tensorflow',
        architecture='convlstm_encdec_h128_f64_k2',
        run_name='test_only',
        run_dir=RUN_DIR,  # not used to write anything new, just kept consistent
    )

    print(f"Loading model + configs from: {RUN_DIR}")
    t0 = time.time()
    acc, f1 = model.test_chunk(
        beh_seq_test,
        data_opts,
        RUN_DIR,
        False,
    )
    print(f"test_chunk finished in {time.time() - t0:.1f}s")
    print(f"\n[Paper metrics] Test accuracy: {acc:.4f}")
    print(f"[Paper metrics] Test F1 score:  {f1:.4f}")

    results_path = os.path.join(RUN_DIR, 'ped_intents.pkl')
    print(f"Per-pedestrian predictions saved to: {results_path}")

    print("\nComputing extra metrics from saved predictions...")
    import pickle
    import numpy as np
    with open(results_path, 'rb') as fid:
        ped_intents = pickle.load(fid)
    y_prob = [np.asarray(r).ravel()[0] for r in ped_intents['results']]
    y_true = ped_intents['gt']
    metrics_path = os.path.join(RUN_DIR, 'metrics.json')
    extra = compute_extra_metrics(y_true, y_prob, save_path=metrics_path)
    print(f"Extra metrics saved to: {metrics_path}")
    for k, v in extra.items():
        print(f"  {k}: {v}")