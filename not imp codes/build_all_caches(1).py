"""
build_all_caches.py

Standalone, one-shot script that pre-builds the VGG16 feature cache for
ALL THREE splits (train, val, test) before any training or evaluation
script is run. Consolidates the pre-caching strategy that was previously
scattered across:

  - extract_pie_reference_features.py  (build_and_cache_robust, the actual
    batched-GPU extraction/resume/repair logic -- unchanged, imported as-is)
  - tensorflow_modelwith_tests.ipynb, cell 10 (the CORRECT apply_balance=False
    call for train/val, added after discovering the apply_balance=True bug)
  - pie_intent_experiment.py (confirms test_chunk() hardcodes overlap=1,
    which is why the test split below also uses overlap=1)

WHY apply_balance=False FOR TRAIN AND VAL:
    train()/get_train_val_data() in pie_intent.py never calls
    imdb.balance_samples_count() itself -- balancing (if any) happens in
    the driver script BEFORE calling train(), on top of the FULL sequence
    returned by generate_data_trajectory_sequence(). Caching with
    apply_balance=False builds every frame in the full (unbalanced)
    superset, which covers a balanced subset automatically -- so this
    cache is correct whether or not a given driver script balances its
    data afterward. Using apply_balance=True here (as the old __main__
    block in extract_pie_reference_features.py still does) was the
    ORIGINAL cache-miss bug: it only caches the balanced subset (e.g. 482
    of 880 train tracks), leaving every "extra" unbalanced frame missing.

WHY overlap=data_opts['seq_overlap_rate'] (0.5) FOR TRAIN/VAL BUT
overlap=1 FOR TEST:
    pie_intent.py's train() calls get_train_val_data(..., seq_overlap_rate)
    for train/val (confirmed at lines 836-841 of pie_intent.py).
    pie_intent_experiment.py's test_chunk() hardcodes `overlap = 1`
    (comment: "unchanged from reference") for the test split, regardless
    of data_opts. These two values MUST be reproduced exactly here, or
    the cache built will not match what get_tracks() actually requests at
    train/eval time, and you will see the same "residual cache miss"
    warnings again despite having "already pre-cached everything".

USAGE:
    python build_all_caches.py

    Safe to re-run any time: build_and_cache_robust() only extracts
    frames that are missing or corrupted (checked via a fast existence
    pass, then a deep unpickle-validation pass), so a second run over an
    already-complete cache does no wasted work.
"""

import os
import sys

import extract_pie_reference_features as epf
from keras.applications import vgg16
from pie_data_Hesam import PIE
from pie_intent import PIEIntent

# ----------------------------------------------------------------------
# Must match the data_opts used by every training/evaluation script that
# will consume this cache (train_pie_intent_tensorflow_gpu.py /
# train_pie_intent_tensorflow_cpu.py / evaluate_pretrained_model.py /
# finetune_from_baseline_gpu_lr1e4.py all share this exact dict).
# If you change ANY of these values, the save_path computed inside
# build_and_cache_robust() changes too (it's derived from
# crop_type/crop_mode), so a mismatched data_opts here would silently
# build a cache under the WRONG folder and every downstream script would
# still report 100% misses.
# ----------------------------------------------------------------------
epf.data_opts = {
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
epf.data_type = {
    'encoder_input_type': epf.data_opts['encoder_input_type'],
    'decoder_input_type': epf.data_opts['decoder_input_type'],
    'output_type':        epf.data_opts['output_type'],
}
epf.SEQ_LENGTH = epf.data_opts['max_size_observe']

PIE_PATH = os.environ.get('PIE_PATH', epf.PIE_PATH)
os.environ['PIE_PATH'] = PIE_PATH


def main() -> int:
    epf.configure_gpu()

    imdb = PIE(data_path=PIE_PATH)

    # Model architecture values here only affect get_tracks()'s windowing
    # (seq_length) and get_path()'s folder naming -- they do NOT need to
    # match any specific trained checkpoint's hyperparameters, since the
    # feature cache is architecture-independent (same VGG16 features are
    # reused across every ConvLSTM variant). Keep these as the project
    # defaults used everywhere else.
    model = PIEIntent(
        num_hidden_units=128,
        regularizer_val=0.001,
        lstm_dropout=0.4,
        lstm_recurrent_dropout=0.2,
        convlstm_num_filters=64,
        convlstm_kernel_size=2,
    )
    model.context_model = vgg16.VGG16(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet',
    )

    results = {}

    print("\n" + "#" * 70)
    print("# Pre-caching TRAIN split (FULL, unbalanced -- matches what")
    print("# train() actually consumes; apply_balance=False is intentional,")
    print("# see module docstring)")
    print("#" * 70)
    results['train'] = epf.build_and_cache_robust(
        imdb, model, 'train',
        overlap=epf.data_opts['seq_overlap_rate'],
        apply_balance=False,
    )

    print("\n" + "#" * 70)
    print("# Pre-caching VAL split (FULL, unbalanced)")
    print("#" * 70)
    results['val'] = epf.build_and_cache_robust(
        imdb, model, 'val',
        overlap=epf.data_opts['seq_overlap_rate'],
        apply_balance=False,
    )

    print("\n" + "#" * 70)
    print("# Pre-caching TEST split (overlap=1, matching test_chunk()'s")
    print("# hardcoded 'overlap = 1  # unchanged from reference')")
    print("#" * 70)
    results['test'] = epf.build_and_cache_robust(
        imdb, model, 'test',
        overlap=1,
        apply_balance=False,
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    all_ok = True
    for split, ok in results.items():
        status = "COMPLETE" if ok else "INCOMPLETE"
        print(f"  {split:8s}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n[OK] All three splits are fully cached and deep-validated.")
        print("load_images_and_process() should now hit disk cache for every")
        print("frame in every split -- zero on-demand VGG16 calls during")
        print("either training or evaluation.")
        cache_dir = os.path.join(PIE_PATH, 'data', 'data', 'pie')
        print(f"\nBackup targets:")
        print(f"  {cache_dir}")
        print(f"  {os.path.join(PIE_PATH, 'data_cache')} (pie_database.pkl + seq_*.pkl)")
    else:
        print("\n[STOP] Some splits are still incomplete -- check the")
        print("missing_<split>.txt file(s) written next to this script.")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())