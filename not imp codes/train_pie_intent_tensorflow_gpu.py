'''
train_pie_intent_tensorflow_gpu.py

Purpose: the GPU version of the baseline (convlstm_encdec) training script
on the PIE dataset, using the same routing through PIEIntentExperiment and
the same shared VGG16/database/sequence caches keyed off PIE_PATH.

NOTE: this script is now pure orchestration -- interactive selection ->
build -> balance -> train -> eval -> calibrate -> summary. All computation
and formatting logic lives in train_helpers.py.
'''
from focal_loss import binary_focal_loss, weighted_binary_crossentropy
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
import sys
import time
import json
import pickle
from log_mode import is_full
from train_helpers import (print_best_epoch_summary,
    interactive_pick_val_augment_mode,
    interactive_pick_architecture,
    interactive_pick_new_run_name,
    interactive_pick_balance_mode,
    interactive_pick_monitor_metric,
    interactive_pick_loss_choice,
    interactive_pick_log_mode,
    interactive_pick_data_split,
    interactive_pick_clip_mode,
    interactive_pick_weight_decay,
    build_split_sequences,
    resolve_balance_mode_preset,
    apply_balancing_strategy,
    build_run_hyperparam_tag,
    compute_pos_weight_from_train,
    Tee,
    print_full_epoch_metrics_table,
    print_full_history_table,
    load_ped_intents_pkl,
    find_best_threshold_by_f1,
    evaluate_split_full_metrics,
    recompute_split_metrics_at_threshold,
    print_summary_table,
    print_phase,
    PHASE_BUILD_SEQUENCES,
    PHASE_BALANCING,
    PHASE_LOSS_FINALIZE,
    PHASE_TRAINING,
    PHASE_FULL_EVAL,
    PHASE_CALIBRATION,
    PHASE_RECOMPUTE_CALIBRATED,
    build_calibration_dict,
    build_data_split_tag
    
)

from plot_metrics import plot_training_metrics
from log_mode import configure_tf_logging

PIE_PATH = r'D:\PIE codes\PIE_dataset'
os.environ['PIE_PATH'] = PIE_PATH

from pie_data_Hesam import PIE
from augmentation import augment_pie_tracks

FRAMEWORK = 'tensorflow'
TRAIN_LR        = 2e-6
TRAIN_CLIPVALUE = .2

NUM_HIDDEN_UNITS       = 128        # 128
REGULARIZER_VAL         = 1.53e-5   # 0.005
LSTM_DROPOUT            = 0.5
LSTM_RECURRENT_DROPOUT  = 0.3
CONVLSTM_NUM_FILTERS    = 64        # 64
CONVLSTM_KERNEL_SIZE    = 2

EARLY_STOP_MIN_DELTA = 0.00005
EARLY_STOP_PATIENCE  = 15

SEED = 42                  # change per run for multi-seed experiments
DETERMINISTIC_OPS = False  # True = bit-exact GPU ops, slower, may raise on unsupported ops

EARLY_STOP_MIN_DELTA = 0.00005
EARLY_STOP_PATIENCE  = 15

data_opts = {
    'fstride': 1,
    'sample_type': 'all',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'data_split_type': 'default',
    'seq_type': 'intention',
    'min_track_size': 15,
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

    LOG_MODE = interactive_pick_log_mode()
    print(f"[config] LOG_MODE = {LOG_MODE}")

    BALANCE_MODE = interactive_pick_balance_mode()
    print(f"[config] BALANCE_MODE = {BALANCE_MODE}")

    VAL_AUGMENT_MODE = (interactive_pick_val_augment_mode()
                         if BALANCE_MODE == 'augmentation' else None)
    if VAL_AUGMENT_MODE:
        print(f"[config] VAL_AUGMENT_MODE = {VAL_AUGMENT_MODE}")
    print(f"[config] BALANCE_MODE = {BALANCE_MODE}")

    MONITOR_METRIC, MONITOR_MODE = interactive_pick_monitor_metric()
    print(f"[config] MONITOR_METRIC = {MONITOR_METRIC} (mode={MONITOR_MODE})")

    LOSS_CHOICE_NAME, LOSS_CHOICE_FN = interactive_pick_loss_choice()
    print(f"[config] LOSS_FUNCTION (family) = {LOSS_CHOICE_NAME}")

    if BALANCE_MODE == "undersample_base_article":
        # forced to 'none' regardless -- skip asking, avoid a pointless prompt
        CLIP_TYPE, CLIP_AMOUNT = 'clipvalue', TRAIN_CLIPVALUE  # placeholder, overridden below
    else:
        CLIP_TYPE, CLIP_AMOUNT = interactive_pick_clip_mode(default_amount=TRAIN_CLIPVALUE)
    print(f"[config] CLIP_TYPE = {CLIP_TYPE}  CLIP_AMOUNT = {CLIP_AMOUNT}")

    WEIGHT_DECAY = interactive_pick_weight_decay(default_amount=0.0)
    print(f"[config] WEIGHT_DECAY = {WEIGHT_DECAY}")

    CLIP_TYPE, CLIP_AMOUNT, BATCH_SIZE, NORMALIZE_DECODER_BBOX = resolve_balance_mode_preset(
        BALANCE_MODE, CLIP_TYPE, CLIP_AMOUNT, TRAIN_LR)


    DATA_SPLIT_TYPE, DATA_SPLIT_OPTS, _ = interactive_pick_data_split()
    data_opts['data_split_type'] = DATA_SPLIT_TYPE
    data_opts.update(DATA_SPLIT_OPTS)
    print(f"[config] data_split_type = {DATA_SPLIT_TYPE}")

    print_phase(PHASE_BUILD_SEQUENCES)
    imdb = PIE(data_path=PIE_PATH)
    beh_seq_train, beh_seq_val, beh_seq_test = build_split_sequences(imdb, data_opts)

    lens = [len(t) for t in beh_seq_train['image']]
    print(f"tracks={len(lens)}  min_len={min(lens)}  max_len={max(lens)}")
    min_ts = data_opts['min_track_size']
    print(f"tracks shorter than {min_ts}:", sum(l < min_ts for l in lens))

    print_phase(PHASE_BALANCING)
    beh_seq_train, beh_seq_val = apply_balancing_strategy(
        imdb, beh_seq_train, beh_seq_val, BALANCE_MODE, augment_pie_tracks,
        val_augment_mode=(VAL_AUGMENT_MODE or 'undersample_val'))

    print_phase(PHASE_LOSS_FINALIZE)
    if LOSS_CHOICE_NAME == "weighted_bce":
        POS_WEIGHT = compute_pos_weight_from_train(beh_seq_train)
        LOSS_FUNCTION_NAME = f"weighted_bce_pw{POS_WEIGHT:.4f}"
        LOSS_FUNCTION = weighted_binary_crossentropy(pos_weight=POS_WEIGHT)
    else:
        POS_WEIGHT = None
        LOSS_FUNCTION_NAME = LOSS_CHOICE_NAME
        LOSS_FUNCTION = LOSS_CHOICE_FN
    print(f"[config] LOSS_FUNCTION (final) = {LOSS_FUNCTION_NAME}")

    EXPERIMENTS_ROOT = os.path.join(PIE_PATH, 'experiments')
    DEFAULT_ARCHITECTURE = (
        f"convlstm_encdec_h{NUM_HIDDEN_UNITS}"
        f"_f{CONVLSTM_NUM_FILTERS}_k{CONVLSTM_KERNEL_SIZE}_"
    )
    DATA_SPLIT_TAG = build_data_split_tag(DATA_SPLIT_TYPE, data_opts)
    print(f"[config] DATA_SPLIT_TAG = {DATA_SPLIT_TAG}")

    RUN_HYPERPARAM_TAG = build_run_hyperparam_tag(REGULARIZER_VAL,
                            CLIP_TYPE, CLIP_AMOUNT, BALANCE_MODE, TRAIN_LR, LSTM_DROPOUT,
                            LSTM_RECURRENT_DROPOUT, LOSS_FUNCTION_NAME, MONITOR_METRIC,
                            DATA_SPLIT_TAG, weight_decay=WEIGHT_DECAY,
                            early_stop_min_delta=EARLY_STOP_MIN_DELTA,
                            early_stop_patience=EARLY_STOP_PATIENCE,
                            seed=SEED)

    ARCHITECTURE = interactive_pick_architecture(
        EXPERIMENTS_ROOT, FRAMEWORK, default_architecture=DEFAULT_ARCHITECTURE)
    RUN_NAME = interactive_pick_new_run_name(
        os.path.join(EXPERIMENTS_ROOT, FRAMEWORK, ARCHITECTURE),
        hyperparam_tag=RUN_HYPERPARAM_TAG)

    print(f"\n[config] ARCHITECTURE = {ARCHITECTURE}")
    print(f"[config] RUN_NAME     = {RUN_NAME}")

    import tensorflow as tf
    if DETERMINISTIC_OPS:
        tf.config.experimental.enable_op_determinism()
        print("[seed] TF op determinism ENABLED")
    configure_tf_logging()  # silences TF's own WARNING logs in 'clean' mode
    from experiment_paths import get_experiment_paths

    paths = get_experiment_paths(
        pie_path=PIE_PATH, framework=FRAMEWORK,
        architecture=ARCHITECTURE, run_name=RUN_NAME)

    log_path = os.path.join(paths['log_dir'], f"console_{time.strftime('%d%b%Y-%Hh%Mm%Ss')}.txt")
    _log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, _log_file)
    sys.stderr = Tee(sys.stderr, _log_file)

    print(f"Run dir      -> {paths['run_dir']}")
    print(f"Console log  -> {log_path}")

    print("\n--- architecture hyperparams ---")
    print(f"hidden_units       -> {NUM_HIDDEN_UNITS}")
    print(f"convlstm_filters   -> {CONVLSTM_NUM_FILTERS}")
    print(f"convlstm_kernel    -> {CONVLSTM_KERNEL_SIZE}")
    print(f"regularizer_val    -> {REGULARIZER_VAL}")
    print(f"lstm_dropout       -> {LSTM_DROPOUT}")
    print(f"lstm_rec_dropout   -> {LSTM_RECURRENT_DROPOUT}")

    print("\n--- optimizer/training hyperparams ---")
    print(f"lr                 -> {TRAIN_LR}")
    print(f"clip_type          -> {CLIP_TYPE}")
    print(f"clip_amount        -> {CLIP_AMOUNT}")
    print(f"weight_decay       -> {WEIGHT_DECAY}")
    print(f"batch_size         -> {BATCH_SIZE}")
    print(f"es_min_delta       -> {EARLY_STOP_MIN_DELTA}")
    print(f"es_patience        -> {EARLY_STOP_PATIENCE}")
    print(f"monitor_metric     -> {MONITOR_METRIC} (mode={MONITOR_MODE})")
    print(f"loss               -> {LOSS_FUNCTION_NAME}")

    print("\n--- data hyperparams ---")
    print(f"data_split_type    -> {DATA_SPLIT_TYPE}")
    print(f"balance_mode       -> {BALANCE_MODE}")
    print(f"val_augment_mode   -> {VAL_AUGMENT_MODE}")
    print(f"normalize_decoder_bbox -> {NORMALIZE_DECODER_BBOX}")
    print()

    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs visible to TF:", gpus)
    assert len(gpus) > 0, (
        "No GPU visible to TensorFlow -- check driver/CUDA install, or run "
        "the CPU variant (train_pie_intent_tensorflow_cpu.py) instead."
    )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[warning] could not set memory growth on {gpu}: {e}")
    print("Running with GPU enabled.")

    from pie_intent_experiment import PIEIntentExperiment

    model = PIEIntentExperiment(
        framework=FRAMEWORK, architecture=ARCHITECTURE,
        run_name=RUN_NAME, run_dir=paths['run_dir'],
        num_hidden_units=NUM_HIDDEN_UNITS, regularizer_val=REGULARIZER_VAL,
        lstm_dropout=LSTM_DROPOUT, lstm_recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        convlstm_num_filters=CONVLSTM_NUM_FILTERS, convlstm_kernel_size=CONVLSTM_KERNEL_SIZE,
    )
    model.normalize_decoder_bbox = NORMALIZE_DECODER_BBOX
    print(f"[config] model.normalize_decoder_bbox = {model.normalize_decoder_bbox}")
    print(model)

    n_epochs = 400
    print(f"Epochs for this run -> {n_epochs}")

    print_phase(PHASE_TRAINING)
    print("min_track_size =", data_opts['min_track_size'])
    print("seq_overlap_rate =", data_opts['seq_overlap_rate'])
    saved_files_path = model.train(
                            data_train=beh_seq_train, data_val=beh_seq_val, data_test=beh_seq_test,
                            epochs=n_epochs, loss=[LOSS_FUNCTION], metrics=['accuracy'],
                            batch_size=BATCH_SIZE, optimizer_type='rmsprop',
                            optimizer_params={
                                'lr': TRAIN_LR,
                                'clip_type': CLIP_TYPE,
                                'clip_amount': CLIP_AMOUNT,
                                'weight_decay': WEIGHT_DECAY,
                            },
                            data_opts=data_opts, monitor_metric=MONITOR_METRIC, monitor_mode=MONITOR_MODE,
                            early_stop_min_delta=EARLY_STOP_MIN_DELTA,
                            early_stop_patience=EARLY_STOP_PATIENCE,
                            seed=SEED,
                                    )

    print("encoder seq length:", model._encoder_seq_length)
    print("decoder seq length:", model._decoder_seq_length)
    print("model input shapes:", [i.shape for i in model.train_model.inputs])

    print(f"Model + configs saved to: {saved_files_path}")
    assert os.path.abspath(saved_files_path) == os.path.abspath(paths['run_dir']), (
        "saved_files_path diverged from the intended run_dir -- check "
        "PIEIntentExperiment.get_path overriding logic."
    )

    history_path = paths.get('history_path', os.path.join(paths['run_dir'], 'history.pkl'))
    if not os.path.exists(history_path):
        history_path = os.path.join(saved_files_path, 'history.pkl')

    history = None 

    if os.path.exists(history_path):
        with open(history_path, 'rb') as fid:
            history = pickle.load(fid)
        # NOTE: these metrics use the TRAINING overlap (seq_overlap_rate),
        # not the overlap=1 used in the final per-split evaluation below --
        # they are NOT directly comparable to the SUMMARY tables further down.
        print(f"[note] per-epoch metrics below use seq_overlap_rate="
              f"{data_opts['seq_overlap_rate']} (training overlap) -- "
              f"NOT comparable to the overlap=1 final evaluation numbers below.")
        print_full_history_table(history)
        epoch_metrics_path = os.path.join(saved_files_path, 'epoch_metrics.json')
        print(f"[note] per-epoch metrics below use seq_overlap_rate="
              f"{data_opts['seq_overlap_rate']} (training overlap) -- "
              f"NOT comparable to the overlap=1 final evaluation numbers below.")

        if is_full():
            print_best_epoch_summary(epoch_metrics_path, monitor_metric=MONITOR_METRIC, monitor_mode=MONITOR_MODE)
    else:
        print(f"[warning] could not find history.pkl -- checked: "
              f"{paths.get('history_path')} and {os.path.join(saved_files_path, 'history.pkl')}")

    tf.keras.backend.clear_session()

    print_phase(PHASE_FULL_EVAL)
    train_metrics = evaluate_split_full_metrics(
        model, beh_seq_train, data_opts, saved_files_path, paths, 'train', threshold=0.5)
    val_metrics = evaluate_split_full_metrics(
        model, beh_seq_val, data_opts, saved_files_path, paths, 'val', threshold=0.5)
    test_metrics = evaluate_split_full_metrics(
        model, beh_seq_test, data_opts, saved_files_path, paths, 'test', threshold=0.5)

    print_phase(PHASE_CALIBRATION)
    val_prob, val_true = load_ped_intents_pkl(saved_files_path, 'val')
    best_thresh, best_val_f1 = find_best_threshold_by_f1(val_prob, val_true)

    calibration = build_calibration_dict(
        threshold=best_thresh, metric='f1', val_true=val_true, val_prob=val_prob,
        pos_weight_used=(POS_WEIGHT if LOSS_CHOICE_NAME == "weighted_bce" else None))

    calib_path = os.path.join(saved_files_path, 'calibration.json')
    with open(calib_path, 'w', encoding='utf-8') as fid:
        json.dump(calibration, fid, indent=2)
    print(f"[calibration] best threshold on val = {best_thresh:.2f} "
        f"(val {calibration['metric_optimized']} = {calibration['val_score_at_threshold']:.4f})")
    print_phase(PHASE_RECOMPUTE_CALIBRATED)

    train_metrics_cal = recompute_split_metrics_at_threshold(saved_files_path, paths, 'train', best_thresh)
    val_metrics_cal   = recompute_split_metrics_at_threshold(saved_files_path, paths, 'val',   best_thresh)
    test_metrics_cal  = recompute_split_metrics_at_threshold(saved_files_path, paths, 'test',  best_thresh)
    plot_training_metrics(
                        history, train_metrics, val_metrics, test_metrics, saved_files_path,
    calibrated_metrics=(train_metrics_cal, val_metrics_cal, test_metrics_cal))

    print_summary_table(
        "SUMMARY -- ALL SPLITS (threshold=0.5)",
        f"[note] overlap=1 for all splits above -- not comparable to "
        f"per-epoch val/test numbers printed during training "
        f"(seq_overlap_rate={data_opts['seq_overlap_rate']}).",
        train_metrics, val_metrics, test_metrics)

    print_summary_table(
        f"SUMMARY -- ALL SPLITS (calibrated threshold={best_thresh:.2f})",
        f"[note] same overlap=1 predictions as threshold=0.5 SUMMARY above, "
        f"re-thresholded (no re-inference) using the val-selected threshold "
        f"{best_thresh:.2f}.",
        train_metrics_cal, val_metrics_cal, test_metrics_cal)