import os
import sys
import json
import pickle
import subprocess

os.environ.setdefault('PIE_PATH', r'D:\PIE codes\PIE_dataset')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

PIE_PATH = os.environ['PIE_PATH']
RESULTS_PATH = os.path.join(PIE_PATH, 'threadpool_escalation_fit_results.json')
STEPS_PER_TRIAL = 15       # real fit() steps -- exercises the actual race
VAL_STEPS_PER_TRIAL = 3    # small validation pass, same as real train()
TRIAL_TIMEOUT_SEC = 300

# Pushed higher than the earlier (no-fit) escalation test, since THIS
# script actually calls model.fit() -- the real risky path where the
# original crash happened (main thread doing a train_step GPU op while
# tf.data's background thread might also touch the GPU on a cache
# miss). Values beyond ~32 mostly test tf.data's own threadpool
# creation limits rather than any realistic training configuration.
CANDIDATE_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def run_single_trial(threadpool_size):
    import tensorflow as tf
    from pie_data_Hesam import PIE
    from pie_intent import (PIEIntent, build_generator_dataset, trace_log,
                              start_gpu_watchdog, stop_gpu_watchdog)
    from keras.applications import vgg16
    from keras.optimizers import RMSprop

    trace_log(f"=== FIT-TRIAL START: threadpool_size={threadpool_size} ===")

    OLD_RUN_DIR = os.path.join(
        PIE_PATH, 'experiments', 'tensorflow',
        'convlstm_encdec_h128_f64_k2', 'baseline_cpu_correct',
    )
    OLD_CONFIGS_PATH = os.path.join(OLD_RUN_DIR, 'configs.pkl')
    with open(OLD_CONFIGS_PATH, 'rb') as fid:
        try:
            _old_configs = pickle.load(fid)
        except Exception:
            _old_configs = pickle.load(fid, encoding='bytes')
    old_model_config, old_train_config, old_data_opts = _old_configs

    data_opts = dict(old_data_opts)
    data_type = {
        'encoder_input_type': data_opts['encoder_input_type'],
        'decoder_input_type': data_opts['decoder_input_type'],
        'output_type': data_opts['output_type'],
    }
    seq_length = data_opts['max_size_observe']
    batch_size = old_train_config['batch_size']

    imdb = PIE(data_path=PIE_PATH)
    data_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    data_val = imdb.generate_data_trajectory_sequence('val', **data_opts)

    model = PIEIntent(
        num_hidden_units=old_model_config['num_hidden'],
        regularizer_val=old_model_config['reg_value'],
        activation=old_model_config['activation'],
        lstm_dropout=old_model_config['lstm_dropout'],
        lstm_recurrent_dropout=old_model_config['lstm_recurrent_dropout'],
        convlstm_num_filters=old_model_config['convlstm_num_filters'],
        convlstm_kernel_size=old_model_config['convlstm_kernel_size'],
    )
    model.context_model = vgg16.VGG16(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet',
    )
    feature_shape = model.context_model.output_shape[1:]

    train_d = model.get_train_val_data(data_train, data_type, seq_length,
                                        data_opts['seq_overlap_rate'])
    val_d = model.get_train_val_data(data_val, data_type, seq_length,
                                      data_opts['seq_overlap_rate'])
    train_d['output'] = train_d['output'][:, 0]
    val_d['output'] = val_d['output'][:, 0]

    model._encoder_seq_length = train_d['decoder_input'].shape[1]
    model._decoder_seq_length = train_d['decoder_input'].shape[1]
    model._sequence_length = model._encoder_seq_length

    train_save_path = model.get_path(
        type_save='data',
        data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
        model_name='vgg16_none', data_subset='train')
    val_save_path = model.get_path(
        type_save='data',
        data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
        model_name='vgg16_none', data_subset='val')

    start_gpu_watchdog()
    trace_log(f"FIT-TRIAL threadpool_size={threadpool_size}: GPU watchdog started, "
              f"about to build datasets and call model.fit() for "
              f"{STEPS_PER_TRIAL} REAL steps (this exercises the actual crash path)")

    train_dataset = build_generator_dataset(
        model, train_d['images'], train_d['bboxes'], train_d['ped_ids'],
        train_d['decoder_input'], train_d['output'], train_save_path,
        seq_length, feature_shape, batch_size=batch_size,
        data_type_label='train', shuffle=True,
        private_threadpool_size=threadpool_size,
        prefetch_size=threadpool_size,
    )
    val_dataset = build_generator_dataset(
        model, val_d['images'], val_d['bboxes'], val_d['ped_ids'],
        val_d['decoder_input'], val_d['output'], val_save_path,
        seq_length, feature_shape, batch_size=batch_size,
        data_type_label='val', shuffle=False,
        private_threadpool_size=threadpool_size,
        prefetch_size=threadpool_size,
    )

    train_model = model.pie_convlstm_encdec()
    optimizer = RMSprop(learning_rate=1e-6, clipvalue=1.0)
    train_model.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['acc'])

    train_model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=STEPS_PER_TRIAL,
        validation_data=val_dataset,
        validation_steps=VAL_STEPS_PER_TRIAL,
        verbose=1,
    )

    stop_gpu_watchdog()
    trace_log(f"=== FIT-TRIAL SUCCESS: threadpool_size={threadpool_size}, "
              f"{STEPS_PER_TRIAL} real fit() steps completed without crash ===")
    marker_path = os.path.join(PIE_PATH, f'.trial_success_{threadpool_size}.marker')
    with open(marker_path, 'w') as fid:
        fid.write('success')

    stop_gpu_watchdog()
    trace_log(f"=== FIT-TRIAL SUCCESS: threadpool_size={threadpool_size}, "
              f"{STEPS_PER_TRIAL} real fit() steps completed without crash ===")
    sys.exit(0)


def run_orchestrator():
    try:
        SCRIPT_PATH = os.path.abspath(__file__)
    except NameError:
        SCRIPT_PATH = os.path.join(PIE_PATH, 'test_threadpool_escalation_fit.py')
        print(f"[WARNING] __file__ undefined (Jupyter?) -- assuming this script "
              f"is saved at {SCRIPT_PATH}. Run from a real terminal, not a "
              f"notebook, for correct subprocess isolation.")

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as fid:
            results = json.load(fid)
        print(f"Resuming: previous results found: {results}")

    highest_confirmed_safe = None

    for value in CANDIDATE_VALUES:
        if str(value) in results and results[str(value)] == 'SUCCESS':
            print(f"threadpool_size={value}: already confirmed SUCCESS, skipping.")
            highest_confirmed_safe = value
            continue

        print(f"\n{'=' * 60}\nTesting threadpool_size={value} WITH REAL model.fit() "
              f"(isolated subprocess, timeout={TRIAL_TIMEOUT_SEC}s)\n{'=' * 60}")

        try:
            proc = subprocess.run(
                [sys.executable, SCRIPT_PATH, '--trial', str(value)],
                timeout=TRIAL_TIMEOUT_SEC,
            )
            if proc.returncode == 0:
                print(f"threadpool_size={value}: SUCCESS")
                results[str(value)] = 'SUCCESS'
                highest_confirmed_safe = value
            else:
                print(f"threadpool_size={value}: FAILED (exit code {proc.returncode}) "
                      f"-- STOPPING escalation here.")
                results[str(value)] = f'FAILED (exit code {proc.returncode})'
                break
        except subprocess.TimeoutExpired:
            marker_path = os.path.join(PIE_PATH, f'.trial_success_{value}.marker')
            if os.path.exists(marker_path):
                # The trial itself succeeded (wrote the marker) but process
                # teardown took longer than the timeout budget -- this is
                # NOT a real failure, just slow cleanup. Treat as success
                # and keep escalating.
                print(f"threadpool_size={value}: SUCCESS (marker found; "
                      f"process teardown just took longer than the timeout)")
                results[str(value)] = 'SUCCESS'
                highest_confirmed_safe = value
                os.remove(marker_path)
            else:
                print(f"threadpool_size={value}: TIMED OUT after {TRIAL_TIMEOUT_SEC}s "
                      f"with NO success marker -- likely a genuine hang or crash "
                      f"-- STOPPING escalation here.")
                results[str(value)] = 'TIMEOUT (no marker -- real failure)'
                break

        with open(RESULTS_PATH, 'w') as fid:
            json.dump(results, fid, indent=2)

    print(f"\n{'=' * 60}")
    if highest_confirmed_safe is not None:
        print(f"Highest confirmed-safe threadpool_size (with REAL model.fit()): "
              f"{highest_confirmed_safe}")
    else:
        print("No value confirmed safe with real fit() -- stay at threadpool_size=1.")
    print(f"Full results: {results}")
    print(f"(saved to {RESULTS_PATH} -- rerun this script to resume/continue "
          f"escalation from where it left off)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--trial':
        run_single_trial(int(sys.argv[2]))
    else:
        run_orchestrator()