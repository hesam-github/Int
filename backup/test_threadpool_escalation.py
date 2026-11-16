import os
import sys
import json
import subprocess
import pickle

os.environ.setdefault('PIE_PATH', r'D:\PIE codes\PIE_dataset')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

PIE_PATH = os.environ['PIE_PATH']
RESULTS_PATH = os.path.join(PIE_PATH, 'threadpool_escalation_results.json')
BATCHES_PER_TRIAL = 15  # small on purpose -- just enough to exercise the
                        # real generator/threading code path repeatedly,
                        # without waiting for a full epoch each time
TRIAL_TIMEOUT_SEC = 300  # 5 min per trial; if a trial hangs this long
                          # without finishing, treat it as a failure
# ------------------------------------------------------------------
# FIX: __file__ is undefined when this code runs inside a Jupyter
# cell (it's only defined when Python loads code from an actual .py
# file on disk). The orchestrator below needs to re-invoke ITSELF as
# a subprocess, so it needs a real, on-disk path to this exact script
# regardless of how it was launched.
# ------------------------------------------------------------------
try:
    SCRIPT_PATH = os.path.abspath(__file__)
except NameError:
    SCRIPT_PATH = os.path.join(os.environ['PIE_PATH'], 'test_threadpool_escalation.py')
    print(f"[WARNING] __file__ not defined (running in Jupyter/interactive mode). "
          f"Falling back to SCRIPT_PATH={SCRIPT_PATH} -- "
          f"this ONLY works if you have ALREADY SAVED this exact script "
          f"to that exact path on disk. If you haven't, save this cell's "
          f"content to that file first, then re-run.")
    
CANDIDATE_VALUES = [1, 2, 3, 4, 6, 8]


# ------------------------------------------------------------------
# TRIAL MODE: this branch runs INSIDE the subprocess for exactly one
# candidate value. It builds the real train dataset (using the already
# pre-cached, verified-complete feature cache) and iterates a small,
# fixed number of batches through it -- exercising the exact same
# generator + threading code path that caused the original crash --
# without paying for a full epoch or calling model.fit() at all.
# ------------------------------------------------------------------
def run_single_trial(threadpool_size):
    import tensorflow as tf
    from pie_data_Hesam import PIE
    from pie_intent import PIEIntent, build_generator_dataset, trace_log
    from keras.applications import vgg16

    trace_log(f"=== TRIAL START: threadpool_size={threadpool_size} ===")

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
    train_d['output'] = train_d['output'][:, 0]

    train_save_path = model.get_path(
        type_save='data',
        data_type='features' + '_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
        model_name='vgg16_none',
        data_subset='train')

    from pie_intent import start_gpu_watchdog, stop_gpu_watchdog
    start_gpu_watchdog()
    trace_log(f"TRIAL threadpool_size={threadpool_size}: GPU watchdog started, "
              f"about to iterate {BATCHES_PER_TRIAL} batches")

    train_dataset = build_generator_dataset(
        model, train_d['images'], train_d['bboxes'], train_d['ped_ids'],
        train_d['decoder_input'], train_d['output'], train_save_path,
        seq_length, feature_shape, batch_size=batch_size,
        data_type_label='train', shuffle=True,
        private_threadpool_size=threadpool_size,
        prefetch_size=threadpool_size,
    )

    batch_count = 0
    for (enc_in, dec_in), y in train_dataset.take(BATCHES_PER_TRIAL):
        batch_count += 1
        trace_log(f"TRIAL threadpool_size={threadpool_size}: "
                  f"consumed batch {batch_count}/{BATCHES_PER_TRIAL}, "
                  f"enc_in shape={enc_in.shape}")

    stop_gpu_watchdog()

    if batch_count == BATCHES_PER_TRIAL:
        trace_log(f"=== TRIAL SUCCESS: threadpool_size={threadpool_size}, "
                  f"all {BATCHES_PER_TRIAL} batches consumed without crash ===")
        sys.exit(0)
    else:
        trace_log(f"=== TRIAL INCOMPLETE: threadpool_size={threadpool_size}, "
                  f"only {batch_count}/{BATCHES_PER_TRIAL} batches consumed ===")
        sys.exit(2)


# ------------------------------------------------------------------
# ORCHESTRATOR MODE: spawns each trial as a FRESH, ISOLATED subprocess.
# This is deliberate -- if a trial crashes (even a hard OS-level crash),
# it only takes down that one subprocess. The orchestrator (this
# process) stays alive, detects the failure via exit code or timeout,
# and stops escalating immediately rather than trying a larger value
# on top of an already-crashed one.
# ------------------------------------------------------------------
def run_orchestrator():
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as fid:
            results = json.load(fid)
        print(f"Resuming: previous results found: {results}")

    highest_confirmed_safe = None

    for value in CANDIDATE_VALUES:
        if str(value) in results and results[str(value)] == 'SUCCESS':
            print(f"threadpool_size={value}: already confirmed SUCCESS "
                  f"in a previous run, skipping.")
            highest_confirmed_safe = value
            continue

        print(f"\n{'=' * 60}\nTesting threadpool_size={value} "
              f"(isolated subprocess, timeout={TRIAL_TIMEOUT_SEC}s)\n{'=' * 60}")

        try:
            proc = subprocess.run(
            [sys.executable, SCRIPT_PATH, '--trial', str(value)],
            timeout=TRIAL_TIMEOUT_SEC,
                                  )
            if proc.returncode == 0:
                print(f"threadpool_size={value}: SUCCESS "
                      f"(exit code 0, all batches consumed cleanly)")
                results[str(value)] = 'SUCCESS'
                highest_confirmed_safe = value
            else:
                print(f"threadpool_size={value}: FAILED "
                      f"(exit code {proc.returncode}) -- STOPPING escalation here.")
                results[str(value)] = f'FAILED (exit code {proc.returncode})'
                break

        except subprocess.TimeoutExpired:
            print(f"threadpool_size={value}: TIMED OUT after "
                  f"{TRIAL_TIMEOUT_SEC}s -- likely hung or crashed silently. "
                  f"STOPPING escalation here.")
            results[str(value)] = 'TIMEOUT'
            break

        with open(RESULTS_PATH, 'w') as fid:
            json.dump(results, fid, indent=2)

    print(f"\n{'=' * 60}")
    if highest_confirmed_safe is not None:
        print(f"Highest confirmed-safe threadpool_size: {highest_confirmed_safe}")
    else:
        print("No value was confirmed safe -- stay at threadpool_size=1.")
    print(f"Full results: {results}")
    print(f"(saved to {RESULTS_PATH} -- rerun this script to resume/continue "
          f"escalation from where it left off)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--trial':
        run_single_trial(int(sys.argv[2]))
    else:
        run_orchestrator()