"""
experiment_paths.py

Framework-agnostic helper for organizing per-experiment outputs while
keeping heavy, architecture-independent caches (VGG16 features, PIE
database/sequence pickles) shared and reused across every experiment.

Use the SAME function from both the TensorFlow script and any PyTorch
script so folder naming stays consistent and comparable.
"""

import os
import time
import jdatetime


def get_experiment_paths(pie_path, framework, architecture, run_name,
                          experiments_root=None, timestamp=None):
    """
    :param pie_path: root of the dataset (same PIE_PATH used everywhere;
        NEVER change this per-experiment, since pie_data_Hesam.py's own
        internal caching (pie_database.pkl, seq_*.pkl) is keyed off it).
    :param framework: e.g. 'tensorflow' or 'pytorch'
    :param architecture: free-form name identifying the model variant,
        e.g. 'convlstm_encdec_h128_f64_k2', 'gru_baseline', 'transformer_v1'
    :param run_name: your label for this specific run, e.g. 'baseline',
        'lr_1e-5_run2'
    :param experiments_root: override if you want experiments stored
        somewhere other than PIE_PATH/experiments
    :param timestamp: override for reproducibility/testing; default is
        current time expressed as a Jalali (Persian solar hijri) date,
        e.g. '10_5_1405' for day 10, month 5, year 1405. Includes
        hour/minute/second too, to avoid collisions between multiple
        runs started on the same Jalali day.
    :return: dict of paths, all already created on disk except model_path/
        config_path/history_path/results_path/metrics_path themselves
        (only their parent run_dir is created).
    """
    experiments_root = experiments_root or os.path.join(pie_path, 'experiments')
    if timestamp is None:
        now = jdatetime.datetime.now()
        # DD_M_YYYY-HHhMMmSSs, e.g. '10_5_1405-14h32m07s'
        timestamp = now.strftime("%d_%m_%Y-%Hh%Mm%Ss")
        # strip leading zero from month/day for the exact '10_5_1405' shape
        # requested (day/month without zero-padding, year unchanged).
        date_part, time_part = timestamp.split('-', 1)
        d, m, y = date_part.split('_')
        timestamp = f"{int(d)}_{int(m)}_{y}-{time_part}"
    run_folder_name = f"{run_name}_{timestamp}"

    run_dir = os.path.join(experiments_root, framework, architecture, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)

    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    return {
        'pie_path': pie_path,                              # shared base — do not vary per run
        'run_dir': run_dir,                                 # this run's own output folder
        'log_dir': log_dir,
        'model_path': os.path.join(run_dir, 'model.h5'),    # TF/Keras extension; PyTorch script
                                                             # can build its own e.g. 'model.pt'
                                                             # from run_dir instead of this key
        'config_path': os.path.join(run_dir, 'configs'),
        'history_path': os.path.join(run_dir, 'history.pkl'),
        'results_path': os.path.join(run_dir, 'ped_intents.pkl'),
        'metrics_path': os.path.join(run_dir, 'metrics.json'),
        'run_folder_name': run_folder_name,
    }