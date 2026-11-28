"""Evaluate one finished run. Does not train and does not open menus.

Reads <run_dir>/choices.json and hyperparams.json, rebuilds the same
tracks through prepare_data, then calls evaluate_run. Overlap 1 on
train, val and test is the normal final pass, not the training overlap.
"""
import os
import re
import sys
import json
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_AUTO_LOSS = re.compile(r'^wbce_focal_g([0-9.]+)_w([0-9.]+)_auto$')
_FINAL_PW = re.compile(r'_pw([0-9.]+)$')


def _restore_loss_marker(opts):
    """choices.json drops loss_fn. Rebuild only the markers prepare_data understands."""
    name = opts.get('loss_name')
    if name == 'weighted_bce':
        opts['loss_fn'] = None
        return
    if name == 'binary_crossentropy':
        opts['loss_fn'] = 'binary_crossentropy'
        return
    if isinstance(name, str):
        match = _AUTO_LOSS.fullmatch(name)
        if match:
            opts['loss_fn'] = (
                'wbce_focal_auto', float(match.group(1)), float(match.group(2)))
            return
    opts['loss_fn'] = name


def _paths_for_existing_run(pie_path, run_dir):
    """Do not call get_experiment_paths: a non-empty folder would become _run2."""
    return {
        'pie_path': pie_path,
        'run_dir': run_dir,
        'log_dir': os.path.join(run_dir, 'logs'),
        'model_path': os.path.join(run_dir, 'model.h5'),
        'config_path': os.path.join(run_dir, 'configs'),
        'history_path': os.path.join(run_dir, 'history.pkl'),
        'results_path': os.path.join(run_dir, 'ped_intents.pkl'),
        'metrics_path': os.path.join(run_dir, 'metrics.json'),
        'run_folder_name': os.path.basename(run_dir),
    }


def _check_pos_weight(final_loss_name, pos_weight):
    match = _FINAL_PW.search(final_loss_name or '')
    if match is None:
        return
    expected = float(match.group(1))
    if pos_weight is None:
        print('[eval] this loss baked pos_weight into a closure at training. '
              'prepare_data stored pos_weight=None, so prior-correction stays off.')
        return
    if abs(float(pos_weight) - expected) > 1e-6:
        raise RuntimeError(
            f'rebuilt pos_weight={pos_weight} but this run trained with {expected}. '
            'The tracks are not the training split. Evaluation stopped.')


def evaluate_saved_run(run_dir, pie_path=None):
    """Same final report as run_training_pipeline, without fit.

    run_dir is the folder that contains model.h5, configs.pkl,
    hyperparams.json and choices.json.
    """
    run_dir = os.path.abspath(run_dir)
    pie_path = os.path.abspath(pie_path or os.environ.get('PIE_PATH') or '')
    if not pie_path:
        raise RuntimeError('Pass pie_path or set PIE_PATH.')
    os.environ['PIE_PATH'] = pie_path

    required = ('model.h5', 'configs.pkl', 'hyperparams.json', 'choices.json')
    missing = [name for name in required if not os.path.isfile(os.path.join(run_dir, name))]
    if missing:
        raise RuntimeError(f'{run_dir} is missing {missing}')

    with open(os.path.join(run_dir, 'choices.json'), encoding='utf-8') as fid:
        choices = json.load(fid)
    with open(os.path.join(run_dir, 'hyperparams.json'), encoding='utf-8') as fid:
        saved_hp = json.load(fid)

    opts = dict(choices['opts'])
    hp = dict(choices['hp'])
    _restore_loss_marker(opts)
    data_opts = saved_hp.get('data_opts')
    if not data_opts:
        raise RuntimeError('hyperparams.json has no data_opts. The notebook DATA_OPTS is not a safe substitute.')

    saved_decoder = list(data_opts.get('decoder_input_type') or [])
    chosen_decoder = list(choices['decoder_features'])
    if saved_decoder and saved_decoder != chosen_decoder:
        raise RuntimeError(
            f'decoder mismatch: choices={chosen_decoder} hyperparams={saved_decoder}')

    import tensorflow as tf
    from train_pipeline import (
        prepare_data, evaluate_run, setup_gpu, attach_bbox_branch,
        start_console_log, stop_console_log,
    )
    from pie_intent_experiment import PIEIntentExperiment

    if hp.get('deterministic_ops'):
        tf.config.experimental.enable_op_determinism()
    setup_gpu(tf)

    paths = _paths_for_existing_run(pie_path, run_dir)
    os.makedirs(paths['log_dir'], exist_ok=True)
    log_path = os.path.join(
        paths['log_dir'],
        'console_eval_' + os.path.basename(run_dir) + '.txt')
    start_console_log(log_path)
    try:
        print(f'[eval] run_dir={run_dir}')
        print('[eval] no fit. Best model.h5 is loaded inside test_chunk.')
        data = prepare_data(opts, data_opts, pie_path)
        data['data_opts']['decoder_input_type'] = chosen_decoder
        if 'bbox_branch' in chosen_decoder:
            attach_bbox_branch(data['train'], data['val'], data['test'])
        _check_pos_weight(choices.get('loss_name'), data.get('pos_weight'))

        model = PIEIntentExperiment(
            framework=hp['framework'],
            architecture=choices['architecture'],
            run_name=choices['run_name'],
            run_dir=run_dir,
            num_hidden_units=hp['hidden_units'],
            regularizer_val=hp['reg'],
            lstm_dropout=hp['lstm_dropout'],
            lstm_recurrent_dropout=hp['lstm_recurrent_dropout'],
            convlstm_num_filters=hp['convlstm_filters'],
            convlstm_kernel_size=hp['convlstm_kernel'],
            lstm_recurrent_activation=opts['rec_act'])
        model.flow_representation = opts.get('flow_representation', 'summary')
        model.flow_root = os.path.join(pie_path, 'extra', 'flow')
        model.use_visual = opts.get('use_visual', True)
        model.backbone_name = opts.get('backbone') or 'vgg16'
        model.normalize_decoder_bbox = opts.get('normalize_bbox', False)
        model.feature_norm = opts.get('feature_norm') or {}

        history = None
        if os.path.isfile(paths['history_path']):
            with open(paths['history_path'], 'rb') as fid:
                history = pickle.load(fid)
        else:
            print('[eval] history.pkl is missing. Plots of the training curve are skipped.')

        results = evaluate_run(model, data, run_dir, paths, history, hp)
        print(f'[eval] log -> {log_path}')
        return results
    finally:
        stop_console_log()


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            'usage: python evaluate_saved_run.py <run_dir> [pie_path]')
    _run = sys.argv[1]
    _pie = sys.argv[2] if len(sys.argv) == 3 else None
    evaluate_saved_run(_run, _pie)