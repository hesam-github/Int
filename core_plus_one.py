"""One parallel feature on the 2019 core. Does not change training updates.

Copy this whole file to D:\\PIE codes\\PIE\\core_plus_one.py and run:

    python core_plus_one.py core
    python core_plus_one.py core 2
    python core_plus_one.py light_color
    python core_plus_one.py table

Recipe pinned to the 23 Nov default-split runs: batch 128, seed 1,
hard_sigmoid, raw bbox, no gradient clip, balance none, val_auc,
weighted BCE + focal, early stop 1e-4 / 50, eval_every off.
Aux loss is 0. Those runs used 0.2 to 0.5, which trains each branch
to predict the label by itself. That is a different experiment.

The core job is branchattn_temporal + gated_add with decoder ['bbox'] only.
With no extra, that graph is ConvLSTM context, RepeatVector, bbox, one LSTM.
If the installed gated_add is the later independent-gate version, this
script builds that version. It does not rebuild the 23 Nov single
relation LSTM.
"""
import builtins
import glob
import json
import os
import sys
import time

import numpy as np

PIE_PATH = os.environ.get('PIE_PATH', r'D:\PIE codes\PIE_dataset')
TABLE_PATH = os.path.join(PIE_PATH, 'experiments', 'core_plus_one_val.json')
LOG_DIR = os.path.join(PIE_PATH, 'experiments', 'core_plus_one_logs')

HP = dict(
    framework='tensorflow',
    lr=1e-5, clipvalue=None,
    batch_size=128,
    hidden_units=128, reg=0.001,
    lstm_dropout=0.4,
    lstm_recurrent_dropout=0.2,
    convlstm_filters=64, convlstm_kernel=2,
    es_min_delta=1e-4,
    es_patience=50,
    plateau_patience=5,
    plateau_factor=0.5,
    plateau_min_lr=1e-7,
    plateau_min_delta=1e-3,
    n_epochs=400, seed=1, deterministic_ops=True,
    attach_pose=True,
    log_mode='clean',
    threshold_space='raw',
    decision_threshold=0.5,
)

DATA_OPTS = {
    'fstride': 1, 'sample_type': 'all',
    'height_rng': [0, float('inf')], 'squarify_ratio': 0,
    'data_split_type': 'default', 'seq_type': 'intention',
    'min_track_size': 15, 'max_size_observe': 15, 'max_size_predict': 5,
    'seq_overlap_rate': 0.5,
    'balance': True,
    'crop_type': 'context', 'crop_mode': 'pad_resize',
    'encoder_input_type': [], 'decoder_input_type': ['bbox'],
    'output_type': ['intention_binary'],
}

# offline answer is the offline-features menu, not a feature name.
JOBS = {
    'core': {
        'group': 'core', 'offline': 'n', 'feature': None, 'flow': 'summary',
        'what': '2019 core only. No residual.',
    },
    'core_b32': {
        'group': 'core', 'offline': 'n', 'feature': None, 'flow': 'summary',
        'batch': 32,
        'what': 'Same core at batch 32. The only control for flow_dense.',
    },
    'velocity': {
        'group': 'derived', 'offline': 'n', 'feature': 'velocity', 'flow': 'summary',
        'what': 'Image-plane bbox velocity. Not ego-vehicle speed.',
    },
    'pose_critical': {
        'group': 'offline', 'offline': '0', 'feature': 'pose_critical', 'flow': 'summary',
        'what': '16 body ratios. One pose representative, not pose and pose_dyn too.',
    },
    'scene_critical': {
        'group': 'offline', 'offline': '1', 'feature': 'scene_critical', 'flow': 'summary',
        'what': '16 scene relations. Not the whole scene vector.',
    },
    'light_color': {
        'group': 'offline', 'offline': '1', 'feature': 'light_color', 'flow': 'summary',
        'what': '4 traffic-light numbers from the whole frame, shared by every pedestrian in that frame.',
    },
    'flow_dense': {
        'group': 'offline', 'offline': '2', 'feature': None, 'flow': 'dense',
        'batch': 32,
        'what': 'Dense flow branch. Compare with core_b32, not with core.',
    },
    'look': {
        'group': 'annotation', 'offline': 'n', 'feature': 'look', 'flow': 'summary',
        'what': 'Human look label. Not available as a sensor at deployment.',
    },
    'gesture': {
        'group': 'annotation', 'offline': 'n', 'feature': 'gesture', 'flow': 'summary',
        'what': 'Human gesture label. Not available as a sensor at deployment.',
    },
    'action': {
        'group': 'annotation', 'offline': 'n', 'feature': 'action', 'flow': 'summary',
        'what': 'Human action label. Not the intention label, but annotated by the same people.',
    },
    'occlusion': {
        'group': 'annotation', 'offline': 'n', 'feature': 'occlusion', 'flow': 'summary',
        'what': 'Human occlusion flag, only if it is numeric on the tracks.',
    },
    'cross': {
        'group': 'near_label', 'offline': 'n', 'feature': 'cross', 'flow': 'summary',
        'what': 'Human crossing-action mark. Agreement here is agreement between two human labels.',
    },
    'obd_speed': {
        'group': 'leakage', 'offline': 'n', 'feature': 'obd_speed', 'flow': 'summary',
        'what': 'Ego-vehicle speed. A gain here is the driver leak, not a pedestrian feature.',
    },
}


def _job_or_die(name):
    if name not in JOBS:
        names = ', '.join(JOBS)
        raise SystemExit(f'Unknown job {name!r}. Choose one of: {names}')
    return JOBS[name]


class _Answers(object):
    def __init__(self, job_name, train_seed):
        self.job_name = job_name
        self.job = JOBS[job_name]
        self.train_seed = train_seed

    def __call__(self, prompt=''):
        text = str(prompt)
        value = self._value(text)
        if value is None:
            raise RuntimeError(
                'This screen has no answer for a menu prompt. Training was not started.\n'
                + text)
        shown = value if value != '' else '<Enter>'
        print(f'[screen] {shown}  <- {text.strip()[:140]}', flush=True)
        return value

    def _value(self, text):
        job = self.job
        if '[offline_features]' in text:
            return job['offline']
        if 'default pie_convlstm' in text:
            return '5'
        if 'Select [1-4] (default 1)' in text:
            return '4'
        if 'replace the visual temporal block' in text:
            return 'n'
        if 'replace the per-branch LSTM' in text:
            return 'n'
        if '[balance_mode]' in text:
            return '2'
        if '[monitor_metric]' in text:
            return '1'
        if '[loss_function]' in text:
            return '4'
        if '[clip_mode]' in text:
            return '3'
        if '[flow_representation]' in text:
            return '1' if job['flow'] == 'dense' else '0'
        if '[extra_norm]' in text:
            return '1' if job['feature'] else '0'
        if '[run_name]' in text:
            return 'scr_' + self.job_name
        if 'Choose [1/2]' in text:
            return '2'
        if '[normalize_bbox]' in text:
            return '0'
        if '[eval_every]' in text:
            return '4'
        if '[pose_min_height]' in text:
            return '1'
        if '[seed] train seed' in text and self.train_seed is not None:
            return str(self.train_seed)
        if '[seed] shuffle seed' in text and self.train_seed is not None:
            return str(self.train_seed)
        blanks = (
            '[use_visual]', '[backbone]', '[bbox_mode]', '[visual_norm]',
            '[visual_percentile]', '[capacity]', '[log_mode]',
            '[bce_plus_focal]', 'choice [0/1]', '[weight_decay]',
            'Choose [1/2]', '[normalize_bbox]', '[plateau]', '[eval_every]',
            '[data_split_type]', '[seed]', '[pose_scale]', '[pose_min_height]',
            '[tte]', 'aux_loss_weight', 'Select [0/1] (default 0)', '0..',
        )
        if any(bit in text for bit in blanks):
            return ''
        return None


def _install_decoder_picker(job_name):
    import train_helpers
    import train_pipeline
    feature = JOBS[job_name]['feature']

    def pick(beh_seq, use_visual=True):
        stds = train_helpers.print_feature_diagnostics(beh_seq)
        if feature is None:
            print('[screen] decoder_input_type = [bbox]', flush=True)
            return ['bbox'], stds
        if feature not in beh_seq:
            numeric = []
            for key, value in beh_seq.items():
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                arr = np.asarray(value[0])
                if arr.dtype != object and getattr(arr.dtype, 'kind', '') in 'iufb':
                    numeric.append(key)
            raise SystemExit(
                f'[screen] {feature!r} is not on the tracks. Numeric keys: {numeric}')
        arr = np.asarray(beh_seq[feature][0])
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            raise SystemExit(f'[screen] {feature!r} is not a numeric feature.')
        print(f'[screen] decoder_input_type = [bbox, {feature}]', flush=True)
        return ['bbox', feature], stds

    train_pipeline.interactive_pick_decoder_features = pick


def _score_binary(y, prob):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int).ravel()
    prob = np.asarray(prob, dtype=np.float64).ravel()
    n = min(len(y), len(prob))
    y, prob = y[:n], prob[:n]
    pred = (prob >= 0.5).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    recall = float(((pred == 1) & (y == 1)).sum() / pos) if pos else float('nan')
    spec = float(((pred == 0) & (y == 0)).sum() / neg) if neg else float('nan')
    bacc = 0.5 * (recall + spec)
    try:
        auc = float(roc_auc_score(y, prob)) if pos and neg else float('nan')
    except ValueError:
        auc = float('nan')
    return auc, bacc


def _group_of(name):
    if name.startswith(('branch_', 'gate_', 'flow_')):
        return 'branch'
    if name in ('core_decoder', 'decoder_network') or 'conv_lstm' in name:
        return 'core'
    if name == 'decoder_dense':
        return 'head'
    return 'other'


def _move(now, init):
    num = 0.0
    den = 0.0
    for left, right in zip(now, init):
        num += float(np.sum(np.square(left - right)))
        den += float(np.sum(np.square(right)))
    return float(np.sqrt(num) / (np.sqrt(den) + 1e-8))


class BranchTrainingEffect(object):
    """Logs weight movement and, every few epochs, validation with gates forced shut.

    This object is appended to the existing callback list. It does not edit
    gradients, the loss, or the optimizer.
    """

    def __init__(self, dataset, steps, probe_every, log_path):
        self.dataset = dataset
        self.steps = None if steps is None else int(steps)
        self.probe_every = int(probe_every)
        self.log_path = log_path
        self.model = None
        self.init = {}
        self.y = None
        self._told_core = False

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.init = {}
        for layer in self.model.layers:
            weights = layer.get_weights()
            if weights:
                self.init[layer.name] = [np.array(w, copy=True) for w in weights]
        self.y = self._labels()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        moves = {'core': [], 'branch': [], 'head': []}
        for name, init in self.init.items():
            layer = self.model.get_layer(name)
            now = layer.get_weights()
            if now and _group_of(name) in moves:
                moves[_group_of(name)].append(_move(now, init))
        summary = {key: (float(np.mean(vals)) if vals else 0.0) for key, vals in moves.items()}
        record = {
            'epoch': int(epoch) + 1,
            'overlap': 0.5,
            'core_move': summary['core'],
            'branch_move': summary['branch'],
            'head_move': summary['head'],
            'open_auc': None if logs.get('val_auc') is None else float(logs['val_auc']),
        }
        gates = self._gate_layers()
        do_probe = gates and self.dataset is not None and self.steps and (
            epoch == 0 or (epoch + 1) % self.probe_every == 0)
        if do_probe:
            saved = [(layer, layer.get_weights()) for layer in gates]
            try:
                for layer, weights in saved:
                    layer.set_weights([
                        np.zeros_like(weights[0]),
                        np.full_like(weights[1], -30.0),
                    ])
                closed = self._predict()
            finally:
                for layer, weights in saved:
                    layer.set_weights(weights)
            if closed is not None and self.y is not None:
                auc, bacc = _score_binary(self.y, closed)
                record['closed_auc'] = auc
                record['closed_bacc'] = bacc
                if record['open_auc'] is not None and auc == auc:
                    record['d_auc'] = float(record['open_auc'] - auc)
        elif not gates and not self._told_core:
            print('[effect] no residual gate. Weight movement is the core itself.', flush=True)
            self._told_core = True
        line = (
            f"[effect] epoch {record['epoch']:3d}  overlap=0.5"
            f"  core_move={record['core_move']:.4f}"
            f"  branch_move={record['branch_move']:.4f}"
            f"  head_move={record['head_move']:.4f}"
        )
        if 'd_auc' in record:
            line += (
                f"  open_auc={record['open_auc']:.4f}"
                f"  closed_auc={record['closed_auc']:.4f}"
                f"  d_auc={record['d_auc']:+.4f}"
                f"  closed_bAcc={record['closed_bacc']:.4f}"
            )
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as fid:
            fid.write(json.dumps(record) + '\n')

    def _gate_layers(self):
        found = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            if layer.name.startswith('gate_') and len(weights) == 2:
                found.append(layer)
        return found

    def _labels(self):
        if self.dataset is None or not self.steps:
            return None
        rows = []
        for index, batch in enumerate(self.dataset):
            if index >= self.steps:
                break
            rows.append(np.asarray(batch[1]).ravel())
        if not rows:
            return None
        return np.concatenate(rows)

    def _predict(self):
        pred = self.model.predict(self.dataset, steps=self.steps, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return np.asarray(pred).ravel()


def _install_effect_logger(log_path):
    import keras
    import tensorflow as tf
    original = tf.keras.Model.fit

    class _EffectCallback(tf.keras.callbacks.Callback):
        """Keras adapter. The inner object never replaces a training step."""

        def __init__(self, dataset, steps):
            super().__init__()
            self._screen = True
            self.inner = BranchTrainingEffect(dataset, steps, probe_every=5, log_path=log_path)

        def on_train_begin(self, logs=None):
            self.inner.model = self.model
            self.inner.on_train_begin(logs)

        def on_epoch_end(self, epoch, logs=None):
            self.inner.model = self.model
            self.inner.on_epoch_end(epoch, logs)

    def fit(self, *args, **kwargs):
        if getattr(self, 'name', None) in ('model', 'model_aux'):
            callbacks = list(kwargs.get('callbacks') or [])
            if not any(getattr(item, '_screen', False) for item in callbacks):
                callbacks.append(_EffectCallback(
                    kwargs.get('validation_data'),
                    kwargs.get('validation_steps'),
                ))
                kwargs['callbacks'] = callbacks
        return original(self, *args, **kwargs)

    tf.keras.Model.fit = fit
    keras.Model.fit = fit


def _best_overlap_half(run_dir):
    path = os.path.join(run_dir, 'epoch_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fid:
        rows = json.load(fid)
    usable = [row for row in rows if row.get('val_roc_auc') is not None]
    if not usable:
        return None
    best = max(usable, key=lambda row: row['val_roc_auc'])
    return {
        'epoch': best.get('epoch'),
        'val_roc_auc_overlap_0.5': best.get('val_roc_auc'),
        'val_balanced_acc_overlap_0.5': best.get('val_balanced_acc'),
    }


def _latest_run(job_name):
    root = os.path.join(PIE_PATH, 'experiments', 'tensorflow', 'branchattn_temporal')
    prefix = 'scr_' + job_name + '_'
    found = []
    for path in glob.glob(os.path.join(root, prefix + '*')):
        if os.path.isdir(path):
            found.append(path)
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def _append_row(row):
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    if os.path.isfile(TABLE_PATH):
        with open(TABLE_PATH, encoding='utf-8') as fid:
            table = json.load(fid)
    else:
        table = {'rows': []}
    table['protocol'] = {
        'split': 'default video sets',
        'train_overlap': 0.5,
        'reported_agreement': 'validation, overlap 1, raw threshold 0.5',
        'callback': 'overlap 0.5 only, not the reported agreement',
        'architecture': 'branchattn_temporal + gated_add',
        'core': 'no extra residual. Same equation as pie_convlstm on bbox.',
        'balance': 'none',
        'loss': 'weighted BCE + focal, pos_weight from the train split',
        'monitor': 'val_auc',
        'aux_loss': 0,
        'clip': 'none',
        'not_comparable_to': 'the multi-branch checkpoint val AUC 0.7159',
    }
    table['rows'] = [old for old in table['rows']
                     if not (old.get('job') == row['job'] and old.get('seed') == row['seed'])]
    table['rows'].append(row)
    tmp = TABLE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fid:
        json.dump(table, fid, indent=2)
    os.replace(tmp, TABLE_PATH)


def _print_table():
    if not os.path.isfile(TABLE_PATH):
        print(f'[screen] no table yet: {TABLE_PATH}')
        return
    with open(TABLE_PATH, encoding='utf-8') as fid:
        table = json.load(fid)
    rows = table.get('rows', [])
    cores = {(row.get('seed'), row.get('batch')): row
             for row in rows if row.get('job') in ('core', 'core_b32')}
    print()
    print(f"{'job':<16} {'group':<12} {'seed':>4} {'batch':>5} {'val_auc':>8} {'val_bAcc':>8} {'d_auc':>8} {'d_bAcc':>8}  core_move")
    for row in rows:
        control_name = 'core_b32' if row.get('batch') == 32 else 'core'
        control = cores.get((row.get('seed'), row.get('batch')))
        if control is None:
            d_auc = d_bacc = None
        else:
            d_auc = None if row.get('val_auc') is None or control.get('val_auc') is None else row['val_auc'] - control['val_auc']
            d_bacc = None if row.get('val_bacc') is None or control.get('val_bacc') is None else row['val_bacc'] - control['val_bacc']
        if row.get('job') == 'flow_dense':
            control_name = 'core_b32'
        auc = '  n/a' if row.get('val_auc') is None else f"{row['val_auc']:.4f}"
        bacc = '  n/a' if row.get('val_bacc') is None else f"{row['val_bacc']:.4f}"
        da = '  n/a' if d_auc is None else f"{d_auc:+.4f}"
        db = '  n/a' if d_bacc is None else f"{d_bacc:+.4f}"
        move = row.get('core_move_last')
        move_s = '  n/a' if move is None else f"{move:.4f}"
        print(f"{row['job']:<16} {row['group']:<12} {row['seed']:>4} {row['batch']:>5} {auc:>8} {bacc:>8} {da:>8} {db:>8}  {move_s}  vs {control_name}")
    print()
    print('[screen] d_auc is this job minus the core of the same seed and batch.')
    print('[screen] val_auc is overlap 1. A row without overlap 1 is not in this table.')
    print('[screen] flow_dense is compared with core_b32. The batch-128 core is not its control.')
    print('[screen] annotation, near_label and leakage are not perception results.')


def _effect_summary(log_path):
    if not os.path.isfile(log_path):
        return {}
    rows = []
    with open(log_path, encoding='utf-8') as fid:
        for line in fid:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}
    probed = [row for row in rows if row.get('d_auc') is not None and row.get('open_auc') is not None]
    out = {'core_move_last': rows[-1].get('core_move'),
           'branch_move_last': rows[-1].get('branch_move')}
    if probed:
        best = max(probed, key=lambda row: row['open_auc'])
        out['probe_epoch'] = best['epoch']
        out['probe_open_auc_overlap_0.5'] = best['open_auc']
        out['probe_closed_auc_overlap_0.5'] = best.get('closed_auc')
        out['probe_d_auc_overlap_0.5'] = best.get('d_auc')
    return out


def run_job(job_name, train_seed):
    job = _job_or_die(job_name)
    hp = dict(HP)
    hp['seed'] = int(HP['seed'] if train_seed is None else train_seed)
    if 'batch' in job:
        hp['batch_size'] = int(job['batch'])
    os.environ['PIE_PATH'] = PIE_PATH
    print(f"[screen] job={job_name} group={job['group']} seed={hp['seed']} batch={hp['batch_size']}", flush=True)
    print(f"[screen] {job['what']}", flush=True)
    print('[screen] reported number will be val overlap 1 at threshold 0.5.', flush=True)
    print('[screen] test lines from the pipeline are not the result of this screen.', flush=True)

    builtins.input = _Answers(job_name, None if train_seed is None else hp['seed'])
    _install_decoder_picker(job_name)
    log_path = os.path.join(LOG_DIR, f"{job_name}_seed{hp['seed']}.jsonl")
    if os.path.isfile(log_path):
        os.remove(log_path)
    _install_effect_logger(log_path)

    from train_pipeline import run_training_pipeline
    started = time.time()
    result = run_training_pipeline(hp, DATA_OPTS, PIE_PATH)
    val = {}
    if isinstance(result, dict):
        metrics = (result.get('results') or {}).get('metrics') or {}
        if 'val' not in metrics and 'metrics' in result and isinstance(result['metrics'], dict):
            metrics = result['metrics']
        val = metrics.get('val') or {}
    run_dir = _latest_run(job_name)
    if val.get('roc_auc') is None and isinstance(result, dict):
        nested = result.get('results')
        if isinstance(nested, dict):
            val = (nested.get('metrics') or {}).get('val') or val
    row = {
        'job': job_name,
        'group': job['group'],
        'feature': job['feature'] if job['feature'] else job['flow'],
        'seed': hp['seed'],
        'batch': hp['batch_size'],
        'what': job['what'],
        'val_auc': None if val.get('roc_auc') is None else float(val['roc_auc']),
        'val_bacc': None if val.get('balanced_acc') is None else float(val['balanced_acc']),
        'overlap': 1,
        'threshold': 0.5,
        'run_dir': run_dir,
        'seconds': round(time.time() - started, 1),
    }
    row.update(_effect_summary(log_path))
    if run_dir:
        half = _best_overlap_half(run_dir)
        if half:
            row.update(half)
    if row['val_auc'] is None:
        print('[screen] overlap-1 val AUC was not returned. No row was written.', flush=True)
        print('[screen] The pipeline still ran. Read its log. Do not fill this table by hand from overlap 0.5.', flush=True)
        return
    _append_row(row)
    print(f"[screen] wrote {TABLE_PATH}", flush=True)
    _print_table()


def main(argv):
    if len(argv) < 2 or argv[1] in ('-h', '--help', 'jobs'):
        print('Usage: python core_plus_one.py <job> [train_seed]')
        print('       python core_plus_one.py table')
        print()
        for name, job in JOBS.items():
            print(f"  {name:<16} {job['group']:<12} {job['what']}")
        print()
        print('Run core, then core again with a second train seed, before any feature.')
        print('A feature effect smaller than the gap between those two cores is not a result.')
        return 0
    if argv[1] == 'table':
        _print_table()
        return 0
    seed = int(argv[2]) if len(argv) > 2 else None
    run_job(argv[1], seed)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))