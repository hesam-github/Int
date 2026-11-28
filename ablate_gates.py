"""Close one residual gate at a time on a saved gated_add checkpoint.

Copy this file to D:\\PIE codes\\PIE\\ablate_gates.py and run:

    cd /d "D:\\PIE codes\\PIE"
    python ablate_gates.py

Validation only. The test split is not scored.
Overlap 0.5 is the training window. It is not the overlap=1 table.

A drop means this checkpoint's fused score uses that residual.
It does not mean a model trained without the branch would score the same.
The core (VGG + bbox) has no gate. The last row closes every extra gate
and leaves the core alone.
"""
import gc
import hashlib
import json
import os
import pickle

import numpy as np

PIE_PATH = r'D:\PIE codes\PIE_dataset'
RUN_DIR = r'D:\PIE codes\PIE_dataset\experiments\tensorflow\branchattn_temporal\gated_add_BAttnT7_bin_defaultSplit'
OVERLAP = 0.5
BATCH = 8
os.environ['PIE_PATH'] = PIE_PATH


def _load_pickle(path):
    with open(path, 'rb') as fid:
        try:
            return pickle.load(fid)
        except Exception:
            fid.seek(0)
            return pickle.load(fid, encoding='bytes')


def _load_choices(run_dir):
    path = os.path.join(run_dir, 'choices.json')
    if not os.path.isfile(path):
        raise RuntimeError(
            'choices.json is missing. hyperparams.json does not store '
            'pose_min_height, split_seed, balance_mode or split_opts.')
    with open(path, encoding='utf-8') as fid:
        choices = json.load(fid)
    opts = choices.get('opts') or {}
    for key in ('pose_min_height', 'split_seed', 'data_seed', 'balance_mode', 'split_type'):
        if key not in opts:
            raise RuntimeError(f'choices.json opts has no {key}. Refusing to guess.')
    return opts


def _last_dim(shape):
    """Keras 2.10 may return (None, 15, 60) or [(None, 15, 60)]."""
    while isinstance(shape, (list, tuple)) and shape and not isinstance(shape[-1], (int, np.integer)):
        shape = shape[0]
    return int(shape[-1])


def _gate_names(model):
    names = []
    for layer in model.layers:
        name = layer.name
        if not name.startswith('gate_'):
            continue
        if name.startswith('gate_in_') or name == 'gate_stack':
            continue
        names.append(name[len('gate_'):])
    return names


def _force_closed(layer):
    weights = layer.get_weights()
    closed = [np.zeros_like(weights[0])]
    if len(weights) > 1:
        closed.append(np.full_like(weights[1], -30.0))
    layer.set_weights(closed)
    return weights


def _forward(keras_model, arrays, batch_size):
    """One fixed batch size. The last partial batch is zero-padded and discarded.

    A flat tuple of length 3 is read by Keras 2.10 predict as
    (inputs, labels, sample_weight), so only the encoder reaches the model.
    The extra outer tuple makes the whole triple the input.
    """
    import tensorflow as tf
    if batch_size < 2:
        raise RuntimeError('BATCH must be > 1. Batch 1 breaks ConvLSTM2D in this TensorFlow.')
    n = int(np.asarray(arrays[0]).shape[0])
    specs = tuple(
        tf.TensorSpec(
            shape=(batch_size,) + tuple(int(x) for x in np.asarray(a).shape[1:]),
            dtype=tf.float32)
        for a in arrays)
    steps = (n + batch_size - 1) // batch_size

    def gen():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = []
            for array in arrays:
                piece = np.array(array[start:stop], dtype=np.float32)
                short = batch_size - piece.shape[0]
                if short:
                    pad = np.zeros((short,) + piece.shape[1:], dtype=np.float32)
                    piece = np.concatenate([piece, pad], axis=0)
                batch.append(piece)
            yield (tuple(batch),)

    dataset = tf.data.Dataset.from_generator(gen, output_signature=(specs,))
    pred = keras_model.predict(dataset, steps=steps, verbose=0)
    if not isinstance(pred, (list, tuple)):
        pred = [pred]
    return [np.asarray(part, dtype=np.float64)[:n] for part in pred]


def _scores(y_true, y_prob):
    from train_helpers import compute_full_metrics
    metrics = compute_full_metrics(
        y_true, np.asarray(y_prob).reshape(-1), save_path=None, threshold=0.5)
    keep = ('roc_auc', 'balanced_acc', 'recall', 'specificity', 'accuracy', 'f1')
    return {key: float(metrics[key]) for key in keep}


def _build_val(opts, data_opts):
    """Same caches as training. Does not attach the 8-d flow summary."""
    from pie_data_Hesam import PIE, print_annotated_pedestrian_fields
    from train_helpers import ensure_split_seed
    from train_pipeline import (
        OFFLINE_CACHE_VERSION, attach_with_cache, build_split_sequences,
        apply_balancing_strategy, attach_bbox_branch)
    from extra_features import (
        attach_pose, attach_pose_dynamics, attach_pose_critical,
        attach_scene, attach_scene_critical, attach_light_color, attach_velocity)
    from augmentation import augment_pie_tracks

    data_opts = dict(data_opts)
    if opts['split_type'] != data_opts.get('data_split_type'):
        raise RuntimeError(
            f"split_type choices={opts['split_type']} but configs data_opts="
            f"{data_opts.get('data_split_type')}.")
    data_opts['data_split_type'] = opts['split_type']
    pose_scale = opts.get('pose_scale', 'bbox_relative')
    pose_h = int(opts['pose_min_height'])
    split_seed = int(opts['split_seed'])
    data_seed = int(opts['data_seed'])
    offline = set(opts.get('offline') or [])

    imdb = PIE(data_path=PIE_PATH)
    ensure_split_seed(imdb, data_opts['data_split_type'], data_opts, split_seed)
    print_annotated_pedestrian_fields(imdb)
    train, val, test = build_split_sequences(imdb, data_opts, carve_seed=split_seed)

    cache_dir = os.path.join(PIE_PATH, 'data_cache', 'offline_features')
    os.makedirs(cache_dir, exist_ok=True)
    split_tag = hashlib.md5(json.dumps(
        {'split_type': opts['split_type'],
         'split_seed': split_seed,
         'split_opts': opts.get('split_opts') or {},
         'data': {k: data_opts.get(k) for k in (
             'seq_type', 'min_track_size', 'fstride',
             'height_rng', 'squarify_ratio', 'subset')}},
        sort_keys=True, default=str).encode()).hexdigest()[:10]
    splits = {'train': train, 'val': val, 'test': test}

    def attach_pose_family(split, verbose):
        pose_dir = os.path.join(PIE_PATH, 'extra', 'pose')
        attach_pose(split, pose_dir, scale=pose_scale, min_bbox_height=pose_h, verbose=verbose)
        attach_pose_dynamics(split, pose_dir, min_bbox_height=pose_h, verbose=verbose)
        attach_pose_critical(split, pose_dir, min_bbox_height=pose_h, verbose=verbose)

    def attach_scene_family(split, verbose):
        scene_dir = os.path.join(PIE_PATH, 'extra', 'scene')
        attach_scene(split, scene_dir, verbose=verbose)
        attach_scene_critical(split, scene_dir, verbose=verbose)
        attach_light_color(split, scene_dir, verbose=verbose)

    if 'pose' in offline:
        pose_path = os.path.join(
            cache_dir, f'pose_{pose_scale}_h{pose_h}_crit1_{OFFLINE_CACHE_VERSION}_{split_tag}.pkl')
        if not os.path.isfile(pose_path):
            raise RuntimeError(f'pose cache missing, will not build a new one: {pose_path}')
        attach_with_cache('pose', splits, pose_path, attach_pose_family)
    if 'scene' in offline:
        scene_path = os.path.join(
            cache_dir, f'scene_light1_{OFFLINE_CACHE_VERSION}_{split_tag}.pkl')
        if not os.path.isfile(scene_path):
            raise RuntimeError(f'scene cache missing, will not build a new one: {scene_path}')
        attach_with_cache('scene', splits, scene_path, attach_scene_family)

    train, val = apply_balancing_strategy(
        imdb, train, val, opts['balance_mode'], augment_pie_tracks,
        val_augment_mode=(opts.get('val_augment_mode') or 'undersample_val'),
        data_seed=data_seed)
    attach_velocity(val)
    decoder = list(data_opts.get('decoder_input_type') or [])
    if 'bbox_branch' in decoder:
        attach_bbox_branch(val)
    return val


def _ped_ids(data, train_params, seq_length):
    from pie_data_Hesam import get_tracks
    _tracks, _images, _bboxes, ped_ids = get_tracks(
        data, train_params['data_type'], seq_length, OVERLAP)
    ids = []
    for ped in ped_ids:
        flat = np.asarray(ped, dtype=object).reshape(-1)
        ids.append(str(flat[0]))
    return np.asarray(ids)


def _bootstrap_delta(y_true, base, other, ped_ids, draws=200):
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(0)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    base = np.asarray(base).reshape(-1)
    other = np.asarray(other).reshape(-1)
    uniq = np.unique(ped_ids)
    deltas = []
    for _ in range(draws):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        index = np.concatenate([np.flatnonzero(ped_ids == ped) for ped in chosen])
        if len(np.unique(y_true[index])) < 2:
            continue
        deltas.append(float(
            roc_auc_score(y_true[index], base[index])
            - roc_auc_score(y_true[index], other[index])))
    if not deltas:
        return None, None, None
    arr = np.asarray(deltas, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _print_saved_val_auc(run_dir, fresh_auc):
    path = os.path.join(run_dir, 'epoch_metrics.json')
    if not os.path.isfile(path):
        print('[ablate] epoch_metrics.json missing. No training-val comparison.')
        return
    with open(path, encoding='utf-8') as fid:
        rows = json.load(fid)
    vals = [float(row['val_roc_auc']) for row in rows if isinstance(row.get('val_roc_auc'), (int, float))]
    if not vals:
        print('[ablate] epoch_metrics.json has no val_roc_auc.')
        return
    best = max(vals)
    print(f'[ablate] best training val_roc_auc={best:.4f}  this full pass={fresh_auc:.4f}  '
          f'diff={fresh_auc - best:+.4f}')
    print('[ablate] both are overlap 0.5. A gap of several points means these windows '
          'are not the training windows. Do not read the drops.')


def main():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    from keras.models import Model, load_model
    from feature_routing import restore_routing, CUSTOM_OBJECTS
    from temporal_encoders import CUSTOM_OBJECTS as TEMPORAL_OBJECTS
    from pie_intent import PIEIntent

    opts = _load_choices(RUN_DIR)
    custom_objects = dict(CUSTOM_OBJECTS)
    custom_objects.update(TEMPORAL_OBJECTS)
    configs = _load_pickle(os.path.join(RUN_DIR, 'configs.pkl'))
    with open(os.path.join(RUN_DIR, 'hyperparams.json'), encoding='utf-8') as fid:
        saved_hp = json.load(fid)
    train_params = configs[1]
    data_opts = configs[2] if len(configs) > 2 else {}

    intent = PIEIntent()
    intent.normalize_decoder_bbox = bool(saved_hp.get('normalize_decoder_bbox', False))
    restore_routing(intent, saved_hp)
    intent.load_model_config(configs[0])
    if getattr(intent, 'flow_root', None) in (None, ''):
        intent.flow_root = os.path.join(PIE_PATH, 'extra', 'flow')
    if getattr(intent, 'use_visual', True):
        intent._ensure_context_model()

    print(f'[ablate] split=val overlap={OVERLAP}. Test is not used.')
    val = _build_val(opts, data_opts)
    seq_length = int(getattr(intent, '_sequence_length', 15) or 15)
    arrays, y_true = intent.get_test_data(
        val, train_params, seq_length, OVERLAP, data_subset='val')
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    intent.context_model = None
    gc.collect()
    tf.keras.backend.clear_session()

    model = load_model(
        os.path.join(RUN_DIR, 'model.h5'), compile=False, custom_objects=custom_objects)
    input_names = [layer.name for layer in model.inputs]
    if 'flow_input' in input_names and getattr(intent, 'flow_representation', 'summary') != 'dense':
        print('[ablate] checkpoint has flow_input. Dense volume was required from get_test_data.')
    if 'flow_input' in input_names and len(arrays) != 3:
        raise RuntimeError('model expects flow_input but get_test_data returned no flow volume.')
    names = _gate_names(model)
    if not names:
        raise SystemExit('this checkpoint has no gate_* layers. It is not gated_add.')
    for name in names:
        layer = model.get_layer('gate_' + name)
        if not layer.get_weights():
            raise RuntimeError(f'gate_{name} has no weights. It is not a Dense gate.')

    print('[ablate] gates:', ', '.join(names))
    print(f'[ablate] windows={len(y_true)} positive_rate={float(y_true.mean()):.4f}')
    expected = _last_dim(model.get_layer('decoder_input').output_shape)
    got = int(np.asarray(arrays[1]).shape[-1])
    if got != expected:
        raise SystemExit(
            f'[ablate] decoder width {got} != model {expected}. '
            'Pose/scene cache did not match this run. Stop.')
    ped_ids = _ped_ids(val, train_params, seq_length)
    if len(ped_ids) != len(y_true):
        print(f'[ablate] ped_id count {len(ped_ids)} != windows {len(y_true)}. Bootstrap off.')
        ped_ids = None

    probe = Model(
        inputs=model.inputs,
        outputs=[model.output] + [model.get_layer('gate_' + name).output for name in names])
    packed = _forward(probe, arrays, BATCH)
    base_prob = packed[0].reshape(-1)
    print('[ablate] gate means on the open model, val overlap 0.5:')
    for name, gate in zip(names, packed[1:]):
        values = np.asarray(gate, dtype=np.float64).reshape(-1)
        print(f'  {name:16}  mean={float(values.mean()):.4f}  '
              f'p10={float(np.percentile(values, 10)):.4f}  '
              f'p50={float(np.percentile(values, 50)):.4f}')
    base = _scores(y_true, base_prob)
    print('[ablate] full  auc={roc_auc:.4f}  bAcc={balanced_acc:.4f}  '
          'recall={recall:.4f}  spec={specificity:.4f}'.format(**base))
    _print_saved_val_auc(RUN_DIR, base['roc_auc'])

    rows = [{'branch': 'full', **base, 'delta_auc': 0.0, 'delta_bacc': 0.0,
             'auc_drop_mean': 0.0, 'auc_drop_lo': 0.0, 'auc_drop_hi': 0.0}]

    def run_closed(label, closed_names):
        backups = {}
        try:
            for name in closed_names:
                backups[name] = _force_closed(model.get_layer('gate_' + name))
            prob = _forward(model, arrays, BATCH)[0].reshape(-1)
        finally:
            for name, weights in backups.items():
                model.get_layer('gate_' + name).set_weights(weights)
        score = _scores(y_true, prob)
        drop_auc = base['roc_auc'] - score['roc_auc']
        drop_bacc = base['balanced_acc'] - score['balanced_acc']
        mean = lo = hi = None
        if ped_ids is not None:
            mean, lo, hi = _bootstrap_delta(y_true, base_prob, prob, ped_ids)
        print(f'[ablate] off {label:16}  auc={score["roc_auc"]:.4f}  '
              f'd_auc={drop_auc:+.4f}  bAcc={score["balanced_acc"]:.4f}  '
              f'd_bacc={drop_bacc:+.4f}  recall={score["recall"]:.4f}  '
              f'spec={score["specificity"]:.4f}')
        if mean is not None:
            print(f'         ped bootstrap d_auc mean={mean:.4f}  95%=[{lo:.4f}, {hi:.4f}]')
        rows.append({
            'branch': label, **score,
            'delta_auc': drop_auc, 'delta_bacc': drop_bacc,
            'auc_drop_mean': mean, 'auc_drop_lo': lo, 'auc_drop_hi': hi,
        })

    for name in names:
        run_closed(name, [name])
    run_closed('all_extras', names)

    out_path = os.path.join(RUN_DIR, 'gate_ablation_val.json')
    with open(out_path, 'w', encoding='utf-8') as fid:
        json.dump({
            'run_dir': RUN_DIR,
            'split': 'val',
            'overlap': OVERLAP,
            'threshold': 0.5,
            'batch': BATCH,
            'n': int(len(y_true)),
            'rows': rows,
            'note': 'Gate forced near 0 at inference. Not a retrained ablation. Test set not used.',
        }, fid, indent=2)
    print(f'[ablate] wrote {out_path}')
    ranked = sorted(
        (row for row in rows if row['branch'] not in ('full', 'all_extras')),
        key=lambda row: row['delta_auc'], reverse=True)
    print('[ablate] largest val AUC drop when that gate is closed:')
    for row in ranked:
        print(f'  {row["branch"]:16}  d_auc={row["delta_auc"]:+.4f}  d_bacc={row["delta_bacc"]:+.4f}')


if __name__ == '__main__':
    main()