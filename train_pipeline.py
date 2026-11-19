"""
train_pipeline.py

The body of train_pie_intent_tensorflow_gpu.py, moved out of the notebook.
The notebook keeps only HP and DATA_OPTS, then calls run_training_pipeline.

Nothing in the training order was changed, except:
  - stdout/stderr are restored after the run, and a second cell execution
    does not wrap Tee inside Tee
  - DATA_OPTS is copied, so the notebook dict is not mutated
  - batch_size is an explicit HP key
  - the overlap 0.5 pass passes feature_subset=<split name> to test_chunk
  - velocity is always attached; pose/scene/flow only if selected in the offline menu
  - use_visual=False skips VGG16 entirely (decoder-only model)
"""

import copy
import json
import os
import pickle
import sys
import time

from focal_loss import weighted_binary_crossentropy
from log_mode import configure_tf_logging, is_full
from capacity_budget import interactive_pick_capacity, capacity_tag
from train_helpers import (
    ensure_split_seed,
    interactive_pick_seeds,
    Tee,
    PHASE_BALANCING,
    PHASE_BUILD_SEQUENCES,
    PHASE_CALIBRATION,
    PHASE_FULL_EVAL,
    PHASE_LOSS_FINALIZE,
    PHASE_RECOMPUTE_CALIBRATED,
    PHASE_TRAINING,
    apply_balancing_strategy,
    build_calibration_dict,
    build_data_split_tag,
    build_run_hyperparam_tag,
    build_short_run_tag,
    write_run_summary,
    interactive_pick_normalize_bbox,
    append_model_summary,
    build_split_sequences,
    compute_pos_weight_from_train,
    evaluate_split_full_metrics,
    find_best_threshold_by_f1,
    interactive_pick_pose_min_height,
    interactive_pick_use_visual,
    interactive_pick_offline_features,
    interactive_pick_balance_mode,
    interactive_pick_clip_mode,
    interactive_pick_data_split,
    interactive_pick_decoder_features,
    interactive_pick_eval_every,
    interactive_pick_feature_norms,
    save_choices_json,
    interactive_pick_log_mode,
    interactive_pick_loss_choice,
    interactive_pick_lstm_recurrent_activation,
    interactive_pick_monitor_metric,
    interactive_pick_new_run_name,
    interactive_pick_pose_scale,
    interactive_pick_extra_norm,
    interactive_pick_visual_norm,
    interactive_pick_feature_target,
    interactive_pick_fusion,
    interactive_pick_zero_decoder_bbox,
    interactive_pick_val_augment_mode,
    interactive_pick_weight_decay,
    interactive_pick_yes_no,
    load_ped_intents_pkl,
    recompute_split_metrics_at_threshold,
    print_best_epoch_summary,
    print_full_history_table,
    print_phase,
    print_summary_table,
    recompute_split_metrics_prior_corrected,
    recompute_split_metrics_custom,
    resolve_balance_mode_preset,
)
from visual_norm import confirm_visual_pipeline, visual_experiment_code
_CONSOLE = {}


def start_console_log(log_path):
    """Copy stdout/stderr into log_path. A previous redirect is undone first."""
    stop_console_log()
    fid = open(log_path, 'w', encoding='utf-8')
    _CONSOLE.update(stdout=sys.stdout, stderr=sys.stderr, file=fid)
    sys.stdout = Tee(sys.stdout, fid)
    sys.stderr = Tee(sys.stderr, fid)


def stop_console_log():
    if not _CONSOLE:
        return
    sys.stdout, sys.stderr = _CONSOLE['stdout'], _CONSOLE['stderr']
    _CONSOLE['file'].close()
    _CONSOLE.clear()


def pick_run_options(hp):
    """Menus that do not need the tracks. Same order as the old script."""
    opts = {}
    opts['offline'] = interactive_pick_offline_features()
    print(f"[config] OFFLINE_FEATURES = {opts['offline']}")

    opts['use_visual'] = interactive_pick_use_visual()
    print(f"[config] USE_VISUAL = {opts['use_visual']}")

    opts['zero_decoder_bbox'] = interactive_pick_zero_decoder_bbox(opts['use_visual'])
    print(f"[config] ZERO_DECODER_BBOX = {opts['zero_decoder_bbox']}")

    while True:
        opts['visual_norm'] = interactive_pick_visual_norm() if opts['use_visual'] else False
        print(f"[config] VISUAL_NORM = {opts['visual_norm']}")
        if opts['use_visual']:
            raw = input("[visual_percentile] use positive-percentile scaling (q=99, no center/clip)? [y/N]: ").strip().lower()
            opts['use_percentile'] = raw in ('y', 'yes')
        else:
            opts['use_percentile'] = False
        print(f"[config] USE_PERCENTILE = {opts['use_percentile']}")
        opts['capacity'] = interactive_pick_capacity() if opts['use_visual'] else None
        print(f"[config] CAPACITY = {opts['capacity']}")

        if not opts['use_visual'] or confirm_visual_pipeline(opts['visual_norm'], opts['capacity']):
            break
        print("[visual_pipeline] re-asking visual_norm and capacity.")

    opts['log_mode'] = interactive_pick_log_mode()
    print(f"[config] LOG_MODE = {opts['log_mode']}")

    opts['balance_mode'] = interactive_pick_balance_mode()
    print(f"[config] BALANCE_MODE = {opts['balance_mode']}")
    opts['val_augment_mode'] = (interactive_pick_val_augment_mode()
                                if opts['balance_mode'] == 'augmentation' else None)
    if opts['val_augment_mode']:
        print(f"[config] VAL_AUGMENT_MODE = {opts['val_augment_mode']}")

    opts['monitor_metric'], opts['monitor_mode'] = interactive_pick_monitor_metric()
    print(f"[config] MONITOR_METRIC = {opts['monitor_metric']} (mode={opts['monitor_mode']})")

    opts['loss_name'], opts['loss_fn'] = interactive_pick_loss_choice(
        balance_mode=opts['balance_mode'])
    print(f"[config] LOSS_FUNCTION (family) = {opts['loss_name']}")

    if opts['balance_mode'] == 'undersample_base_article':
        clip_type, clip_amount = 'clipvalue', hp['clipvalue']
    else:
        clip_type, clip_amount = interactive_pick_clip_mode(default_amount=hp['clipvalue'])
        if clip_type != 'none' and clip_amount is None:
            print("[clip] no clip amount given -> gradient clipping DISABLED")
            clip_type = 'none'

    opts['weight_decay'] = interactive_pick_weight_decay(default_amount=0.0)
    print(f"[config] WEIGHT_DECAY = {opts['weight_decay']}")

    opts['rec_act'] = interactive_pick_lstm_recurrent_activation(default='sigmoid')
    normalize_bbox = interactive_pick_normalize_bbox(default=True)
    print(f"[config] normalize_decoder_bbox = {normalize_bbox}")
    opts['use_plateau'] = interactive_pick_yes_no(
        "[plateau] use ReduceLROnPlateau?", default=True)
    print(f"[config] LSTM_REC_ACT={opts['rec_act']} NORM_BBOX={normalize_bbox} "
          f"USE_PLATEAU={opts['use_plateau']}")

    opts['eval_every'] = interactive_pick_eval_every()
    print(f"[config] EVAL_EVERY = {opts['eval_every'] if opts['eval_every'] is not None else 'never'}")

    opts['pose_scale'] = (interactive_pick_pose_scale()
                          if 'pose' in opts['offline'] else 'bbox_relative')
    opts['pose_min_height'] = (interactive_pick_pose_min_height()
                               if 'pose' in opts['offline'] else 0)
    print(f"[config] POSE_SCALE = {opts['pose_scale']}  POSE_MIN_HEIGHT = {opts['pose_min_height']}")
    opts['use_tte'] = interactive_pick_yes_no(
        "[tte] attach time-to-event? (annotation-derived future information: ablation only)",
        default=False)
    print(f"[config] USE_TTE = {opts['use_tte']}")

    (opts['clip_type'], opts['clip_amount'], opts['batch_size'],
     opts['normalize_bbox']) = resolve_balance_mode_preset(
        opts['balance_mode'], clip_type, clip_amount, hp['lr'],
        batch_size=hp['batch_size'],
        normalize_decoder_bbox=normalize_bbox)
    # Printed AFTER the resolve: these are the values training will actually use.
    print(f"[config] CLIP_TYPE = {opts['clip_type']}  CLIP_AMOUNT = {opts['clip_amount']}")
    print(f"[config] BATCH_SIZE = {opts['batch_size']}  (hp requested {hp['batch_size']})")

    opts['split_type'], opts['split_opts'], _ = interactive_pick_data_split()
    print(f"[config] data_split_type = {opts['split_type']}")

    (opts['train_seed'], opts['shuffle_seed'],
     opts['data_seed'], opts['split_seed']) = interactive_pick_seeds(
        opts['split_type'], opts['balance_mode'], hp['seed'],
        hp.get('data_seed', 42), hp.get('split_seed', 42))
    print(f"[config] TRAIN_SEED={opts['train_seed']} SHUFFLE_SEED={opts['shuffle_seed']} "
          f"DATA_SEED={opts['data_seed']} SPLIT_SEED={opts['split_seed']}")
    return opts

def _require_dir(path, family):
    """A family chosen in the menu must exist: attach_pose/attach_scene would silently write zeros."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"[offline] '{family}' was selected but this folder does not exist: {path}\n"
            f"Run its extractor, or answer the offline-features question without '{family}'.")


def prepare_data(opts, data_opts, pie_path):
    """Sequences, optional offline features, balancing, velocity, and the final loss object."""
    from pie_data_Hesam import PIE, print_annotated_pedestrian_fields
    from augmentation import augment_pie_tracks

    data_opts = copy.deepcopy(data_opts)
    data_opts['data_split_type'] = opts['split_type']
    data_opts.update(opts['split_opts'])

    offline = set(opts.get('offline', []))     # subset of {'pose', 'scene', 'flow'}
    pose_dir = os.path.join(pie_path, 'extra', 'pose')
    scene_dir = os.path.join(pie_path, 'extra', 'scene')
    flow_dir = os.path.join(pie_path, 'extra', 'flow')

    print_phase(PHASE_BUILD_SEQUENCES)
    imdb = PIE(data_path=pie_path)
    ensure_split_seed(imdb, data_opts['data_split_type'], data_opts, opts['split_seed'])
    print_annotated_pedestrian_fields(imdb)
    train, val, test = build_split_sequences(imdb, data_opts, carve_seed=opts['split_seed'])

    # Offline features: only read from disk when selected in the first menu.
    if 'pose' in offline:
        _require_dir(pose_dir, 'pose')
        from extra_features import attach_pose
        for split in (train, val, test):
            attach_pose(split, pose_dir, scale=opts.get('pose_scale', 'bbox_relative'),
                        min_bbox_height=opts.get('pose_min_height', 0))
                        
    if 'scene' in offline:
        _require_dir(scene_dir, 'scene')
        from extra_features import attach_scene, attach_scene_critical
        for split in (train, val, test):
            attach_scene(split, scene_dir)
            attach_scene_critical(split, scene_dir)   # new targeted features

    lens = [len(track) for track in train['image']]
    print(f"tracks={len(lens)}  min_len={min(lens)}  max_len={max(lens)}")
    min_ts = data_opts['min_track_size']
    print(f"tracks shorter than {min_ts}:", sum(length < min_ts for length in lens))

    print_phase(PHASE_BALANCING)
    train, val = apply_balancing_strategy(
        imdb, train, val, opts['balance_mode'], augment_pie_tracks,
        val_augment_mode=(opts['val_augment_mode'] or 'undersample_val'),
        data_seed=opts['data_seed'])

    if opts['balance_mode'] == 'augmentation' and (offline & {'pose', 'scene'}):
        print("[warning] augmentation mode: pose/scene of synthetic copies are only re-indexed, "
              "not mirrored/jittered together with the bbox -- do not interpret pose/scene "
              "results obtained in this mode.")

    # velocity is computed from bbox only (no disk reads), so it is always attached
    from extra_features import attach_velocity, attach_tte, attach_flow,attach_flow_relative
    for split in (train, val, test):
        attach_velocity(split)
    if opts.get('use_tte'):
        for split in (train, val, test):
            attach_tte(split, verbose=(split is train))

    if 'flow' not in offline:
        print("[flow] not selected: nothing loaded")
    elif opts['balance_mode'] == 'augmentation':
        print("[warning] flow was selected but is SKIPPED in augmentation mode: "
              "no flow_* feature will appear in the menu.")
    else:
        _require_dir(flow_dir, 'flow')
        for split in (train, val, test):
            attach_flow(split, flow_dir, verbose=(split is train))
            attach_flow_relative(train)
            attach_flow_relative(val)
            attach_flow_relative(test)

    print_phase(PHASE_LOSS_FINALIZE)
    from focal_loss import bce_plus_focal_loss

    if opts['loss_name'] == 'weighted_bce':
        pos_weight = compute_pos_weight_from_train(train)
        loss_name = f"weighted_bce_pw{pos_weight:.4f}"
        loss_fn = weighted_binary_crossentropy(pos_weight=pos_weight)
    elif (isinstance(opts.get('loss_fn'), tuple)
          and len(opts['loss_fn']) == 3
          and opts['loss_fn'][0] == 'wbce_focal_auto'):
        _, gamma, focal_weight = opts['loss_fn']
        pos_weight = compute_pos_weight_from_train(train)
        loss_fn = bce_plus_focal_loss(
            gamma=gamma,
            focal_weight=focal_weight,
            pos_weight=pos_weight)
        loss_name = loss_fn.__name__
    else:
        pos_weight = None
        loss_name = opts['loss_name']
        loss_fn = opts['loss_fn']
    print(f"[config] LOSS_FUNCTION (final) = {loss_name}")
    return dict(imdb=imdb, train=train, val=val, test=test, data_opts=data_opts,
                loss_name=loss_name, loss_fn=loss_fn, pos_weight=pos_weight)

def attach_bbox_branch(*splits):
    """Adds beh_seq['bbox_branch']: a copy of the (already augmented) raw bbox."""
    for split in splits:
        split['bbox_branch'] = [[[float(v) for v in box] for box in track]
                                for track in split['bbox']]
    print(f"[bbox_branch] attached to {len(splits)} split(s)")
def choose_run_identity(opts, data, hp, pie_path):
    """Decoder features, per-feature scaling, architecture and run name."""
    experiments_root = os.path.join(pie_path, 'experiments')
    use_visual = opts['use_visual']
    split_tag = build_data_split_tag(opts['split_type'], data['data_opts'])
    print(f"[config] DATA_SPLIT_TAG = {split_tag}")

    decoder_features, feature_stds = interactive_pick_decoder_features(
        data['train'], use_visual=use_visual)
    data['data_opts']['decoder_input_type'] = decoder_features
    if 'bbox_branch' in decoder_features:
        attach_bbox_branch(data['train'], data['val'], data['test'])

    # Global answer: default for every ordinary extra (the old question is kept).
    opts['extra_norm'] = interactive_pick_extra_norm()
    opts['feature_norm'] = interactive_pick_feature_norms(
        decoder_features, opts['extra_norm'], feature_stds=feature_stds)

    from architecture_registry import (
        interactive_pick_architecture, get_architecture_config,
        interactive_pick_branch_mode, interactive_pick_aux_loss,
        interactive_pick_fusion_mode,
    )
    # The architecture menu is only meaningful when the VGG16 context is used.
    arch_name = interactive_pick_architecture() if use_visual else 'pie_convlstm'
    arch_cfg = get_architecture_config(arch_name)

    opts['branch_mode'] = None
    if arch_name in ('branchattn', 'branchattn_temporal'):
        opts['branch_mode'] = interactive_pick_branch_mode()
        print(f"[config] BRANCH_MODE = {opts['branch_mode']}")

    opts['feature_target'] = arch_cfg['feature_target']
    opts['fusion'] = arch_cfg['fusion']
    opts['zero_visual'] = arch_cfg.get('zero_visual', False)

    print(f"[config] ARCHITECTURE  = {arch_name}")
    print(f"[config] FUSION        = {opts['fusion']}")
    print(f"[config] FEATURE_TARGET= {opts['feature_target']}")
    print(f"[config] ZERO_VISUAL   = {opts['zero_visual']}")
    print(f"[config] EXTRA_NORM    = {opts['extra_norm']}")
    print(f"[config] FEATURE_NORM  = {opts['feature_norm']}")

    if use_visual and 'bbox_branch' in decoder_features and opts['fusion'] == 'pie':
        print("[warning] bbox_branch + fusion=pie: the copy is only concatenated into the "
              "decoder input. For a PARALLEL bbox branch choose branchattn or branchattn_temporal.")

    # Temporal encoder menu (only for temporal_encoders architecture)
    if arch_cfg.get('use_temporal_menu', False):
        from temporal_encoders import choose_temporal_for_menu
        choose_temporal_for_menu(opts)
    else:
        opts['temporal_encoder'] = None
        opts['temporal_hp'] = {}

    # Interactive fusion mode when offline features are present
    has_offline = any(f not in ('bbox',) for f in decoder_features)
    if arch_name in ('temporal_encoders', 'branchattn', 'branchattn_temporal', 'pcpa') and has_offline:
        opts['fusion_mode'] = interactive_pick_fusion_mode(has_offline)
        print(f"[config] FUSION_MODE = {opts['fusion_mode']}")
    else:
        opts['fusion_mode'] = 'pie_concat'

    # Auxiliary loss (only meaningful for multi-branch modes)
    opts['aux_loss_weight'] = 0.0
    multi_branch = (
        (arch_name in ('branchattn', 'branchattn_temporal')
         and opts.get('branch_mode') in ('static_pool', 'core_plus_extras'))
        or opts.get('fusion_mode') == 'core_plus_extras'
    )
    if multi_branch:
        opts['aux_loss_weight'] = interactive_pick_aux_loss()
        print(f"[config] AUX_LOSS_WEIGHT = {opts['aux_loss_weight']}")

    print(f"[config] TEMPORAL_ENCODER = {opts.get('temporal_encoder')}")
    print(f"[config] TEMPORAL_HP      = {opts.get('temporal_hp')}")
    print(f"[config] USE_TTE  = {opts.get('use_tte')}")
    print(f"[config] FLOW_FEATURES   = {[f for f in decoder_features if f.startswith('flow_')]}")

    full_tag = build_run_hyperparam_tag(
        hp['reg'], opts['clip_type'], opts['clip_amount'], opts['balance_mode'],
        hp['lr'], hp['lstm_dropout'], hp['lstm_recurrent_dropout'],
        data['loss_name'], opts['monitor_metric'], split_tag,
        weight_decay=opts['weight_decay'],
        early_stop_min_delta=hp['es_min_delta'],
        early_stop_patience=hp['es_patience'],
        seed=opts['train_seed'], rec_activation=opts['rec_act'],
        normalize_decoder_bbox=opts['normalize_bbox'],
        use_plateau=opts['use_plateau'], decoder_features=decoder_features,
        feature_target=opts['feature_target'], fusion=opts['fusion'],
        extra_norm=opts['extra_norm'], pose_scale=opts.get('pose_scale', 'bbox_relative'),
        eval_every=opts['eval_every'], val_augment_mode=opts['val_augment_mode'],
        pose_min_height=opts.get('pose_min_height', 0))

    short_tag = build_short_run_tag(
        opts['fusion'], opts['feature_target'], decoder_features,
        opts['split_type'], split_tag, temporal_encoder=opts.get('temporal_encoder'))
    if not use_visual:
        short_tag = 'NoVis_' + short_tag.replace('_context', '', 1)

    if opts.get('zero_decoder_bbox'):
        short_tag = 'ZeroBbox_' + short_tag

    ctag = capacity_tag(opts.get('capacity'))
    if ctag:
        short_tag = ctag + short_tag

    if opts.get('visual_norm'):
        log_first = bool((opts.get('capacity') or {}).get('log1p'))
        short_tag = ('VNormLog_' if log_first else 'VNorm_') + short_tag

    architecture = arch_name
    run_name = interactive_pick_new_run_name(
        os.path.join(experiments_root, hp['framework'], architecture),
        hyperparam_tag=short_tag)

    print(f"\n[config] ARCHITECTURE = {architecture}")
    print(f"[config] RUN_NAME     = {run_name}")
    return dict(architecture=architecture, run_name=run_name,
                decoder_features=decoder_features, full_tag=full_tag)

def _print_section(title, items):
    print(f"\n--- {title} ---")
    for key, value in items:
        print(f"{key:<22}-> {value}")
    return title, items


def setup_gpu(tf):
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs visible to TF:", gpus)
    assert len(gpus) > 0, (
        "No GPU visible to TensorFlow -- check driver/CUDA install, or run "
        "the CPU variant instead.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"[warning] could not set memory growth on {gpu}: {exc}")
    print("Running with GPU enabled.")


def start_run(opts, hp, data, ident, pie_path):
    """TensorFlow init, run folder, console log, model object."""
    import tensorflow as tf
    if hp['deterministic_ops']:
        tf.config.experimental.enable_op_determinism()
        print("[seed] TF op determinism ENABLED")
    configure_tf_logging()

    from experiment_paths import get_experiment_paths
    paths = get_experiment_paths(
        pie_path=pie_path, framework=hp['framework'],
        architecture=ident['architecture'], run_name=ident['run_name'])

    log_path = os.path.join(
        paths['log_dir'], f"console_{time.strftime('%d%b%Y-%Hh%Mm%Ss')}.txt")
    start_console_log(log_path)
    print(f"Run dir      -> {paths['run_dir']}")
    print(f"Console log  -> {log_path}")
    _choices = {
        'opts': {k: v for k, v in opts.items() if k != 'loss_fn'},
        'hp': hp,
        'loss_name': data['loss_name'],
        'architecture': ident['architecture'],
        'run_name': ident['run_name'],
        'decoder_features': ident['decoder_features'],
        'split_opts': {k: data['data_opts'].get(k)
                       for k in ('random_params', 'kfold_params')},
    }
    save_choices_json(paths['run_dir'], _choices)

    sec_arch = _print_section('architecture hyperparams', [
        ('hidden_units', hp['hidden_units']),
        ('convlstm_filters', hp['convlstm_filters']),
        ('convlstm_kernel', hp['convlstm_kernel']),
        ('regularizer_val', hp['reg']),
        ('lstm_dropout', hp['lstm_dropout']),
        ('lstm_rec_dropout', hp['lstm_recurrent_dropout']),
        ('lstm_rec_activation', opts['rec_act']),
        ('branch_mode', opts.get('branch_mode')),
        ('aux_loss_weight', opts.get('aux_loss_weight', 0.0))])
    sec_opt = _print_section('optimizer/training hyperparams', [
        ('lr', hp['lr']),
        ('clip_type', opts['clip_type']),
        ('clip_amount', opts['clip_amount']),
        ('weight_decay', opts['weight_decay']),
        ('batch_size', opts['batch_size']),
        ('es_min_delta', hp['es_min_delta']),
        ('es_patience', hp['es_patience']),
        ('use_plateau', opts['use_plateau']),
        ('monitor_metric', f"{opts['monitor_metric']} (mode={opts['monitor_mode']})"),
        ('loss', data['loss_name'])])
    sec_data = _print_section('data hyperparams', [
        ('data_split_type', opts['split_type']),
        ('balance_mode', opts['balance_mode']),
        ('val_augment_mode', opts['val_augment_mode']),
        ('normalize_decoder_bbox', opts['normalize_bbox']),
        ('pose_scale', opts.get('pose_scale')),
        ('feature_target', opts.get('feature_target')),
        ('fusion', opts.get('fusion')),
        ('extra_norm', opts.get('extra_norm')),
        ('feature_norm', opts.get('feature_norm')),
        ('zero_visual', opts.get('zero_visual', False))])

    write_run_summary(paths['run_dir'], ident['full_tag'], [
        ('run identity', [
            ('architecture', ident['architecture']),
            ('run_name (folder)', paths['run_folder_name']),
            ('train_seed', opts.get('train_seed')),
            ('shuffle_seed', opts.get('shuffle_seed')),
            ('data_seed', opts.get('data_seed')),
            ('split_seed', opts.get('split_seed')),
            ('use_visual', opts.get('use_visual')),
            ('visual_norm', opts.get('visual_norm')),
            ('visual_pipeline', visual_experiment_code(
            opts.get('visual_norm'), (opts.get('capacity') or {}).get('log1p'))),
            ('decoder_features', ident['decoder_features']),
            ('temporal_encoder', opts.get('temporal_encoder')),
            ('temporal_hp', opts.get('temporal_hp')),
            ('use_tte', opts.get('use_tte')),
            ('log_mode', opts.get('log_mode')),
            ('eval_every', opts.get('eval_every')),
            ('pose_min_height', opts.get('pose_min_height')),
            ('split_opts', {k: data['data_opts'].get(k)
                            for k in ('random_params', 'kfold_params')}),
        ]),
        sec_arch, sec_opt, sec_data,
    ])

    print()
    setup_gpu(tf)

    from pie_intent_experiment import PIEIntentExperiment

    model = PIEIntentExperiment(
        framework=hp['framework'], architecture=ident['architecture'],
        run_name=ident['run_name'], run_dir=paths['run_dir'],
        num_hidden_units=hp['hidden_units'], regularizer_val=hp['reg'],
        lstm_dropout=hp['lstm_dropout'],
        lstm_recurrent_dropout=hp['lstm_recurrent_dropout'],
        convlstm_num_filters=hp['convlstm_filters'],
        convlstm_kernel_size=hp['convlstm_kernel'],
        lstm_recurrent_activation=opts['rec_act'])
    model.temporal_encoder = opts.get('temporal_encoder')
    model.temporal_hp = opts.get('temporal_hp', {})
    print(f"[config] temporal_encoder = {model.temporal_encoder}")

    model.normalize_decoder_bbox = opts['normalize_bbox']
    model.feature_target = opts.get('feature_target', 'decoder')
    model.fusion = opts.get('fusion', 'pie')
    model.branch_mode = opts.get('branch_mode')  # None for non-branch architectures
    model.aux_loss_weight = float(opts.get('aux_loss_weight', 0.0) or 0.0)
    model.temporal_fusion = opts.get('temporal_fusion', 'pie_concat')
    model.extra_norm = opts.get('extra_norm', 'none')
    model.pose_scale = opts.get('pose_scale', 'bbox_relative')
    model.zero_visual = opts.get('zero_visual', False)
    model.feature_norm = opts.get('feature_norm') or {}
    model.zero_decoder_bbox = opts.get('zero_decoder_bbox', False)
    model.use_visual = opts.get('use_visual', True)
    model.capacity = opts.get('capacity')
    cap = opts.get('capacity') or {}
    if cap.get('hidden_units') and int(cap['hidden_units']) != model._num_hidden_units:
        model._num_hidden_units = int(cap['hidden_units'])
        print(f"[capacity] decoder hidden units overridden -> {model._num_hidden_units}")
    model.visual_norm_enabled = bool(opts.get('visual_norm', False))
    model.use_percentile = bool(opts.get('use_percentile', False))
    model.percentile_scale = None
    model.visual_norm = None
    model.extra_norm_stats = None
    print(f"[config] model.normalize_decoder_bbox = {model.normalize_decoder_bbox}")
    print(f"[config] branch_mode={model.branch_mode} temporal_fusion={model.temporal_fusion}")
    print(f"[config] routing fusion={model.fusion} target={model.feature_target} "
          f"extra_norm={model.extra_norm} pose_scale={model.pose_scale} "
          f"zero_visual={model.zero_visual} use_visual={model.use_visual}")
    print(model)
    return model, paths

def train_and_report(model, opts, hp, data, paths):
    import tensorflow as tf
    d_opts = data['data_opts']
    print(f"Epochs for this run -> {hp['n_epochs']}")
    print_phase(PHASE_TRAINING)
    print("min_track_size =", d_opts['min_track_size'])
    print("seq_overlap_rate =", d_opts['seq_overlap_rate'])

    saved = model.train(
        data_train=data['train'], data_val=data['val'],
        data_test=(data['test'] if opts['eval_every'] is not None else None),
        epochs=hp['n_epochs'], loss=[data['loss_fn']], metrics=['accuracy'],
        batch_size=opts['batch_size'], optimizer_type='rmsprop',
        optimizer_params={
            'lr': hp['lr'],
            'clip_type': opts['clip_type'],
            'clip_amount': opts['clip_amount'],
            'weight_decay': opts['weight_decay'],
        },
        data_opts=d_opts,
        monitor_metric=opts['monitor_metric'],
        monitor_mode=opts['monitor_mode'],
        early_stop_min_delta=hp['es_min_delta'],
        early_stop_patience=hp['es_patience'],
        seed=opts['train_seed'],
        shuffle_seed=opts['shuffle_seed'],
        use_plateau=opts['use_plateau'],
        eval_every=opts['eval_every'],
        plateau_patience=hp.get('plateau_patience', 3),
        plateau_factor=hp.get('plateau_factor', 0.5),
        plateau_min_lr=hp.get('plateau_min_lr', 1e-7),
        plateau_min_delta=hp.get('plateau_min_delta', 1e-4))

    append_model_summary(paths['run_dir'], model.train_model)

    print("encoder seq length:", model._encoder_seq_length)
    print("L2 reg term now:", float(tf.add_n(model.train_model.losses)))
    print("decoder seq length:", model._decoder_seq_length)
    print("model input shapes:", [i.shape for i in model.train_model.inputs])
    print(f"Model + configs saved to: {saved}")
    assert os.path.abspath(saved) == os.path.abspath(paths['run_dir']), (
        "saved_files_path diverged from the intended run_dir -- check "
        "PIEIntentExperiment.get_path.")

    history_path = paths.get('history_path', os.path.join(paths['run_dir'], 'history.pkl'))
    if not os.path.exists(history_path):
        history_path = os.path.join(saved, 'history.pkl')
    history = None
    if os.path.exists(history_path):
        with open(history_path, 'rb') as fid:
            history = pickle.load(fid)
        print(f"[note] per-epoch metrics below use seq_overlap_rate="
              f"{d_opts['seq_overlap_rate']} (training overlap) -- "
              f"NOT comparable to the overlap=1 final evaluation numbers below.")
        print_full_history_table(history)
        if is_full():
            print_best_epoch_summary(
                os.path.join(saved, 'epoch_metrics.json'),
                monitor_metric=opts['monitor_metric'],
                monitor_mode=opts['monitor_mode'])
    else:
        print(f"[warning] could not find history.pkl -- checked: "
              f"{paths.get('history_path')} and {os.path.join(saved, 'history.pkl')}")
    tf.keras.backend.clear_session()
    return saved, history


def evaluate_run(model, data, saved, paths, history, hp):
    """Overlap 1 on train/val/test, then a val-only threshold (re-thresholding only, no re-inference)."""
    from plot_metrics import plot_training_metrics
    d_opts = data['data_opts']
    splits = (('train', data['train']), ('val', data['val']), ('test', data['test']))

    print_phase(PHASE_FULL_EVAL)
    metrics = {}
    for name, seq in splits:
        metrics[name] = evaluate_split_full_metrics(
            model, seq, d_opts, saved, paths, name, threshold=0.5)

    print_phase(PHASE_CALIBRATION)
    val_prob, val_true = load_ped_intents_pkl(saved, 'val')
    best_thresh, _ = find_best_threshold_by_f1(val_prob, val_true)
    calibration = build_calibration_dict(
        threshold=best_thresh, metric='f1', val_true=val_true, val_prob=val_prob,
        pos_weight_used=data['pos_weight'])
    with open(os.path.join(saved, 'calibration.json'), 'w', encoding='utf-8') as fid:
        json.dump(calibration, fid, indent=2)
    print(f"[calibration] best threshold on val = {best_thresh:.2f} "
          f"(val {calibration['metric_optimized']} = "
          f"{calibration['val_score_at_threshold']:.4f})")

    print_phase(PHASE_RECOMPUTE_CALIBRATED)
    calibrated = {
        name: recompute_split_metrics_at_threshold(saved, paths, name, best_thresh)
        for name, _ in splits}
    plot_training_metrics(
        history, metrics['train'], metrics['val'], metrics['test'], saved,
        calibrated_metrics=(calibrated['train'], calibrated['val'], calibrated['test']))

    print_summary_table(
        "SUMMARY -- ALL SPLITS (threshold=0.5)",
        f"[note] overlap=1 for all splits above -- not comparable to "
        f"per-epoch val/test numbers printed during training "
        f"(seq_overlap_rate={d_opts['seq_overlap_rate']}).",
        metrics['train'], metrics['val'], metrics['test'])
    print_summary_table(
        f"SUMMARY -- ALL SPLITS (calibrated threshold={best_thresh:.2f})",
        f"[note] same overlap=1 predictions, re-thresholded with the "
        f"val-selected threshold {best_thresh:.2f}.",
        calibrated['train'], calibrated['val'], calibrated['test'])
    prior_corrected = None
    if data['pos_weight'] is not None:
        prior_corrected = {
            name: recompute_split_metrics_prior_corrected(saved, paths, name, data['pos_weight'])
            for name, _ in splits}
        print_summary_table(
            f"SUMMARY -- ALL SPLITS (prior-corrected, pos_weight={data['pos_weight']:.4f})",
            "[note] same predictions, probabilities mapped back to the natural class prior. "
            "AUC is unchanged; accuracy/NLL/ECE are now comparable with majority_acc.",
            prior_corrected['train'], prior_corrected['val'], prior_corrected['test'])

    # --- extra threshold from HP (re-thresholding only, no re-inference) ---
    import numpy as np
    train_labels = [int(lbl[0][0]) for lbl in data['train']['intention_binary']]
    track_rate = sum(train_labels) / len(train_labels)
    _, train_true = load_ped_intents_pkl(saved, 'train')
    window_rate = float(np.mean(np.asarray(train_true).astype(int)))
    print(f"[threshold] train positive rate: tracks={track_rate:.4f} "
          f"windows(overlap=1)={window_rate:.4f}")

    space = hp.get('threshold_space', 'raw')
    thr = hp.get('decision_threshold', 0.5)
    if space == 'corrected' and data['pos_weight'] is None:
        print("[warning] threshold_space='corrected' needs loss=weighted_bce; using 'raw'.")
        space = 'raw'

    custom = None
    if thr == 0.5:
        print("[threshold] decision_threshold=0.5: raw@0.5 and prior-corrected@0.5 "
              "are already reported above.")
    else:
        custom = {name: recompute_split_metrics_custom(
                      saved, paths, name, thr, space, data['pos_weight'],
                      tag=f'custom_{space}_{thr}')
                  for name, _ in splits}
        print_summary_table(
            f"SUMMARY -- CUSTOM (space={space}, threshold={thr})",
            f"[note] train positive rate: tracks={track_rate:.4f} windows={window_rate:.4f}",
            custom['train'], custom['val'], custom['test'])

    return dict(metrics=metrics, metrics_calibrated=calibrated, threshold=best_thresh,
                metrics_prior_corrected=prior_corrected, metrics_custom=custom)


def run_training_pipeline(hp, data_opts, pie_path):
    """Menus, training, and both evaluation overlaps. The console-log redirect is always undone."""
    opts = pick_run_options(hp)
    data = prepare_data(opts, data_opts, pie_path)
    ident = choose_run_identity(opts, data, hp, pie_path)
    try:
        model, paths = start_run(opts, hp, data, ident, pie_path)
        saved, history = train_and_report(model, opts, hp, data, paths)
        results = evaluate_run(model, data, saved, paths, history, hp)
    finally:
        stop_console_log()
    return dict(model=model, paths=paths, saved_files_path=saved,
                history=history, opts=opts, **results)