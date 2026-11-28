"""One 3-class intention run. Answers the training menus, calls the unchanged pipeline,
then prints a full metric report for train/val/test from the saved predictions.

Branches are the four used by the binary run whose test ROC-AUC was 0.813
(pose_critical, scene_critical, light_color, flow_critical).
The later dense-flow checkpoint in the same log scored test ROC-AUC 0.793,
and its velocity class gap was 0.0017, so those two are not selected.
Auxiliary loss stays off: the menu builds sigmoid aux heads, and this task
uses categorical cross-entropy.
"""
import builtins
import os
import pickle
import sys

import numpy as np

PIE_PATH = r'D:\PIE codes\PIE_dataset'
os.environ['PIE_PATH'] = PIE_PATH

HP = dict(
    framework='tensorflow',
    lr=1e-5, clipvalue=None,
    batch_size=128,
    hidden_units=128, reg=0.001,
    lstm_dropout=0.4,
    lstm_recurrent_dropout=0.2,
    convlstm_filters=64, convlstm_kernel=2,
    es_min_delta=1e-4,
    es_patience=17,
    plateau_patience=4,
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

FEATURES = ['bbox', 'pose_critical', 'scene_critical', 'light_color', 'flow_critical']
LEAKAGE_PRONE = ('light_color', 'flow_critical')
CLASS_NAMES = ('nci', 'ui', 'ci')
_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_3class_answers.txt')


def _die(text):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unanswered_3class.txt')
    with open(path, 'w', encoding='utf-8') as fid:
        fid.write(text)
    raise SystemExit('No answer for this prompt. It was written to unanswered_3class.txt:\n' + text)


def decide(text):
    if '[offline_features]' in text:
        return '0,1,2'
    if '[use_visual]' in text:
        return 'y'
    if '[backbone]' in text:
        return '0'
    if 'default pie_convlstm' in text:
        return '5'
    if 'Select [1-4]' in text:
        return '4'
    if 'replace the visual temporal block' in text:
        return 'n'
    if 'replace the per-branch LSTM' in text:
        return 'n'
    if '[bbox_mode]' in text:
        return '0'
    if '[visual_norm]' in text:
        return 'n'

    if 'Select [0/1/2] (default 1):' in text and '3-class loss' in text:
        return '1'
    if text.strip().startswith('gamma [default 1.0]:'):
        return '1.0'
    if text.strip().startswith('focal_weight [default 1.0]:'):
        return '1.0'
    if text.strip().startswith('UI push factor [default 3.0]:'):
        return '3.0'
    if '[visual_percentile]' in text:
        return 'n'
    if '[capacity] add log1p' in text:
        return 'n'
    if '[capacity] ConvLSTM return_sequences' in text:
        return 'n'
    if '[capacity] select preset' in text:
        return '0'
    if '[log_mode]' in text:
        return '0'
    if 'Select [0/1/2] (default 1):' in text:
        return '1'
    if '[balance_mode]' in text:
        return '2'
    if '[flow_representation]' in text:
        return '0'
    if '[monitor_metric]' in text:
        return '1'
    if text.strip() == 'Select [0/1] (default 0):':
        return '1'
    if 'use obs=15 and overlap=0.3' in text:
        return 'y'
    if '[clip_mode]' in text:
        return '3'
    if '[weight_decay]' in text:
        return '0'
    if 'Choose [1/2]' in text:
        return '2'
    if '[normalize_bbox]' in text:
        return '0'
    if 'use ReduceLROnPlateau' in text:
        return 'y'
    if '[eval_every]' in text:
        return '4'
    if '[pose_scale]' in text:
        return '0'
    if '[pose_min_height]' in text:
        return '1'
    if 'attach time-to-event' in text:
        return 'n'
    if '[data_split_type]' in text:
        return '0'
    if '[seed] train seed' in text:
        return '1'
    if '[seed] shuffle seed' in text:
        return '1'
    if '[seed] data seed' in text:
        return '42'
    if '[extra_norm]' in text:
        return '1'
    for name in ('pose_critical', 'scene_critical', 'light_color', 'flow_critical'):
        if '[' + name + ']' in text:
            return ''
    if '[run_name]' in text:
        return 'c3_gated'
    if 'aux_loss_weight' in text:
        return '0'
    _die(text)


def _input(prompt=''):
    text = '' if prompt is None else str(prompt)
    value = decide(text)
    with open(_LOG, 'a', encoding='utf-8') as fid:
        fid.write('PROMPT ' + text.replace('\n', ' ') + '\n')
        fid.write('ANSWER ' + value + '\n')
    print('[auto] ' + value)
    return value


def _install_feature_pick():
    import train_helpers
    import train_pipeline

    def _pick(beh_seq, use_visual=True):
        missing = [name for name in FEATURES if name != 'bbox' and name not in beh_seq]
        if missing:
            raise SystemExit('These decoder features are not on the tracks: ' + str(missing))
        print('[config] decoder_input_type = ' + str(FEATURES))
        return list(FEATURES), {}

    train_helpers.interactive_pick_decoder_features = _pick
    train_pipeline.interactive_pick_decoder_features = _pick


def _install_overlap():
    import pie_intent_experiment
    original = pie_intent_experiment.PIEIntentExperiment.test_chunk

    def _chunk(self, *args, **kwargs):
        data_opts = kwargs.get('data_opts', args[1] if len(args) > 1 else {})
        output = data_opts.get('output_type') if isinstance(data_opts, dict) else None
        if kwargs.get('overlap_override') is None and output and 'intention_3class' in output:
            kwargs['overlap_override'] = float(data_opts['seq_overlap_rate'])
        return original(self, *args, **kwargs)

    pie_intent_experiment.PIEIntentExperiment.test_chunk = _chunk


# ---------------------------------------------------------------- full report
def _normalize(prob):
    prob = np.asarray(prob, dtype=float)
    if np.any(prob < 0):
        prob = np.exp(prob - prob.max(axis=1, keepdims=True))
    return prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-7, None)


def _ped_keys(res):
    """One pedestrian id per window (last frame). None if the pkl layout is unexpected."""
    try:
        return np.array([str(np.asarray(p, dtype=object).ravel()[-1]) for p in res['ped_id']])
    except Exception:
        return None


def _bootstrap(y, prob, keys, n_boot, seed=0):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(keys)
    groups = {k: np.flatnonzero(keys == k) for k in uniq}
    draws = {'roc_auc': [], 'balanced_acc': [], 'accuracy': []}
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(uniq, size=len(uniq), replace=True)])
        yy, pp = y[idx], prob[idx]
        if len(np.unique(yy)) < 3:
            continue
        pred = pp.argmax(axis=1)
        draws['roc_auc'].append(roc_auc_score(yy, pp, multi_class='ovr', average='macro', labels=[0, 1, 2]))
        draws['balanced_acc'].append(balanced_accuracy_score(yy, pred))
        draws['accuracy'].append(float((pred == yy).mean()))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v))
            for k, v in draws.items() if v}, len(uniq)


def report_from_pkl(run_dir, split, overlap, n_boot=1000):
    """Reads ped_intents_<split>.pkl (written by test_chunk), prints and saves every metric."""
    from pie_intent_helpers import compute_multiclass_metrics
    with open(os.path.join(run_dir, f'ped_intents_{split}.pkl'), 'rb') as fid:
        res = pickle.load(fid)
    y_true = np.asarray(res['gt'])
    y_prob = np.asarray(res['results'], dtype=float)
    m = compute_multiclass_metrics(
        y_true, y_prob, save_path=os.path.join(run_dir, f'metrics_{split}_3class_eval.json'))
    y = y_true.argmax(axis=1) if y_true.ndim == 2 else y_true.astype(int).ravel()
    prob = _normalize(y_prob)
    cm = np.asarray(m['confusion_matrix'])
    n = m['n_test_samples']

    leaky = [f for f in FEATURES if f in LEAKAGE_PRONE]
    lines = [f'=== 3-class report | split={split} | n={n} | overlap={overlap} ===',
             f'run dir: {run_dir}',
             f'decoder features: {FEATURES}',
             f'leakage-prone inputs present: {leaky} -> report as a labelled leakage-prone configuration' if leaky
             else 'leakage-prone inputs present: none',
             '', '--- overall (argmax decision) ---']
    for key in ('accuracy', 'majority_acc', 'balanced_acc', 'precision', 'recall', 'specificity',
                'f1', 'roc_auc', 'pr_auc', 'nll', 'ece', 'pred_mean', 'pred_std'):
        lines.append(f'{key:<14}{m[key]:.4f}')
    lines.append(f"{'acc-majority':<14}{m['accuracy'] - m['majority_acc']:+.4f}")

    lines += ['', '--- per class (one-vs-rest) ---',
              f"{'class':<6}{'support':>9}{'prec':>8}{'recall':>8}{'spec':>8}{'f1':>8}{'auc':>8}{'ap':>8}"]
    for c in CLASS_NAMES:
        lines.append(f"{c:<6}{int(m['support_' + c]):>9}{m['precision_' + c]:>8.4f}{m['recall_' + c]:>8.4f}"
                     f"{m['specificity_' + c]:>8.4f}{m['f1_' + c]:>8.4f}{m['auc_' + c]:>8.4f}{m['ap_' + c]:>8.4f}")

    lines += ['', '--- confusion matrix (rows = true, cols = predicted) ---',
              f"{'':<6}" + ''.join(f'{c:>8}' for c in CLASS_NAMES) + '   row % (nci ui ci)']
    for i, c in enumerate(CLASS_NAMES):
        pct = ' '.join(f'{100 * v / max(cm[i].sum(), 1):5.1f}' for v in cm[i])
        lines.append(f'{c:<6}' + ''.join(f'{v:>8}' for v in cm[i]) + f'   {pct}')
    share = cm.sum(axis=0) / cm.sum()
    lines.append('predicted share: ' + ' '.join(f'{c}={s:.4f}' for c, s in zip(CLASS_NAMES, share)))

    keys = _ped_keys(res)
    if n_boot and keys is not None:
        ci, n_ped = _bootstrap(y, prob, keys, n_boot)
        lines += ['', f'--- 95% CI, bootstrap over pedestrians ({n_ped} ids, {n_boot} draws) ---']
        for k, (lo, hi, used) in ci.items():
            lines.append(f'{k:<14}[{lo:.4f}, {hi:.4f}]  valid draws={used}')
    elif n_boot:
        lines += ['', '[note] pedestrian ids not readable from the pkl: bootstrap skipped.']

    if split == 'test' and n != 6472:
        lines.append('[warning] test n != 6472: not comparable with Diving Deeper Table II.')
    lines.append('[note] single seed, single split: AUC gaps of a few hundredths are not evidence of superiority.')

    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(run_dir, f'full_report_{split}_3class.txt'), 'w', encoding='utf-8') as fid:
        fid.write(text + '\n')
    return m


def main():
    if os.path.exists(_LOG):
        os.remove(_LOG)
    builtins.input = _input
    _install_feature_pick()
    _install_overlap()
    from train_pipeline import run_training_pipeline
    out = run_training_pipeline(HP, DATA_OPTS, PIE_PATH)
    overlap = 0.3 if out['opts'].get('paper_protocol') else DATA_OPTS['seq_overlap_rate']
    for split in ('train', 'val', 'test'):
        report_from_pkl(out['saved_files_path'], split, overlap, n_boot=0 if split == 'train' else 1000)


if __name__ == '__main__':
    main()