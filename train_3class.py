"""One 3-class intention run. Answers the training menus and then calls the unchanged pipeline.

Branches are the four used by the binary run whose test ROC-AUC was 0.813
(pose_critical, scene_critical, light_color, flow_critical).
The later dense-flow checkpoint in the same log scored test ROC-AUC 0.793,
and its velocity class gap was 0.0017, so those two are not selected.
Auxiliary loss stays off: the menu builds sigmoid aux heads, and this task
uses categorical cross-entropy.
"""
import builtins
import os
import sys

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

FEATURES = ['bbox', 'pose_critical', 'scene_critical', 'light_color', 'flow_critical']
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
    if '[visual_percentile]' in text:
        return 'n'
    if '[capacity] add log1p' in text:
        return 'n'
    if '[capacity] add log1p' in text:
        return 'n'
    if '[capacity] ConvLSTM return_sequences' in text:
        return 'n'
    if '[capacity] select preset' in text:
        return '0'
    if '[log_mode]' in text:
        return '0'
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


def main():
    if os.path.exists(_LOG):
        os.remove(_LOG)
    builtins.input = _input
    _install_feature_pick()
    _install_overlap()
    from train_pipeline import run_training_pipeline
    run_training_pipeline(HP, DATA_OPTS, PIE_PATH)


if __name__ == '__main__':
    main()