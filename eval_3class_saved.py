import builtins
import os
import sys

os.environ['PIE_PATH'] = r'D:\PIE codes\PIE_dataset'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_3class as T

RUN_DIR = (r'D:\PIE codes\PIE_dataset\experiments\tensorflow\branchattn_temporal'
           r'\c3_gated_gated_add_BAttnT6_3c_defaultSplit')
SPLITS = ('val', 'test')
REPORT_ONLY = True   # True: use ped_intents_<split>.pkl already on disk, no model run
OVERLAP = 0.3        # paper_protocol answer 'y' in train_3class.decide


def run_model():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    builtins.input = T._input
    from train_pipeline import pick_run_options, prepare_data
    from pie_intent_experiment import PIEIntentExperiment

    hp = T.HP
    opts = pick_run_options(hp)
    data = prepare_data(opts, T.DATA_OPTS, T.PIE_PATH)
    d_opts = data['data_opts']
    overlap = float(d_opts['seq_overlap_rate'])
    print(f"[eval] output_type={d_opts['output_type']} overlap={overlap}")

    model = PIEIntentExperiment(
        framework=hp['framework'], architecture='branchattn_temporal',
        run_name=os.path.basename(RUN_DIR), run_dir=RUN_DIR,
        num_hidden_units=hp['hidden_units'], regularizer_val=hp['reg'],
        lstm_dropout=hp['lstm_dropout'],
        lstm_recurrent_dropout=hp['lstm_recurrent_dropout'],
        convlstm_num_filters=hp['convlstm_filters'],
        convlstm_kernel_size=hp['convlstm_kernel'],
        lstm_recurrent_activation=opts['rec_act'])
    for name in SPLITS:
        model.test_chunk(data[name], d_opts, RUN_DIR, visualize=False,
                         results_suffix=f'_{name}', overlap_override=overlap)
    return overlap


overlap = OVERLAP if REPORT_ONLY else run_model()
for name in SPLITS:
    T.report_from_pkl(RUN_DIR, name, overlap, n_boot=1000)