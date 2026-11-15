"""probe_common.py -- shared helpers for ablation/probing on a finished run."""
import os
import pickle
import numpy as np

PIE_PATH = r'D:\PIE codes\PIE_dataset'
os.environ['PIE_PATH'] = PIE_PATH  # must be set before importing pie_intent*

from keras.models import load_model
from pie_data_Hesam import PIE, get_tracks


def load_run(run_dir, framework, architecture, run_name):
    """Return (experiment, keras_model, train_params) for a finished run."""
    from pie_intent_experiment import PIEIntentExperiment
    with open(os.path.join(run_dir, 'configs.pkl'), 'rb') as f:
        configs = pickle.load(f)
    exp = PIEIntentExperiment(framework=framework, architecture=architecture,
                              run_name=run_name, run_dir=run_dir)
    exp.load_model_config(configs[0])
    exp.normalize_decoder_bbox = True  # must match the training preset
    from feature_routing import CUSTOM_OBJECTS
    model = load_model(os.path.join(run_dir, 'model.h5'), compile=False,
                       custom_objects=CUSTOM_OBJECTS)
    return exp, model, configs[1]


def build_behavior_lookup(imdb):
    """Map (ped_id, frame_id) -> (look, action) from the annotation database."""
    db = imdb.generate_database()
    lut = {}
    for sid in db:
        for vid in db[sid]:
            if 'ped_annotations' not in db[sid][vid]:
                continue
            for pid, a in db[sid][vid]['ped_annotations'].items():
                beh = a['behavior']
                for fr, lk, ac in zip(a['frames'], beh['look'], beh['action']):
                    lut[(str(pid), int(fr))] = (lk, ac)
    return lut


def window_behavior(lut, images, ped_ids):
    """Per-window look/action at the LAST frame of the window (-1 if missing)."""
    look, act = [], []
    for imgs, pids in zip(images, ped_ids):
        pid = str(np.asarray(pids).ravel()[0])  # ped_ids items look like [pid]
        fr = int(os.path.splitext(os.path.basename(imgs[-1]))[0])
        lk, ac = lut.get((pid, fr), (-1, -1))
        look.append(lk)
        act.append(ac)
    return np.array(look), np.array(act)


def iter_chunks(exp, train_params, data, lut, subset, overlap=0.5,
                tracks_per_chunk=20):
    """Yield (feat, dec, y, look, act) per group of tracks (memory-safe).

    subset: 'val' or 'test' (selects the feature cache folder).
    Assumes every value in `data` is a per-track list.
    """
    L = exp._sequence_length
    n = len(data['image'])
    for i in range(0, n, tracks_per_chunk):
        sub = {k: (v[i:i + tracks_per_chunk] if isinstance(v, (list, tuple)) else v)
               for k, v in data.items()}
        _, imgs, _, pids = get_tracks(sub, train_params['data_type'], L, overlap)
        if not imgs:
            continue
        (feat, dec), y = exp.get_test_data(sub, train_params, L, overlap,
                                           data_subset=subset)
        look, act = window_behavior(lut, imgs, pids)
        yield feat, dec, np.ravel(y), look, act