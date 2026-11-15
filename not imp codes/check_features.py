"""check_features.py -- are the decoder features real numbers, non-empty and informative?
Run in pie_tf or pie_pose (no TensorFlow needed).  Usage:  python check_features.py
"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

PIE_PATH = r'D:\PIE codes\PIE_dataset'
CODE_DIR = r'D:\PIE codes\PIE'
SPLIT_TYPE = 'random'                    # 'default' or 'random' (0.5/0.4/0.1); must match the run you check
FEATURES = ['bbox', 'center', 'occlusion', 'gesture', 'look', 'action', 'cross', 'pose']
MODEL_FEATURES = ['bbox', 'center', 'look']   # what you selected in the menu (exact order)

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
os.environ['PIE_PATH'] = PIE_PATH


def _as_tracks(beh_seq, key):
    out = []
    for t in beh_seq[key]:
        a = np.asarray(t, dtype=np.float32)          # raises ValueError if text is inside
        out.append(a.reshape(len(a), -1))
    return out


def report_feature(beh_seq, key):
    if key not in beh_seq:
        print(f'  [{key}] MISSING: key not in this sequence dict')
        return
    labels = np.array([t[0][0] for t in beh_seq['intention_binary']]).astype(int)
    try:
        tracks = _as_tracks(beh_seq, key)
    except (ValueError, TypeError) as e:
        print(f'  [{key}] NOT NUMERIC -> {e}')
        return
    widths = {a.shape[1] for a in tracks}
    if len(widths) != 1:
        print(f'  [{key}] INCONSISTENT WIDTH across tracks: {sorted(widths)}')
        return
    D = widths.pop()
    frames = np.concatenate(tracks)                   # (all frames, D)
    flags = []
    if not np.isfinite(frames).all():
        flags.append(f'NaN/Inf in {100 * (~np.isfinite(frames)).mean():.2f}% of values')
    zero_vec = float((frames == 0).all(axis=1).mean())
    if zero_vec > 0.95:
        flags.append('EMPTY (>95% of frames are all-zero)')
    elif zero_vec > 0.5:
        flags.append('MOSTLY EMPTY (>50% of frames are all-zero)')
    const_cols = int((frames.std(axis=0) == 0).sum())
    if const_cols == D:
        flags.append('CONSTANT (no variation at all)')
    changes = float(np.mean([np.ptp(a, axis=0).max() > 0 for a in tracks]))
    summary = np.stack([a.mean(axis=0) for a in tracks])          # per-track mean, (tracks, D)
    aucs = []
    if len(set(labels)) == 2:
        for d in range(D):
            if summary[:, d].std() > 0:
                a = roc_auc_score(labels, summary[:, d])
                aucs.append(max(a, 1 - a))                         # direction-free
    best = max(aucs) if aucs else float('nan')
    if best > 0.95:
        flags.append('LEAK? one dimension almost perfectly separates the labels')
    uniq = np.unique(frames[:, 0]) if D == 1 else None
    print(f'  [{key}] width={D} frames={len(frames)} finite={100 * np.isfinite(frames).mean():.1f}% '
          f'all-zero frames={100 * zero_vec:.1f}% constant cols={const_cols}/{D} '
          f'tracks that change inside={100 * changes:.0f}% best track-level AUC={best:.3f}')
    if uniq is not None and len(uniq) <= 8:
        share = {float(u): f'{100 * np.mean(frames[:, 0] == u):.1f}%' for u in uniq}
        print(f'        values: {share}')
        if uniq.max() > 1 or uniq.min() < 0:
            print('        note: these are category CODES, not magnitudes (a larger code is not "more"); '
                  'consider one-hot')
    for f in flags:
        print(f'        !! {f}')


def check_model_input(beh_seq, features, seq_length=15, overlap=0.5):
    from pie_data_Hesam import get_tracks
    from utils import concat_data
    dt = {'encoder_input_type': [], 'decoder_input_type': features, 'output_type': ['intention_binary']}
    d = get_tracks(beh_seq, dt, seq_length, overlap)[0]
    x = concat_data(d, features)                      # (windows, seq_length, D)
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    print(f'  decoder input array: {x.shape}  finite={np.isfinite(flat).all()}')
    print('  col |      mean |       std |       min |       max')
    for c in range(flat.shape[1]):
        col = flat[:, c]
        warn = '  <-- constant' if col.std() == 0 else ('  <-- large scale' if np.abs(col).max() > 100 else '')
        print(f'  {c:3d} | {col.mean():9.3f} | {col.std():9.3f} | {col.min():9.3f} | {col.max():9.3f}{warn}')


def main():
    from pie_data_Hesam import PIE
    from extra_features import attach_pose
    imdb = PIE(data_path=PIE_PATH)
    opts = dict(seq_type='intention', min_track_size=15)
    if SPLIT_TYPE == 'random':
        opts.update(data_split_type='random',
                    random_params={'ratios': [0.5, 0.4, 0.1], 'val_data': True, 'regen_data': False})
    seqs = {s: imdb.generate_data_trajectory_sequence(s, **opts) for s in ('train', 'val', 'test')}
    for s in seqs.values():
        attach_pose(s, os.path.join(PIE_PATH, 'extra', 'pose'))
    for split, seq in seqs.items():
        print(f'\n=== {split}: {len(seq["image"])} tracks ===')
        for key in FEATURES:
            report_feature(seq, key)
    print('\n=== model input (train windows) ===')
    check_model_input(seqs['train'], MODEL_FEATURES)


if __name__ == '__main__':
    main()