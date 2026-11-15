"""test_pose_parts.py -- where does the pose signal come from?  Run after test_each_feature.py:

    python test_pose_parts.py

Reuses its loader and logistic-regression helper. Every row is a logistic regression on TRACK MEANS
(train only, StandardScaler), reporting AUC on val and test.

  xy_all_frames     keypoint positions inside the bbox, zeros included for frames with no keypoint
  xy_usable_frames  same, averaged only over frames that have a confident keypoint
  conf_only         the 17 detector confidences (image quality / size / occlusion proxy)
  nconf, miss       how many keypoints pass the threshold, share of frames with none
  size              log width, log height, aspect ratio of the bbox (distance proxy)
  quality+size      everything about the detection and bbox size, but NO body shape
If quality+size is about as high as the full pose, the pose gain is a distance/quality proxy.
If xy_usable_frames is clearly above it, body shape itself carries signal.
"""
import numpy as np

import test_each_feature as T

NUM_KPTS = 17
MIN_CONF = 0.3        # same threshold as attach_pose


def parts(beh):
    out = {k: [] for k in ('xy_all', 'xy_usable', 'conf', 'nconf', 'miss', 'size')}
    for i in range(len(beh['pose'])):
        p = np.asarray(beh['pose'][i], dtype=np.float64).reshape(-1, NUM_KPTS, 3)   # (T, 17, [x, y, conf])
        b = np.asarray(beh['bbox'][i], dtype=np.float64)                             # (T, 4)
        ok = p[:, :, 2] >= MIN_CONF
        frame_ok = ok.any(axis=1)
        out['xy_all'].append(p[:, :, :2].reshape(len(p), -1).mean(axis=0))
        out['xy_usable'].append(p[frame_ok][:, :, :2].reshape(int(frame_ok.sum()), -1).mean(axis=0)
                                if frame_ok.any() else np.zeros(2 * NUM_KPTS))
        out['conf'].append(p[:, :, 2].mean(axis=0))
        out['nconf'].append([ok.sum(axis=1).mean()])
        out['miss'].append([1.0 - frame_ok.mean()])
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        out['size'].append([np.log(w.mean() + 1), np.log(h.mean() + 1), (w / np.maximum(h, 1)).mean()])
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


VARIANTS = [
    ('size', ['size']),
    ('miss', ['miss']),
    ('nconf', ['nconf']),
    ('conf_only', ['conf']),
    ('xy_all_frames', ['xy_all']),
    ('xy_usable_frames', ['xy_usable']),
    ('quality+size', ['size', 'miss', 'nconf', 'conf']),
    ('xy_usable+quality+size', ['xy_usable', 'size', 'miss', 'nconf', 'conf']),
]


def main():
    train, val, test = T.load_splits()
    y = {'train': T.labels_of(train), 'val': T.labels_of(val), 'test': T.labels_of(test)}
    P = {'train': parts(train), 'val': parts(val), 'test': parts(test)}
    bbox = {'train': T.track_matrix(train, 'bbox'), 'val': T.track_matrix(val, 'bbox'),
            'test': T.track_matrix(test, 'bbox')}
    for name in ('conf', 'xy_all'):                              # layout sanity: conf in [0,1]
        print(f'  sanity {name}: min={P["train"][name].min():.2f} max={P["train"][name].max():.2f}')

    def run(label, keys, with_bbox=False):
        X = {}
        for s in ('train', 'val', 'test'):
            cols = [P[s][k] for k in keys] + ([bbox[s]] if with_bbox else [])
            X[s] = np.hstack(cols)
        v, t = T.fit_auc(X['train'], y['train'], X['val'], y['val'], X['test'], y['test'], scale=True)
        print(f'{label:34} {X["train"].shape[1]:4d}  val={v:.3f}  test={t:.3f}')

    print(f'\n{"variant":34} {"dim":>4}  (track-mean logistic regression, scaled)')
    run('bbox (reference)', [], with_bbox=True)
    for label, keys in VARIANTS:
        run(label, keys)
    run('bbox + xy_usable_frames', ['xy_usable'], with_bbox=True)
    run('bbox + quality+size', ['size', 'miss', 'nconf', 'conf'], with_bbox=True)


if __name__ == '__main__':
    main()