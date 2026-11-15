"""preview_pose.py -- import in Jupyter and call preview_pose(min_h, max_h)."""
import os
import sys
import random
import numpy as np
import cv2

PIE_PATH = r'D:\PIE codes\PIE_dataset'
CODE_DIR = r'D:\PIE codes\PIE'
os.environ['PIE_PATH'] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from pie_data_Hesam import PIE
from extra_features import _parse_frame_path

POSE_DIR = os.path.join(PIE_PATH, 'extra', 'pose')
OUT_DIR = os.path.join(PIE_PATH, 'extra', 'pose_preview')   # short path on purpose
MIN_CONF = 0.3          # must match attach_pose(min_conf=...)
TILE_W, TILE_H = 200, 320

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

_IMDB = {}
_SEQ = {}


def _get_seq(split):
    """Load the training-time cached sequences once per kernel."""
    if 'imdb' not in _IMDB:
        _IMDB['imdb'] = PIE(data_path=PIE_PATH)
    if split not in _SEQ:
        _SEQ[split] = _IMDB['imdb'].generate_data_trajectory_sequence(
            split, seq_type='intention', min_track_size=15)
    return _SEQ[split]


def _conf_color(c):
    t = float(np.clip((c - MIN_CONF) / (1.0 - MIN_CONF), 0, 1))
    return (0, int(255 * t), int(255 * (1 - t)))          # BGR: red = low conf, green = high


def _render(img, k, bbox, label):
    x1, y1, x2, y2 = bbox
    m = 0.25 * max(x2 - x1, y2 - y1)
    cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    cx2, cy2 = int(min(img.shape[1], x2 + m)), int(min(img.shape[0], y2 + m))
    crop = img[cy1:cy2, cx1:cx2]
    s = min(TILE_W / crop.shape[1], TILE_H / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_CUBIC)
    # keypoints are mapped into the RESIZED crop, so marker size is fixed in tile pixels
    pts = np.stack([(k[:, 0] - cx1) * s, (k[:, 1] - cy1) * s], axis=1)
    ok = k[:, 2] >= MIN_CONF
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(crop, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (255, 200, 0), 1, cv2.LINE_AA)
    for j in range(17):
        p = tuple(pts[j].astype(int))
        if ok[j]:
            cv2.circle(crop, p, 3, _conf_color(k[j, 2]), -1, cv2.LINE_AA)
        elif k[j, 2] > 0:
            cv2.circle(crop, p, 2, (160, 160, 160), 1, cv2.LINE_AA)   # rejected joint
    bx = [int((x1 - cx1) * s), int((y1 - cy1) * s), int((x2 - cx1) * s), int((y2 - cy1) * s)]
    cv2.rectangle(crop, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 255), 1)   # tracked bbox
    tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    y0, x0 = (TILE_H - crop.shape[0]) // 2, (TILE_W - crop.shape[1]) // 2
    tile[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    cv2.putText(tile, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    return tile


def preview_pose(min_h=40, max_h=80, split='val', n_samples=48, cols=6,
                 seed=0, bad_only=False, show=True):
    """Contact sheet of pose overlays for frames with min_h <= bbox height < max_h.

    :param min_h, max_h: bbox height band in px (max_h may be float('inf')).
    :param split: 'train', 'val' or 'test'.
    :param seed: change it to see a different random sample.
    :param bad_only: True = only frames with fewer than 8 usable joints (biased, not a rate estimate).
    :param show: display inline in Jupyter.
    :return: path of the saved JPEG, or None if no frame matched.
    """
    seq = _get_seq(split)
    rng = random.Random(seed)

    cand = []   # (track idx, frame idx, bbox height)
    for i, paths in enumerate(seq['image']):
        pid = seq['ped_id'][i][0][0]
        sid, vid, _ = _parse_frame_path(paths[0])
        if not os.path.isfile(os.path.join(POSE_DIR, sid, vid, f'{pid}.npz')):
            continue
        for j, b in enumerate(seq['bbox'][i]):
            h = b[3] - b[1]
            if min_h <= h < max_h:
                cand.append((i, j, h))
    print(f'[{split}] candidate frames with bbox height in [{min_h}, {max_h}): {len(cand)}')
    if not cand:
        return None
    rng.shuffle(cand)

    tiles, cache, n_bad = [], {}, 0
    for i, j, h in cand:
        if len(tiles) >= n_samples:
            break
        pid = seq['ped_id'][i][0][0]
        path = seq['image'][i][j]
        sid, vid, fr = _parse_frame_path(path)
        key = (sid, vid, pid)
        if key not in cache:
            z = np.load(os.path.join(POSE_DIR, sid, vid, f'{pid}.npz'))
            cache[key] = ({int(f_): r for r, f_ in enumerate(z['frames'])}, z['kpts'])
        row_of, kpts = cache[key]
        r = row_of.get(fr)
        if r is None:
            continue
        k = kpts[r]
        n_ok = int((k[:, 2] >= MIN_CONF).sum())
        if bad_only and n_ok >= 8:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        n_bad += int(n_ok < 8)
        tiles.append(_render(img, k, seq['bbox'][i][j], f'{vid[-4:]} f{fr} h={h:.0f} ok={n_ok}'))

    if not tiles:
        print('no frame passed the filters')
        return None
    shown = len(tiles)
    print(f'tiles drawn: {shown}; of these with fewer than 8 usable joints: {n_bad}')
    while len(tiles) % cols:
        tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])

    os.makedirs(OUT_DIR, exist_ok=True)
    hi = 'inf' if max_h == float('inf') else int(max_h)
    out = os.path.join(OUT_DIR, f'pose_h{int(min_h)}-{hi}_{split}_s{seed}.jpg')
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('saved ->', out)
    if show:
        from IPython.display import Image, display
        display(Image(data=cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()))
    return out