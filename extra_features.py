"""extra_features.py -- attach offline-computed pose / scene / velocity / tte / flow features to PIE track dicts."""
import os
import numpy as np

NUM_KPTS = 17  # COCO keypoints


def _parse_frame_path(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3], parts[-2], int(os.path.splitext(parts[-1])[0])  # set, video, frame


def attach_pose(beh_seq, pose_dir, key='pose', min_conf=0.3, scale='bbox_relative',
                min_bbox_height=0, verbose=True):
    """Adds beh_seq[key]: per track, per frame, a 51-d vector (x, y, conf per keypoint).

    Cache format (written by the offline extractor): <pose_dir>/<set>/<video>/<ped_id>.npz with
      frames (T,) int and kpts (T, 17, 3) float32 in ABSOLUTE image pixels (x, y, conf).
    scale='bbox_relative': x,y relative to the current bbox (about [-0.5, 1.5]).
    scale='absolute': x,y left in image pixels (same units as raw bbox, but leaks position).
    Call this BEFORE balancing/augmentation (ped_ids must still be the original ones).
    verbose=False skips the coverage report.
    """
    if scale not in ('bbox_relative', 'absolute'):
        raise ValueError(f'unknown pose scale: {scale}')
    out, missing_tracks, zero_frames, total, masked_small = [], 0, 0, 0, 0
    for i, paths in enumerate(beh_seq['image']):
        pid = beh_seq['ped_id'][i][0][0]
        assert '__aug' not in pid, 'attach_pose must run before augmentation'
        sid, vid, _ = _parse_frame_path(paths[0])
        f = os.path.join(pose_dir, sid, vid, f'{pid}.npz')
        track = []
        if os.path.isfile(f):
            z = np.load(f)
            row_of = {int(fr): r for r, fr in enumerate(z['frames'])}
            kpts = z['kpts']
        else:
            missing_tracks += 1
            row_of, kpts = {}, None
        for j, p in enumerate(paths):
            total += 1
            x1, y1, x2, y2 = beh_seq['bbox'][i][j]
            w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            vec = np.zeros((NUM_KPTS, 3), np.float32)
            small = min_bbox_height > 0 and (y2 - y1) < min_bbox_height
            if small:
                masked_small += 1
            r = None if small else row_of.get(_parse_frame_path(p)[2])
            if r is not None:
                k = kpts[r].copy()
                ok = k[:, 2] >= min_conf
                if scale == 'absolute':
                    vec[ok, 0] = k[ok, 0]
                    vec[ok, 1] = k[ok, 1]
                else:
                    vec[ok, 0] = np.clip((k[ok, 0] - x1) / w, -0.5, 1.5)
                    vec[ok, 1] = np.clip((k[ok, 1] - y1) / h, -0.5, 1.5)
                vec[ok, 2] = k[ok, 2]
            if not vec[:, 2].any():
                zero_frames += 1
            track.append(vec.reshape(-1).tolist())
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[pose] scale={scale}; tracks without cache file: {missing_tracks}/{len(out)}; '
              f'frames with no usable keypoint: {zero_frames}/{total} ({100 * zero_frames / max(total, 1):.1f}%)')
    if min_bbox_height > 0:
        print(f'[pose] zeroed because bbox height < {min_bbox_height}px: '
              f'{masked_small}/{total} ({100 * masked_small / max(total, 1):.1f}%)')
    return beh_seq

# COCO keypoint indices
KP = {'nose':0, 'l_eye':1, 'r_eye':2, 'l_ear':3, 'r_ear':4,
      'l_sho':5, 'r_sho':6, 'l_elb':7, 'r_elb':8, 'l_wri':9, 'r_wri':10,
      'l_hip':11, 'r_hip':12, 'l_kne':13, 'r_kne':14, 'l_ank':15, 'r_ank':16}

def _angle(a, b, c):
    """Angle at b (degrees). Returns 0 if any point invalid."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# Cityscapes trainId groups -> 6 scene classes
_GROUPS = {'road': [0], 'sidewalk': [1], 'vehicle': [13, 14, 15, 16],
           'two_wheeler': [12, 17, 18], 'person': [11]}
GROUP_NAMES = list(_GROUPS) + ['other']


def scene_features(label_map, bbox, scale=1.0):
    """24-d vector: 6 class fractions in 4 regions around the pedestrian.
    label_map: (H, W) uint8 Cityscapes trainIds; bbox in ORIGINAL 1920x1080 pixels;
    scale = label_map_width / 1920 when the map was computed on a downscaled frame."""
    H, W = label_map.shape
    x1, y1, x2, y2 = [v * scale for v in bbox]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    regions = [  # (xa, ya, xb, yb)
        (x1 - 0.25 * w, y2 - 0.05 * h, x2 + 0.25 * w, y2 + 0.15 * h),  # under feet
        (x1 - 1.5 * w, y1, x1, y2 + 0.15 * h),                         # left
        (x2, y1, x2 + 1.5 * w, y2 + 0.15 * h),                         # right
        (x1 - 0.5 * w, y2, x2 + 0.5 * w, y2 + 1.0 * h),                # between ped and camera
    ]
    lut = np.full(256, len(_GROUPS), np.uint8)
    for g, (_, ids) in enumerate(_GROUPS.items()):
        lut[ids] = g
    feats = []
    for xa, ya, xb, yb in regions:
        xa, xb = int(np.clip(round(xa), 0, W)), int(np.clip(round(xb), 0, W))
        ya, yb = int(np.clip(round(ya), 0, H)), int(np.clip(round(yb), 0, H))
        patch = label_map[ya:yb, xa:xb]
        if patch.size == 0:
            feats.extend([0.0] * len(GROUP_NAMES))
            continue
        cnt = np.bincount(lut[patch].ravel(), minlength=len(GROUP_NAMES)).astype(np.float32)
        feats.extend((cnt / cnt.sum()).tolist())
    return feats

def scene_critical_features(label_map, bbox, scale=1.0):
    """Targeted 16-d scene features focused on critical objects relative to the pedestrian.

    Components:
      0-11 : fraction of {sidewalk, traffic_light, traffic_sign} in 4 regions
             (under feet, left, right, front)  -> 3 classes x 4 regions
      12   : normalized min distance to nearest sidewalk pixel (in bbox-height units)
      13   : normalized min distance to nearest traffic-light pixel
      14   : sidewalk fraction specifically under feet (repeated for emphasis)
      15   : soft presence of traffic light in the front region (>0.01 fraction)

    label_map: (H, W) uint8 Cityscapes trainId
    bbox: [x1, y1, x2, y2] in original 1920x1080 pixels
    scale: label_map_width / 1920
    """
    H, W = label_map.shape
    x1, y1, x2, y2 = [v * scale for v in bbox]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    # Critical class ids (Cityscapes trainId)
    SIDEWALK = 1
    TRAFFIC_LIGHT = 6
    TRAFFIC_SIGN = 7
    critical_ids = [SIDEWALK, TRAFFIC_LIGHT, TRAFFIC_SIGN]

    regions = [
        (x1 - 0.25 * w, y2 - 0.05 * h, x2 + 0.25 * w, y2 + 0.15 * h),  # under feet
        (x1 - 1.5 * w, y1, x1, y2 + 0.15 * h),                         # left
        (x2, y1, x2 + 1.5 * w, y2 + 0.15 * h),                         # right
        (x1 - 0.5 * w, y2, x2 + 0.5 * w, y2 + 1.0 * h),                # front (between ped & camera)
    ]

    feats = []
    under_feet_sidewalk = 0.0
    front_tl_frac = 0.0

    for ri, (xa, ya, xb, yb) in enumerate(regions):
        xa = int(np.clip(round(xa), 0, W))
        xb = int(np.clip(round(xb), 0, W))
        ya = int(np.clip(round(ya), 0, H))
        yb = int(np.clip(round(yb), 0, H))
        patch = label_map[ya:yb, xa:xb]
        if patch.size == 0:
            feats.extend([0.0] * 3)
            continue
        total = float(patch.size)
        for cid in critical_ids:
            frac = float(np.sum(patch == cid)) / total
            feats.append(frac)
            if ri == 0 and cid == SIDEWALK:
                under_feet_sidewalk = frac
            if ri == 3 and cid == TRAFFIC_LIGHT:
                front_tl_frac = frac

    # Approximate min distance to nearest critical pixel (in units of bbox height)
    # Use a coarse grid search over the whole map for simplicity and speed
    def min_dist_to_class(class_id):
        ys, xs = np.where(label_map == class_id)
        if len(ys) == 0:
            return 10.0  # large value = "not present"
        # pedestrian foot center
        px = 0.5 * (x1 + x2)
        py = y2
        dists = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        return float(np.min(dists)) / h

    dist_sidewalk = min_dist_to_class(SIDEWALK)
    dist_tl = min_dist_to_class(TRAFFIC_LIGHT)

    feats.append(dist_sidewalk)
    feats.append(dist_tl)
    feats.append(under_feet_sidewalk)
    feats.append(1.0 if front_tl_frac > 0.01 else 0.0)

    assert len(feats) == 16
    return feats


def attach_scene_critical(beh_seq, label_dir, key='scene_critical', cache_size=64, verbose=True):
    """Attach the 16-d targeted critical-object scene features.
    Uses the same label maps as attach_scene.
    """
    from functools import lru_cache
    from PIL import Image

    @lru_cache(maxsize=cache_size)
    def load(sid, vid, fr):
        f = os.path.join(label_dir, sid, vid, f'{fr:05d}.png')
        return np.asarray(Image.open(f)) if os.path.isfile(f) else None

    out, missing, total = [], 0, 0
    for i, paths in enumerate(beh_seq['image']):
        track = []
        for j, p in enumerate(paths):
            total += 1
            sid, vid, fr = _parse_frame_path(p)
            lm = load(sid, vid, fr)
            if lm is None:
                missing += 1
                track.append([0.0] * 16)
            else:
                track.append(scene_critical_features(
                    lm, beh_seq['bbox'][i][j], scale=lm.shape[1] / 1920.0))
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[scene_critical] frames without label map: {missing}/{total}')
    return beh_seq

LIGHT_COLOR_DIM = 4


def traffic_light_color(frame_rgb, label_map, bbox, scale):
    """4-d traffic-light color from the raw frame, masked by the label map.

    Not a pedestrian-signal detector. Cityscapes class 6 is the housing plus
    the lamp. Only the brightest 10 percent of those pixels are used, because
    the housing is dark and would otherwise dominate the mean.
      0 red fraction
      1 green fraction
      2 yellow fraction
      3 1 if enough bright lamp pixels exist
    """
    vec = np.zeros(LIGHT_COLOR_DIM, np.float32)
    if frame_rgb is None or label_map is None:
        return vec
    ys, xs = np.where(label_map == 6)
    if len(ys) < 20:
        return vec
    H, W = frame_rgb.shape[:2]
    fx = np.clip((xs / scale).astype(np.int32), 0, W - 1)
    fy = np.clip((ys / scale).astype(np.int32), 0, H - 1)
    rgb = frame_rgb[fy, fx].astype(np.float32)
    value = rgb.max(axis=1)
    keep = value >= np.quantile(value, 0.9)
    bright = rgb[keep]
    if len(bright) < 8:
        return vec
    r, g, b = bright[:, 0], bright[:, 1], bright[:, 2]
    red = (r > g + 15) & (r > b + 15) & (r > 40)
    green = (g > r + 15) & (g > b + 15) & (g > 40)
    yellow = (r > 40) & (g > 40) & (b + 20 < np.minimum(r, g)) & (np.abs(r - g) < 40)
    n = float(len(bright))
    vec[0] = float(red.sum()) / n
    vec[1] = float(green.sum()) / n
    vec[2] = float(yellow.sum()) / n
    vec[3] = 1.0
    return vec


def attach_light_color(beh_seq, label_dir, key='light_color', cache_size=64, verbose=True):
    """4-d light color per frame; each unique frame is decoded once."""
    from functools import lru_cache
    from PIL import Image

    @lru_cache(maxsize=cache_size)
    def load_label(sid, vid, fr):
        f = os.path.join(label_dir, sid, vid, f'{fr:05d}.png')
        return np.asarray(Image.open(f)) if os.path.isfile(f) else None

    @lru_cache(maxsize=8)
    def load_frame(path):
        return np.asarray(Image.open(path).convert('RGB'))

    memo, out, missing, total = {}, [], 0, 0
    for paths in beh_seq['image']:
        track = []
        for p in paths:
            total += 1
            sid, vid, fr = _parse_frame_path(p)
            k = (sid, vid, fr)
            vec = memo.get(k)
            if vec is None:
                lm = load_label(sid, vid, fr)
                if lm is None or not os.path.isfile(p):
                    vec = [0.0] * LIGHT_COLOR_DIM
                    missing += 1
                else:
                    vec = traffic_light_color(load_frame(p), lm, None, lm.shape[1] / 1920.0).tolist()
                memo[k] = vec
            track.append(list(vec))
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[light_color] unique frames without label or image: {missing}; frames: {total}')
    return beh_seq

def attach_scene(beh_seq, label_dir, key='scene', cache_size=64, verbose=True):
    """Adds beh_seq[key]: per track, per frame, a 24-d scene vector from <label_dir>/<set>/<video>/<frame:05d>.png
    (uint8 Cityscapes trainId map, any resolution with the same 16:9 aspect).
    verbose=False skips the coverage report.
    """
    from functools import lru_cache
    from PIL import Image

    @lru_cache(maxsize=cache_size)
    def load(sid, vid, fr):
        f = os.path.join(label_dir, sid, vid, f'{fr:05d}.png')
        return np.asarray(Image.open(f)) if os.path.isfile(f) else None

    out, missing, total = [], 0, 0
    for i, paths in enumerate(beh_seq['image']):
        track = []
        for j, p in enumerate(paths):
            total += 1
            sid, vid, fr = _parse_frame_path(p)
            lm = load(sid, vid, fr)
            if lm is None:
                missing += 1
                track.append([0.0] * (6 * 4))
            else:
                track.append(scene_features(lm, beh_seq['bbox'][i][j], scale=lm.shape[1] / 1920.0))
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[scene] frames without label map: {missing}/{total}')
    return beh_seq


def attach_velocity(beh_seq, key='velocity'):
    """Per frame (dx, dy, dw, dh) of centre/size vs the previous frame, divided by the
    current bbox height. First frame = 0. Image-plane motion only, no ego-motion compensation.
    Call AFTER balancing/augmentation so it matches the final (jittered/mirrored/frozen) bbox."""
    out = []
    for boxes in beh_seq['bbox']:
        track, prev = [], None
        for x1, y1, x2, y2 in boxes:
            h = max(y2 - y1, 1.0)
            cur = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)
            track.append([0.0] * 4 if prev is None
                         else [(cur[i] - prev[i]) / h for i in range(4)])
            prev = cur
        out.append(track)
    beh_seq[key] = out
    return beh_seq

_FLOW_MODE_KEY = {'mean': 'flow_bbox', 'crop_full': 'flow_crop', 'ped_mask': 'flow_ped'}


def _atomic_save(final, **arrays):
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final[:-len('.npz')] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, final)


def _map_mean(flow, mode):
    """flow (H, W, 2) -> (2,) float32. ped_mask: mean over non-zero pixels, None if there are none."""
    if mode == 'ped_mask':
        nz = np.any(flow != 0, axis=-1)
        return flow[nz].mean(axis=0).astype(np.float32) if nz.any() else None
    return flow.reshape(-1, 2).mean(axis=0).astype(np.float32)


def _track_vectors(flow_root, mode, sid, vid, pid):
    """{frame: (2,) float32} for one track and one flow mode, or None if there is no data.
    'mean' is read directly. For the per-frame map modes the maps are reduced ONCE to
    <flow_root>/summary/<mode>/<set>/<video>/<ped_id>.npz, so later runs read one small file.
    If extraction was incomplete when a summary was built, delete <flow_root>/summary."""
    if mode == 'mean':
        f = os.path.join(flow_root, 'mean', sid, vid, f'{pid}.npz')
        if not os.path.isfile(f):
            return None
        z = np.load(f)
        frames, means, valid = z['frames'], z['mean'], z['valid']
        return {int(fr): means[r].astype(np.float32)
                for r, fr in enumerate(frames) if valid[r]}

    summary = os.path.join(flow_root, 'summary', mode, sid, vid, f'{pid}.npz')
    if os.path.isfile(summary):
        z = np.load(summary)
        frames, vec = z['frames'], z['vec']
        return {int(fr): vec[r] for r, fr in enumerate(frames)}
    folder = os.path.join(flow_root, mode, sid, vid, pid)
    if not os.path.isdir(folder):
        return None
    frames, vecs = [], []
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.npz') or name.endswith('.tmp.npz'):
            continue
        flow = np.load(os.path.join(folder, name))['flow'].astype(np.float32)
        v = _map_mean(flow, mode)
        if v is not None:
            frames.append(int(name[:-len('.npz')]))
            vecs.append(v)
    if not frames:
        return None
    _atomic_save(summary, frames=np.array(frames), vec=np.stack(vecs))
    return dict(zip(frames, vecs))


def attach_flow(beh_seq, flow_root, verbose=True):
    """Adds one 2-d per-frame key for every flow mode whose folder exists (source px / frame):
         mean      -> 'flow_bbox'  mean flow inside the bbox
         crop_full -> 'flow_crop'  mean flow over the whole bbox+context crop
         ped_mask  -> 'flow_ped'   mean flow over pedestrian pixels only
    Does NOT support augmentation copies (asserts). Zero vector = missing file, first frame of
    a track, or empty pedestrian mask. Returns the list of keys attached."""
    attached = []
    for mode, key in _FLOW_MODE_KEY.items():
        if not os.path.isdir(os.path.join(flow_root, mode)):
            print(f'[flow] {mode}: folder missing, {key} not attached')
            continue
        out, missing_tracks, zero_frames, total = [], 0, 0, 0
        for i, paths in enumerate(beh_seq['image']):
            pid = beh_seq['ped_id'][i][0][0]
            assert '__aug' not in pid, 'attach_flow does not support augmentation copies'
            sid, vid, _ = _parse_frame_path(paths[0])
            vectors = _track_vectors(flow_root, mode, sid, vid, pid)
            if vectors is None:
                missing_tracks += 1
            track = []
            for p in paths:
                total += 1
                vec = None if vectors is None else vectors.get(_parse_frame_path(p)[2])
                if vec is None:
                    vec = np.zeros(2, np.float32)
                    zero_frames += 1
                track.append(vec.tolist())
            out.append(track)
        beh_seq[key] = out
        attached.append(key)
        if verbose:
            print(f'[flow] {key} (mode={mode}): tracks without data {missing_tracks}/{len(out)}; '
                  f'zero frames {zero_frames}/{total} ({100 * zero_frames / max(total, 1):.1f}%)')
    return attached

def attach_flow_relative(beh_seq, verbose=True):
    """Derived residual flow features from already-attached flow_* keys.
    flow_rel_ped_crop = flow_ped - flow_crop
    flow_rel_bbox_crop = flow_bbox - flow_crop
    Requires attach_flow to have been called first.
    """
    required = ['flow_ped', 'flow_crop']
    if not all(k in beh_seq for k in required):
        print('[flow_relative] missing required flow keys; skipped')
        return beh_seq

    n_tracks = len(beh_seq['image'])
    rel_ped, rel_bbox = [], []
    for i in range(n_tracks):
        ped = np.asarray(beh_seq['flow_ped'][i], dtype=np.float32)
        crop = np.asarray(beh_seq['flow_crop'][i], dtype=np.float32)
        rel_ped.append((ped - crop).tolist())
        if 'flow_bbox' in beh_seq:
            bbox = np.asarray(beh_seq['flow_bbox'][i], dtype=np.float32)
            rel_bbox.append((bbox - crop).tolist())
        else:
            rel_bbox.append([[0.0, 0.0]] * len(ped))

    beh_seq['flow_rel_ped_crop'] = rel_ped
    beh_seq['flow_rel_bbox_crop'] = rel_bbox
    if verbose:
        print('[flow_relative] attached flow_rel_ped_crop and flow_rel_bbox_crop')
    return beh_seq



def attach_tte(beh_seq, key='tte', verbose=True):
    """Frames remaining until the track's last frame (the critical point).
    ANNOTATION-DERIVED FUTURE INFORMATION: unknown at prediction time, and its values encode
    the length of the whole track. Use ONLY in an explicitly labelled ablation.
    Call AFTER balancing so the values follow the final track length."""
    out = []
    for paths in beh_seq['image']:
        n = len(paths)
        out.append([float(n - 1 - j) for j in range(n)])
    beh_seq[key] = out
    if verbose:
        print(f'[tte] {key}: {len(out)} tracks (ablation only)')
    return beh_seq

_POSE_KEY_JOINTS = ('l_ank', 'r_ank', 'l_kne', 'r_kne', 'l_hip', 'r_hip', 'l_sho', 'r_sho')
_POSE_ANGLES = ((8, 'l_ank', 'l_kne', 'l_hip'), (9, 'r_ank', 'r_kne', 'r_hip'),
                (10, 'l_kne', 'l_hip', 'l_sho'), (11, 'r_kne', 'r_hip', 'r_sho'))


def attach_pose_dynamics(beh_seq, pose_dir, key='pose_dyn', min_conf=0.3,
                         min_bbox_height=0, verbose=True):
    """16-d per frame, coordinates = (xy - bbox centre) / bbox height (isotropic, angles are true).
    [0:8] joint speed (ankle, knee, hip, shoulder; L,R): only if the joint is valid in this AND the
    previous consecutive frame. [8:12] knee L/R, hip L/R angle / 180. [12:14] shoulder, hip width.
    [14:16] nose - shoulder midpoint. 0 = missing. Call BEFORE balancing; invalid for augmented copies."""
    key_idx = [KP[n] for n in _POSE_KEY_JOINTS]
    out, missing_tracks, empty, total = [], 0, 0, 0
    for i, paths in enumerate(beh_seq['image']):
        pid = beh_seq['ped_id'][i][0][0]
        assert '__aug' not in pid, 'attach_pose_dynamics must run before augmentation'
        sid, vid, _ = _parse_frame_path(paths[0])
        f = os.path.join(pose_dir, sid, vid, f'{pid}.npz')
        total += len(paths)
        if not os.path.isfile(f):
            missing_tracks += 1
            empty += len(paths)
            out.append([[0.0] * 16 for _ in paths])
            continue
        with np.load(f) as z:
            row_of = {int(fr): r for r, fr in enumerate(z['frames'])}
            kpts = z['kpts']
        track, prev, prev_fr = [], None, None
        for j, p in enumerate(paths):
            fr = _parse_frame_path(p)[2]
            x1, y1, x2, y2 = beh_seq['bbox'][i][j]
            h = max(y2 - y1, 1.0)
            vec = np.zeros(16, np.float32)
            r = row_of.get(fr)
            if r is None or (min_bbox_height > 0 and h < min_bbox_height):
                prev, prev_fr = None, None
                empty += 1
                track.append(vec.tolist())
                continue
            k = kpts[r]
            ok = k[:, 2] >= min_conf
            rel = (k[:, :2] - np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])) / h
            if not ok.any():
                empty += 1
            if prev is not None and fr - prev_fr == 1:
                p_rel, p_ok = prev
                for slot, ki in enumerate(key_idx):
                    if ok[ki] and p_ok[ki]:
                        vec[slot] = np.linalg.norm(rel[ki] - p_rel[ki])
            prev, prev_fr = (rel, ok), fr

            def get(name):
                return rel[KP[name]] if ok[KP[name]] else None

            for slot, a, b, cc in _POSE_ANGLES:
                pa, pb, pc = get(a), get(b), get(cc)
                if pa is not None and pb is not None and pc is not None:
                    vec[slot] = _angle(pa, pb, pc) / 180.0
            if ok[KP['l_sho']] and ok[KP['r_sho']]:
                vec[12] = np.linalg.norm(rel[KP['l_sho']] - rel[KP['r_sho']])
            if ok[KP['l_hip']] and ok[KP['r_hip']]:
                vec[13] = np.linalg.norm(rel[KP['l_hip']] - rel[KP['r_hip']])
            if ok[KP['nose']] and ok[KP['l_sho']] and ok[KP['r_sho']]:
                vec[14:16] = rel[KP['nose']] - 0.5 * (rel[KP['l_sho']] + rel[KP['r_sho']])
            track.append(vec.tolist())
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[pose_dyn] tracks without file: {missing_tracks}/{len(out)}; '
              f'frames with no usable keypoint: {empty}/{total} ({100 * empty / max(total, 1):.1f}%)')
    return beh_seq

POSE_CRITICAL_DIM = 16


def _kp(xy, ok, name):
    i = KP[name]
    return xy[i] if ok[i] else None


def _mean_angle(xy, ok, triples):
    vals = []
    for a, b, c in triples:
        pa, pb, pc = _kp(xy, ok, a), _kp(xy, ok, b), _kp(xy, ok, c)
        if pa is not None and pb is not None and pc is not None:
            vals.append(_angle(pa, pb, pc) / 180.0)
    return float(np.mean(vals)) if vals else 0.0


def pose_critical_frame(xy, conf, ok, bbox_h, prev_ankles=None):
    """16-d body configuration. Geometry only: no sitting / phone / talking label.

    Scale is shoulder width, so a tight or loose bbox does not erase a folded leg.
    0 is missing or a real zero. Slot 10 is 10 when a wrist or the nose is absent,
    because 0 there would mean the hand is on the nose.
      0 torso uprightness in [-1, 1]: +1 hip-to-shoulder points up the image
      1 mean knee angle / 180 (straight ~ 1)
      2 mean hip angle / 180
      3 mean elbow angle / 180
      4 vertical leg / vertical torso (folded legs -> smaller)
      5 |left ankle x - right ankle x| / shoulder width
      6 |left ankle y - right ankle y| / shoulder width
      7 (nose x - shoulder mid x) / shoulder width
      8 (nose y - shoulder mid y) / shoulder width; image y grows downward
      9 left-ear confidence - right-ear confidence (head-turn proxy, not gaze)
     10 min wrist-to-nose distance / shoulder width, or 10 if unavailable
     11 |left wrist y - right wrist y| / shoulder width
     12 (shoulder mid x - hip mid x) / shoulder width
     13 mean ankle speed since the previous consecutive frame, in bbox heights
     14 fraction of the 17 joints above the confidence gate
     15 1 if both shoulders, both hips and at least one ankle are present, else 0
    When slot 15 is 0 the whole vector is 0.
    """
    vec = np.zeros(POSE_CRITICAL_DIM, np.float32)
    ls, rs = _kp(xy, ok, 'l_sho'), _kp(xy, ok, 'r_sho')
    lh, rh = _kp(xy, ok, 'l_hip'), _kp(xy, ok, 'r_hip')
    if ls is None or rs is None or lh is None or rh is None:
        return vec
    sw = float(np.linalg.norm(ls - rs))
    if sw < 1e-3:
        return vec
    sho = 0.5 * (ls + rs)
    hip = 0.5 * (lh + rh)
    torso = sho - hip
    tlen = float(np.linalg.norm(torso))
    if tlen < 1e-3:
        return vec

    vec[0] = np.clip(-torso[1] / tlen, -1.0, 1.0)
    vec[1] = _mean_angle(xy, ok, (('l_ank', 'l_kne', 'l_hip'), ('r_ank', 'r_kne', 'r_hip')))
    vec[2] = _mean_angle(xy, ok, (('l_kne', 'l_hip', 'l_sho'), ('r_kne', 'r_hip', 'r_sho')))
    vec[3] = _mean_angle(xy, ok, (('l_wri', 'l_elb', 'l_sho'), ('r_wri', 'r_elb', 'r_sho')))

    torso_v = abs(float(sho[1] - hip[1]))
    leg_v = []
    for ank, hp in ((_kp(xy, ok, 'l_ank'), lh), (_kp(xy, ok, 'r_ank'), rh)):
        if ank is not None:
            leg_v.append(abs(float(ank[1] - hp[1])))
    if leg_v and torso_v > 1e-3:
        vec[4] = float(np.clip(np.mean(leg_v) / torso_v, 0.0, 4.0))

    la, ra = _kp(xy, ok, 'l_ank'), _kp(xy, ok, 'r_ank')
    if la is not None and ra is not None:
        vec[5] = float(np.clip(abs(la[0] - ra[0]) / sw, 0.0, 6.0))
        vec[6] = float(np.clip(abs(la[1] - ra[1]) / sw, 0.0, 6.0))

    nose = _kp(xy, ok, 'nose')
    if nose is not None:
        vec[7] = float(np.clip((nose[0] - sho[0]) / sw, -4.0, 4.0))
        vec[8] = float(np.clip((nose[1] - sho[1]) / sw, -4.0, 4.0))
    vec[9] = float(np.clip(conf[KP['l_ear']] - conf[KP['r_ear']], -1.0, 1.0))

    wrist_d = []
    if nose is not None:
        for name in ('l_wri', 'r_wri'):
            w = _kp(xy, ok, name)
            if w is not None:
                wrist_d.append(float(np.linalg.norm(w - nose) / sw))
    vec[10] = float(np.clip(min(wrist_d), 0.0, 10.0)) if wrist_d else 10.0

    lw, rw = _kp(xy, ok, 'l_wri'), _kp(xy, ok, 'r_wri')
    if lw is not None and rw is not None:
        vec[11] = float(np.clip(abs(lw[1] - rw[1]) / sw, 0.0, 6.0))
    vec[12] = float(np.clip((sho[0] - hip[0]) / sw, -4.0, 4.0))

    if prev_ankles is not None and bbox_h > 1e-3:
        speeds = []
        for name, slot in (('l_ank', 0), ('r_ank', 1)):
            cur = _kp(xy, ok, name)
            prv = prev_ankles[slot]
            if cur is not None and prv is not None:
                speeds.append(float(np.linalg.norm(cur - prv) / bbox_h))
        if speeds:
            vec[13] = float(np.clip(np.mean(speeds), 0.0, 5.0))

    vec[14] = float(np.mean(ok))
    vec[15] = 1.0 if (la is not None or ra is not None) else 0.0
    if vec[15] == 0.0:
        return np.zeros(POSE_CRITICAL_DIM, np.float32)
    return vec


def attach_pose_critical(beh_seq, pose_dir, key='pose_critical', min_conf=0.3,
                         min_bbox_height=0, verbose=True):
    """16-d pose_critical per frame. Same npz files as attach_pose. Not valid for augmented copies."""
    out, missing_tracks, empty, total = [], 0, 0, 0
    for i, paths in enumerate(beh_seq['image']):
        pid = beh_seq['ped_id'][i][0][0]
        assert '__aug' not in pid, 'attach_pose_critical must run before augmentation'
        sid, vid, _ = _parse_frame_path(paths[0])
        f = os.path.join(pose_dir, sid, vid, f'{pid}.npz')
        total += len(paths)
        if not os.path.isfile(f):
            missing_tracks += 1
            empty += len(paths)
            out.append([[0.0] * POSE_CRITICAL_DIM for _ in paths])
            continue
        with np.load(f) as z:
            row_of = {int(fr): r for r, fr in enumerate(z['frames'])}
            kpts = z['kpts']
        track, prev_ank, prev_fr = [], None, None
        for j, p in enumerate(paths):
            fr = _parse_frame_path(p)[2]
            x1, y1, x2, y2 = [float(v) for v in beh_seq['bbox'][i][j]]
            h = max(y2 - y1, 1.0)
            r = row_of.get(fr)
            small = min_bbox_height > 0 and h < min_bbox_height
            if r is None or small:
                empty += 1
                track.append([0.0] * POSE_CRITICAL_DIM)
                prev_ank, prev_fr = None, None
                continue
            k = kpts[r]
            ok = k[:, 2] >= min_conf
            use_prev = prev_ank if (prev_fr is not None and fr - prev_fr == 1) else None
            vec = pose_critical_frame(k[:, :2], k[:, 2], ok, h, prev_ankles=use_prev)
            if vec[15] == 0.0:
                empty += 1
            track.append(vec.tolist())
            prev_ank = [k[KP['l_ank'], :2].copy() if ok[KP['l_ank']] else None,
                        k[KP['r_ank'], :2].copy() if ok[KP['r_ank']] else None]
            prev_fr = fr
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[pose_critical] tracks without file: {missing_tracks}/{len(out)}; '
              f'frames with no usable torso/ankle: {empty}/{total} '
              f'({100 * empty / max(total, 1):.1f}%)')
    return beh_seq

_RES_DIM = 7
_RES_MIN_PIXELS = 20


def _res_vector(ped, crop):
    """ped, crop: (H, W, 2) float32 from ped_mask / crop_full. Returns ((7,) float32, inconsistent flag).
    [0:2] mean ped flow - mean flow of the NON-person pixels of the crop; [2:4] std of flow inside the person;
    [4] |upper-body mean - background|, [5] |lower-body mean - background| (split at the mask's own mid row);
    [6] valid flag (person mask has enough pixels)."""
    vec = np.zeros(_RES_DIM, np.float32)
    mask = np.any(ped != 0, axis=-1)
    if int(mask.sum()) < _RES_MIN_PIXELS:
        return vec, False
    bad = bool(np.abs(ped[mask] - crop[mask]).max() > 0.05)
    bg = crop[~mask].mean(axis=0) if (~mask).any() else crop.reshape(-1, 2).mean(axis=0)
    vec[0:2] = ped[mask].mean(axis=0) - bg
    vec[2:4] = ped[mask].std(axis=0)
    ys = np.nonzero(mask)[0]
    mid = 0.5 * (ys.min() + ys.max())
    rows = np.arange(mask.shape[0])[:, None]
    for slot, part in ((4, mask & (rows < mid)), (5, mask & (rows >= mid))):
        if part.any():
            vec[slot] = np.linalg.norm(ped[part].mean(axis=0) - bg)
    vec[6] = 1.0
    return vec, bad


def _res_track_vectors(flow_root, sid, vid, pid):
    """{frame: ((7,) vec, bad)} for one track, or None. Reduced once to <flow_root>/summary/res_v1/...
    If extraction was incomplete when a summary was built, delete <flow_root>/summary/res_v1."""
    summary = os.path.join(flow_root, 'summary', 'res_v1', sid, vid, f'{pid}.npz')
    if os.path.isfile(summary):
        with np.load(summary) as z:
            return {int(fr): (z['vec'][r], bool(z['bad'][r])) for r, fr in enumerate(z['frames'])}
    ped_dir = os.path.join(flow_root, 'ped_mask', sid, vid, pid)
    crop_dir = os.path.join(flow_root, 'crop_full', sid, vid, pid)
    if not (os.path.isdir(ped_dir) and os.path.isdir(crop_dir)):
        return None
    frames, vecs, bads = [], [], []
    for name in sorted(os.listdir(ped_dir)):
        crop_file = os.path.join(crop_dir, name)
        if not name.endswith('.npz') or name.endswith('.tmp.npz') or not os.path.isfile(crop_file):
            continue
        with np.load(os.path.join(ped_dir, name)) as z:
            ped = z['flow'].astype(np.float32)
        with np.load(crop_file) as z:
            crop = z['flow'].astype(np.float32)
        if ped.shape != crop.shape:
            continue
        vec, bad = _res_vector(ped, crop)
        frames.append(int(name[:-len('.npz')]))
        vecs.append(vec)
        bads.append(bad)
    if not frames:
        return None
    _atomic_save(summary, frames=np.array(frames), vec=np.stack(vecs), bad=np.array(bads))
    return dict(zip(frames, zip(vecs, bads)))


def attach_flow_residual_stats(beh_seq, flow_root, key='flow_res', verbose=True):
    """Adds ONE 7-d key (see _res_vector). Needs flow_root/crop_full and flow_root/ped_mask.
    Zero vector with valid flag 0 = missing file, first frame of a track, or empty person mask.
    Not valid for augmented copies. Contains camera ego-motion cues: leakage-prone configuration."""
    for sub in ('crop_full', 'ped_mask'):
        if not os.path.isdir(os.path.join(flow_root, sub)):
            print(f'[flow_res] {sub} folder missing, {key} not attached')
            return beh_seq
    out, missing_tracks, zero_frames, bad_frames, total = [], 0, 0, 0, 0
    for i, paths in enumerate(beh_seq['image']):
        pid = beh_seq['ped_id'][i][0][0]
        assert '__aug' not in pid, 'attach_flow_residual_stats does not support augmentation copies'
        sid, vid, _ = _parse_frame_path(paths[0])
        vectors = _res_track_vectors(flow_root, sid, vid, pid)
        if vectors is None:
            missing_tracks += 1
        track = []
        for p in paths:
            total += 1
            item = None if vectors is None else vectors.get(_parse_frame_path(p)[2])
            if item is None:
                vec = np.zeros(_RES_DIM, np.float32)
                zero_frames += 1
            else:
                vec, bad = item
                zero_frames += int(vec[6] == 0)
                bad_frames += int(bad)
            track.append(vec.tolist())
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[flow_res] tracks without data {missing_tracks}/{len(out)}; '
              f'invalid frames {zero_frames}/{total} ({100 * zero_frames / max(total, 1):.1f}%); '
              f'ped vs crop disagree on {bad_frames} frames (should be 0)')
    return beh_seq

FLOW_CRITICAL_DIM = 8


def _flow_critical_vector(ped, crop):
    """8-d motion relation inside one frame. Not an action class.

    ped, crop: (H, W, 2) float32.
      0 magnitude of (lower-body mean - background)
      1 magnitude of (upper-body mean - background)
      2 lower / (upper + 1e-3), clipped to 20
      3-4 lower-body residual flow (x, y), pixels/frame
      5 |left-half mean - right-half mean| inside the lower body
      6 fraction of person pixels whose flow agrees with the person mean (cosine > 0.5)
      7 1 if the person mask has enough pixels
    """
    vec = np.zeros(FLOW_CRITICAL_DIM, np.float32)
    mask = np.any(ped != 0, axis=-1)
    if int(mask.sum()) < _RES_MIN_PIXELS:
        return vec
    bg = crop[~mask].mean(axis=0) if (~mask).any() else crop.reshape(-1, 2).mean(axis=0)
    ys = np.nonzero(mask)[0]
    mid = 0.5 * (ys.min() + ys.max())
    rows = np.arange(mask.shape[0])[:, None]
    lower = mask & (rows >= mid)
    upper = mask & (rows < mid)
    if not lower.any():
        return vec
    low_res = ped[lower].mean(axis=0) - bg
    vec[0] = float(np.linalg.norm(low_res))
    vec[3:5] = low_res
    if upper.any():
        up_res = ped[upper].mean(axis=0) - bg
        vec[1] = float(np.linalg.norm(up_res))
    vec[2] = float(np.clip(vec[0] / (vec[1] + 1e-3), 0.0, 20.0))
    xs = np.nonzero(lower)[1]
    split = np.median(xs)
    cols = np.arange(mask.shape[1])[None, :]
    left = lower & (cols <= split)
    right = lower & (cols > split)
    if left.any() and right.any():
        vec[5] = float(np.linalg.norm(ped[left].mean(axis=0) - ped[right].mean(axis=0)))
    person = ped[mask]
    mean = person.mean(axis=0)
    mn = float(np.linalg.norm(mean))
    if mn > 1e-3:
        norms = np.linalg.norm(person, axis=1)
        cos = (person @ mean) / (np.maximum(norms, 1e-3) * mn)
        vec[6] = float(np.mean(cos > 0.5))
    vec[7] = 1.0
    return vec


def _flow_critical_track(flow_root, sid, vid, pid):
    """{frame: (8,) vec} or None. Cached under summary/flowcrit_v1, separate from res_v1."""
    summary = os.path.join(flow_root, 'summary', 'flowcrit_v1', sid, vid, f'{pid}.npz')
    if os.path.isfile(summary):
        with np.load(summary) as z:
            return {int(fr): z['vec'][r] for r, fr in enumerate(z['frames'])}
    ped_dir = os.path.join(flow_root, 'ped_mask', sid, vid, pid)
    crop_dir = os.path.join(flow_root, 'crop_full', sid, vid, pid)
    if not (os.path.isdir(ped_dir) and os.path.isdir(crop_dir)):
        return None
    frames, vecs = [], []
    for name in sorted(os.listdir(ped_dir)):
        crop_file = os.path.join(crop_dir, name)
        if not name.endswith('.npz') or name.endswith('.tmp.npz') or not os.path.isfile(crop_file):
            continue
        with np.load(os.path.join(ped_dir, name)) as z:
            ped = z['flow'].astype(np.float32)
        with np.load(crop_file) as z:
            crop = z['flow'].astype(np.float32)
        if ped.shape != crop.shape:
            continue
        frames.append(int(name[:-len('.npz')]))
        vecs.append(_flow_critical_vector(ped, crop))
    if not frames:
        return None
    _atomic_save(summary, frames=np.array(frames), vec=np.stack(vecs))
    return dict(zip(frames, vecs))


def attach_flow_critical(beh_seq, flow_root, key='flow_critical', verbose=True):
    """8-d flow_critical. Needs crop_full and ped_mask. Not valid for augmented copies."""
    for sub in ('crop_full', 'ped_mask'):
        if not os.path.isdir(os.path.join(flow_root, sub)):
            print(f'[flow_critical] {sub} folder missing, {key} not attached')
            return beh_seq
    out, missing_tracks, zero_frames, total = [], 0, 0, 0
    for i, paths in enumerate(beh_seq['image']):
        pid = beh_seq['ped_id'][i][0][0]
        assert '__aug' not in pid, 'attach_flow_critical does not support augmentation copies'
        sid, vid, _ = _parse_frame_path(paths[0])
        vectors = _flow_critical_track(flow_root, sid, vid, pid)
        if vectors is None:
            missing_tracks += 1
        track = []
        for p in paths:
            total += 1
            vec = None if vectors is None else vectors.get(_parse_frame_path(p)[2])
            if vec is None:
                vec = np.zeros(FLOW_CRITICAL_DIM, np.float32)
                zero_frames += 1
            else:
                zero_frames += int(np.asarray(vec)[7] == 0)
            track.append(np.asarray(vec, np.float32).tolist())
        out.append(track)
    beh_seq[key] = out
    if verbose:
        print(f'[flow_critical] tracks without data {missing_tracks}/{len(out)}; '
              f'invalid frames {zero_frames}/{total} ({100 * zero_frames / max(total, 1):.1f}%)')
    return beh_seq