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