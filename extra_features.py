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