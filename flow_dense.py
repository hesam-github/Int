"""Resized crop_full flow for one branch. Does not touch pose, scene, or bbox.

Each saved map is (H, W, 2) in source pixels per frame. It is resized to
32x32 and clipped to [-40, 40] / 40. The 8-d summary is not computed.
This is not a Kinetics I3D. A ConvLSTM sees the field; the existing branch
LSTM then runs in time.
"""
import os
import numpy as np
from PIL import Image

SIDE = 32
CLIP = 40.0


def _ped_id(ped_window):
    flat = np.asarray(ped_window, dtype=object).reshape(-1)
    return str(flat[0])


def _frame_file(flow_root, image_path, ped_id):
    parts = os.path.normpath(image_path).split(os.sep)
    set_id, video, frame = parts[-3], parts[-2], int(os.path.splitext(parts[-1])[0])
    return os.path.join(flow_root, 'crop_full', set_id, video, ped_id, f'{frame:05d}.npz')


def _load_one(path):
    out = np.zeros((SIDE, SIDE, 2), np.float32)
    if not os.path.isfile(path):
        return out, False
    with np.load(path) as z:
        flow = z['flow'].astype(np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2 or flow.shape[0] < 2 or flow.shape[1] < 2:
        return out, False
    for c in range(2):
        channel = Image.fromarray(flow[..., c]).resize((SIDE, SIDE), Image.BILINEAR)
        out[..., c] = np.asarray(channel, np.float32)
    return np.clip(out, -CLIP, CLIP) / CLIP, True


def ensure_flow_volume(images, ped_ids, flow_root, cache_path):
    """(N, T, 32, 32, 2) float32, cached. Missing frames stay zero."""
    n = len(images)
    t = len(images[0])
    shape = (n, t, SIDE, SIDE, 2)
    if os.path.isfile(cache_path):
        cached = np.load(cache_path, mmap_mode='r')
        if cached.shape == shape:
            print(f'[flow dense] cache hit {os.path.basename(cache_path)} {shape}')
            return cached
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + '.tmp.npy'
    mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=shape)
    missing = 0
    total = n * t
    for i, window in enumerate(images):
        pid = _ped_id(ped_ids[i])
        for j, image_path in enumerate(window):
            frame, ok = _load_one(_frame_file(flow_root, image_path, pid))
            mm[i, j] = frame.astype(np.float16)
            missing += int(not ok)
    mm.flush()
    os.replace(tmp, cache_path)
    print(f'[flow dense] wrote {cache_path}')
    print(f'[flow dense] missing frames {missing}/{total} ({100.0 * missing / max(total, 1):.1f}%)')
    return np.load(cache_path, mmap_mode='r')