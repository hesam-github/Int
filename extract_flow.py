"""extract_flow.py -- run ONCE, offline. Three modes (set FLOW_MODE).

  'mean'       : per-track file with the 2-d mean flow inside the bbox (compact, default)
  'crop_full'  : full flow map on the bbox+context crop, one file per frame
  'ped_mask'   : same, but zero outside the 'person' class of the scene label map

Flow at frame j is computed between frame j-1 and frame j, both cropped with the bbox of
frame j, so it measures motion relative to a fixed window. Units: source-image pixels per frame.

Output:
  mean       : <OUT_ROOT>/mean/<set>/<video>/<ped_id>.npz   (frames[T], mean[T,2], valid[T])
  other modes: <OUT_ROOT>/<mode>/<set>/<video>/<ped_id>/<frame:05d>.npz   (flow[H,W,2])
Row/frame 0 of every track has no previous frame: mean = 0, valid = False.
"""
import os
import sys
import numpy as np
from PIL import Image

PIE_PATH = r'D:\PIE codes\PIE_dataset'
CODE_DIR = r'D:\PIE codes\PIE'
OUT_ROOT = os.path.join(PIE_PATH, 'extra', 'flow')
SCENE_DIR = os.path.join(PIE_PATH, 'extra', 'scene')   # only for ped_mask mode
FLOW_MODE = 'crop_full'          # 'mean' | 'crop_full' | 'ped_mask'
LIMIT_TRACKS = None         # smoke test: e.g. 5; None = full run
USE_RAFT = True
CONTEXT_MARGIN = 0.5        # context added on each side, as a fraction of bbox w/h
MIN_CROP = 128              # minimum crop side in SOURCE pixels
RAFT_MIN_SIDE = 128         # RAFT hard minimum: 8 * 16 (short side of the network input)
RAFT_MAX_SIDE = 256         # soft cap on the long side

assert FLOW_MODE in ('mean', 'crop_full', 'ped_mask')
os.environ['PIE_PATH'] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from extra_features import _parse_frame_path


def _load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))      # np.array: writable copy


def _crop_around_bbox(img, bbox, margin, min_side=MIN_CROP):
    """bbox + margin context, never smaller than min_side x min_side.
    The window is shifted (not clipped) at image borders, so its size is preserved."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cw = int(min(w, max((1 + 2 * margin) * bw, min_side)))
    ch = int(min(h, max((1 + 2 * margin) * bh, min_side)))
    xa = int(np.clip(round(cx - cw / 2.0), 0, w - cw))
    ya = int(np.clip(round(cy - ch / 2.0), 0, h - ch))
    return img[ya:ya + ch, xa:xa + cw], (xa, ya)


def _target_size(h, w, min_side, max_side):
    """Network input size: short side >= min_side (hard), long side <= max_side (soft,
    only applied while the short side stays >= min_side). Multiples of 8."""
    scale = 1.0
    if min(h, w) < min_side:
        scale = min_side / float(min(h, w))
    if max(h, w) * scale > max_side:
        capped = max_side / float(max(h, w))
        if min(h, w) * capped >= min_side:
            scale = capped
    new_w = max(min_side, int(round(w * scale / 8.0)) * 8)
    new_h = max(min_side, int(round(h * scale / 8.0)) * 8)
    return new_w, new_h


def _resize(img, new_w, new_h):
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def compute_flow_raft(prev_crop, curr_crop, model, device):
    import torch
    import torchvision.transforms.functional as TF
    H, W = prev_crop.shape[:2]
    new_w, new_h = _target_size(H, W, RAFT_MIN_SIDE, RAFT_MAX_SIDE)
    p = _resize(prev_crop, new_w, new_h)
    c = _resize(curr_crop, new_w, new_h)                  # same size as p by construction
    t1 = TF.to_tensor(p).unsqueeze(0).to(device)
    t2 = TF.to_tensor(c).unsqueeze(0).to(device)
    with torch.no_grad():
        flow_small = model(t1, t2)[-1][0].permute(1, 2, 0).cpu().numpy()   # (new_h, new_w, 2)
    fx = np.asarray(Image.fromarray(flow_small[..., 0].astype(np.float32)).resize((W, H), Image.BILINEAR))
    fy = np.asarray(Image.fromarray(flow_small[..., 1].astype(np.float32)).resize((W, H), Image.BILINEAR))
    sx, sy = new_w / float(W), new_h / float(H)           # x and y scales differ
    return np.stack([fx / sx, fy / sy], axis=-1)


def compute_flow_farneback(prev_crop, curr_crop):
    import cv2
    g1 = cv2.cvtColor(prev_crop, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(curr_crop, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None, pyr_scale=0.5, levels=3, winsize=11,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0)


def flow_for_pair(prev_crop, curr_crop, state):
    """RAFT if available; on CUDA OOM the run falls back to Farneback for the rest of the run."""
    if state['raft'] is not None:
        try:
            return compute_flow_raft(prev_crop, curr_crop, state['raft'], state['device'])
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            print('[flow] CUDA OOM -> Farneback for the rest of the run')
            state['raft'] = None
            if state['device'] == 'cuda':
                import torch
                torch.cuda.empty_cache()
    return compute_flow_farneback(prev_crop, curr_crop)


def load_ped_mask(sid, vid, frame, crop_shape, crop_origin, scene_dir):
    """Boolean mask (crop size) of Cityscapes class 'person' (trainId 11), or None."""
    f = os.path.join(scene_dir, sid, vid, f'{frame:05d}.png')
    if not os.path.isfile(f):
        return None
    lm = np.asarray(Image.open(f))
    xa, ya = crop_origin
    h, w = crop_shape[:2]
    if lm.shape[1] != 1920:                               # label map saved at lower resolution
        s = lm.shape[1] / 1920.0
        xa, ya, w, h = int(xa * s), int(ya * s), int(w * s), int(h * s)
    patch = lm[max(0, ya):ya + h, max(0, xa):xa + w]
    if patch.size == 0:
        return None
    mask = patch == 11
    if mask.shape != tuple(crop_shape[:2]):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (crop_shape[1], crop_shape[0]), Image.NEAREST)) > 127
    return mask


def mean_inside_bbox_on_crop(flow, bbox, crop_origin):
    xa, ya = crop_origin
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1c, y1c = max(0, x1 - xa), max(0, y1 - ya)
    x2c, y2c = min(flow.shape[1], x2 - xa), min(flow.shape[0], y2 - ya)
    if x2c <= x1c or y2c <= y1c:
        return np.zeros(2, np.float32)
    return flow[y1c:y2c, x1c:x2c].mean(axis=(0, 1)).astype(np.float32)


def _atomic_savez(final, **arrays):
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final[:-len('.npz')] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, final)                                # never leaves a half-written file


def track_done(sid, vid, pid, frames):
    if FLOW_MODE == 'mean':
        return os.path.isfile(os.path.join(OUT_ROOT, 'mean', sid, vid, f'{pid}.npz'))
    folder = os.path.join(OUT_ROOT, FLOW_MODE, sid, vid, pid)
    return all(os.path.isfile(os.path.join(folder, f'{fr:05d}.npz')) for fr in frames[1:])


def process_track(paths, boxes, pid, state):
    sid, vid, _ = _parse_frame_path(paths[0])
    frames = [_parse_frame_path(p)[2] for p in paths]
    means = np.zeros((len(paths), 2), np.float32)
    valid = np.zeros(len(paths), bool)
    prev_full = _load_rgb(paths[0])
    for j in range(1, len(paths)):
        curr_full = _load_rgb(paths[j])
        bbox = boxes[j]
        prev_crop, origin = _crop_around_bbox(prev_full, bbox, CONTEXT_MARGIN)
        curr_crop, _ = _crop_around_bbox(curr_full, bbox, CONTEXT_MARGIN)
        flow = flow_for_pair(prev_crop, curr_crop, state)          # (h, w, 2), source px / frame
        if FLOW_MODE == 'mean':
            means[j] = mean_inside_bbox_on_crop(flow, bbox, origin)
            valid[j] = True
        else:
            if FLOW_MODE == 'ped_mask':
                mask = load_ped_mask(sid, vid, frames[j], flow.shape, origin, SCENE_DIR)
                flow = flow * mask[..., None] if mask is not None else np.zeros_like(flow)
            _atomic_savez(os.path.join(OUT_ROOT, FLOW_MODE, sid, vid, pid, f'{frames[j]:05d}.npz'),
                          flow=flow.astype(np.float16))
        prev_full = curr_full
    if FLOW_MODE == 'mean':
        _atomic_savez(os.path.join(OUT_ROOT, 'mean', sid, vid, f'{pid}.npz'),
                      frames=np.array(frames), mean=means, valid=valid)


def main():
    from pie_data_Hesam import PIE
    state = {'raft': None, 'device': None}
    if USE_RAFT:
        try:
            import torch
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            state['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            state['raft'] = raft_small(weights=Raft_Small_Weights.DEFAULT).to(state['device']).eval()
            print(f"[flow] mode={FLOW_MODE}  RAFT-small on {state['device']}")
        except Exception as e:
            print(f'[flow] RAFT unavailable ({e}), falling back to Farneback')

    imdb = PIE(data_path=PIE_PATH)
    done = skipped = 0
    for split in ('train', 'val', 'test'):
        # same parameters as training -> same cache key as extract_pose.py
        seq = imdb.generate_data_trajectory_sequence(split, seq_type='intention', min_track_size=15)
        n = len(seq['image'])
        for i in range(n):
            if LIMIT_TRACKS is not None and i >= LIMIT_TRACKS:
                break
            paths, boxes = seq['image'][i], seq['bbox'][i]
            pid = seq['ped_id'][i][0][0]
            sid, vid, _ = _parse_frame_path(paths[0])
            frames = [_parse_frame_path(p)[2] for p in paths]
            if track_done(sid, vid, pid, frames):
                skipped += 1
                continue
            process_track(paths, boxes, pid, state)
            done += 1
            if done % 10 == 0:
                print(f'[{split}] track {i + 1}/{n}  (done {done}, skipped {skipped})', flush=True)
    print(f'finished mode={FLOW_MODE}: {done} tracks written, {skipped} skipped -> {OUT_ROOT}')


if __name__ == '__main__':
    main()