"""extract_pose.py -- run ONCE, in a separate env that has ultralytics (NOT pie_tf).

Reads the same cached track sequences as training and writes one .npz per pedestrian track:
  <PIE_PATH>/extra/pose/<set>/<video>/<ped_id>.npz   (frames, kpts[T,17,3] in absolute pixels)
"""
import os
import sys
import numpy as np
from PIL import Image

PIE_PATH = r'D:\PIE codes\PIE_dataset'
CODE_DIR = r'D:\PIE codes\PIE'          # folder that contains pie_data_Hesam.py, utils.py, extra_features.py
OUT_DIR = os.path.join(PIE_PATH, 'extra', 'pose')
MODEL_NAME = 'yolov8m-pose.pt'          # downloaded automatically on first use
LIMIT_TRACKS = None                        # smoke test first; set to None for the full run

os.environ['PIE_PATH'] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from extra_features import _parse_frame_path
from utils import jitter_bbox


def pose_for_track(model, paths, boxes, out_dir, ped_id):
    frames, out = [], []
    for p, b in zip(paths, boxes):
        k = np.zeros((17, 3), np.float32)
        x1, y1, x2, y2 = [int(v) for v in jitter_bbox([list(b)], p, 'enlarge', 0.5)[0]]
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            crop = np.asarray(Image.open(p).convert('RGB').crop((x1, y1, x2, y2)))
            crop = np.ascontiguousarray(crop[:, :, ::-1])      # ultralytics expects BGR for numpy input
            res = model.predict(crop, imgsz=256, verbose=False)[0]
            if res.keypoints is not None and len(res.boxes) > 0:
                ctr = np.array([(x2 - x1) / 2, (y2 - y1) / 2])
                i = int(np.argmin(np.linalg.norm(res.boxes.xywh[:, :2].cpu().numpy() - ctr, axis=1)))
                xy = res.keypoints.xy[i].cpu().numpy()
                k[:, 0], k[:, 1] = xy[:, 0] + x1, xy[:, 1] + y1
                k[:, 2] = res.keypoints.conf[i].cpu().numpy()
        frames.append(_parse_frame_path(p)[2])
        out.append(k)
    sid, vid, _ = _parse_frame_path(paths[0])
    folder = os.path.join(out_dir, sid, vid)
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, f'{ped_id}.npz')
    tmp = final[:-len('.npz')] + '.tmp.npz'
    np.savez(tmp, frames=np.array(frames), kpts=np.stack(out))
    os.replace(tmp, final)                                      # never leaves a half-written file


def main():
    from ultralytics import YOLO
    from pie_data_Hesam import PIE

    model = YOLO(MODEL_NAME)
    imdb = PIE(data_path=PIE_PATH)
    done = skipped = 0
    for split in ('train', 'val', 'test'):
        # same parameters as training -> same cache key -> "Loading cached sequence data ..."
        seq = imdb.generate_data_trajectory_sequence(split, seq_type='intention', min_track_size=15)
        n = len(seq['image'])
        for i in range(n):
            if LIMIT_TRACKS is not None and i >= LIMIT_TRACKS:
                break
            pid = seq['ped_id'][i][0][0]
            sid, vid, _ = _parse_frame_path(seq['image'][i][0])
            if os.path.isfile(os.path.join(OUT_DIR, sid, vid, f'{pid}.npz')):
                skipped += 1
                continue
            pose_for_track(model, seq['image'][i], seq['bbox'][i], OUT_DIR, pid)
            done += 1
            if done % 20 == 0:
                print(f'[{split}] track {i + 1}/{n}  (done {done}, skipped {skipped})', flush=True)
    print(f'finished: {done} tracks written, {skipped} already existed -> {OUT_DIR}')


if __name__ == '__main__':
    main()