"""extract_scene.py -- run ONCE (same env as extract_pose.py, plus `pip install transformers`).

Writes one uint8 Cityscapes trainId map per unique frame used by any track:
  <PIE_PATH>/extra/scene/<set>/<video>/<frame:05d>.png     (OUT_W x OUT_H, 16:9)
which is exactly what extra_features.attach_scene() reads. Resumable: existing files are skipped.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PIE_PATH = r'D:\PIE codes\PIE_dataset'
CODE_DIR = r'D:\PIE codes\PIE'            # contains pie_data_Hesam.py, extra_features.py
OUT_DIR = os.path.join(PIE_PATH, 'extra', 'scene')
PREVIEW_DIR = os.path.join(PIE_PATH, 'extra', 'scene_preview')

MODEL_ID = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'
INFER_W, INFER_H = 1024, 576               # network input (multiples of 32, 16:9); try 1536x864 if time allows
OUT_W, OUT_H = 960, 540                    # saved label map (attach_scene derives scale = width / 1920)
SRC_W, SRC_H = 1920, 1080
MIN_TRACK_SIZE = 15                        # must match training / extract_pose.py
BATCH = 4                                  # lower to 2 if CUDA runs out of memory
NUM_WORKERS = 2
LIMIT_FRAMES = None                          # smoke test (spread over the dataset); None = full run
PREVIEW_N = 8
WARMUP_BATCHES = 3

os.environ['PIE_PATH'] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from extra_features import _GROUPS, _parse_frame_path

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle']
PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (119, 11, 32)], np.uint8)


def out_path(key):
    sid, vid, fr = key
    return os.path.join(OUT_DIR, sid, vid, f'{fr:05d}.png')


def collect_frames():
    """Unique (set, video, frame) -> image path over all splits, using the same cached sequences as training."""
    from pie_data_Hesam import PIE
    imdb = PIE(data_path=PIE_PATH)
    frames = {}
    for split in ('train', 'val', 'test'):
        seq = imdb.generate_data_trajectory_sequence(split, seq_type='intention', min_track_size=MIN_TRACK_SIZE)
        before = len(frames)
        for paths in seq['image']:
            for p in paths:
                frames[_parse_frame_path(p)] = p
        print(f'[{split}] tracks: {len(seq["image"])}, new unique frames: {len(frames) - before}', flush=True)
    return sorted(frames.items())


class FrameSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path = self.items[i][1]
        with Image.open(path) as im:
            im = im.convert('RGB')
            if im.size != (SRC_W, SRC_H):
                raise ValueError(f'unexpected frame size {im.size} in {path}')
            arr = np.asarray(im.resize((INFER_W, INFER_H), Image.BILINEAR))
        return torch.from_numpy(arr.copy()), i


def load_model(device, fp16):
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).eval().to(device)
    if fp16:
        model = model.half()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    if len(id2label) != 19 or id2label[0].lower() != 'road' or id2label[11].lower() != 'person':
        raise RuntimeError(f'label space is not Cityscapes trainId: {id2label}')
    for g, ids in _GROUPS.items():
        print(f'  {g:12s} <- ' + ', '.join(f'{i}:{id2label[i]}' for i in ids))
    return model


@torch.inference_mode()
def predict(model, batch_u8, device, fp16, mean, std):
    x = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - mean) / std
    if fp16:
        x = x.half()
    logits = model(pixel_values=x).logits                               # (B, 19, H/4, W/4)
    logits = torch.nn.functional.interpolate(logits.float(), size=(OUT_H, OUT_W),
                                             mode='bilinear', align_corners=False)
    return logits.argmax(1).to(torch.uint8).cpu().numpy()               # (B, OUT_H, OUT_W)


def save_png(label, final):
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final[:-len('.png')] + '.tmp.png'
    Image.fromarray(label).save(tmp, compress_level=1)
    os.replace(tmp, final)                                              # never leaves a half-written file


def save_preview(label, key, path):
    with Image.open(path) as im:
        img = np.asarray(im.convert('RGB').resize((OUT_W, OUT_H), Image.BILINEAR), np.float32)
    blend = (0.5 * img + 0.5 * PALETTE[label].astype(np.float32)).astype(np.uint8)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    sid, vid, fr = key
    Image.fromarray(blend).save(os.path.join(PREVIEW_DIR, f'{sid}_{vid}_{fr:05d}.jpg'), quality=90)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = device == 'cuda'
    if device == 'cpu':
        print('WARNING: CUDA not available, this will be extremely slow.')
    torch.backends.cudnn.benchmark = True

    items = collect_frames()
    todo = [it for it in items if not os.path.isfile(out_path(it[0]))]
    print(f'unique frames needed: {len(items)} | already done: {len(items) - len(todo)} | to do: {len(todo)}')
    if LIMIT_FRAMES is not None and len(todo) > LIMIT_FRAMES:
        todo = [todo[j] for j in np.linspace(0, len(todo) - 1, LIMIT_FRAMES).astype(int)]
        print(f'SMOKE TEST: processing {len(todo)} frames only')
    if not todo:
        print('nothing to do')
        return

    model = load_model(device, fp16)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    loader = DataLoader(FrameSet(todo), batch_size=BATCH, num_workers=NUM_WORKERS,
                        pin_memory=(device == 'cuda'))

    counts = np.zeros(19, np.int64)
    done, n_prev, n0, t0 = 0, 0, 0, time.time()
    for b, (xb, idx) in enumerate(loader):
        if b == WARMUP_BATCHES:        # exclude worker start-up and cudnn autotune from the rate
            n0, t0 = done, time.time()
        labels = predict(model, xb, device, fp16, mean, std)
        for lab, i in zip(labels, idx.tolist()):
            key, path = todo[i]
            save_png(lab, out_path(key))
            counts += np.bincount(lab.ravel(), minlength=19)
            if n_prev < PREVIEW_N:
                save_preview(lab, key, path)
                n_prev += 1
        done += len(labels)
        if b > WARMUP_BATCHES and (done % (BATCH * 50) == 0 or done == len(todo)):
            rate = (done - n0) / (time.time() - t0)
            print(f'{done}/{len(todo)} frames | {rate:.2f} fps (after warmup) | '
                  f'ETA {(len(todo) - done) / rate / 3600:.1f} h', flush=True)

    share = 100.0 * counts / counts.sum()
    print('class share (%):', ', '.join(f'{n}={s:.1f}' for n, s in zip(CLASS_NAMES, share)))
    if device == 'cuda':
        print(f'peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    print(f'finished: {done} label maps -> {OUT_DIR}; previews -> {PREVIEW_DIR}')


if __name__ == '__main__':
    main()