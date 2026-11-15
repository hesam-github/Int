"""
crop_tools.py
Inspect training-style crops and measure occlusion sensitivity by region."""
import numpy as np
from PIL import Image
from keras.applications import vgg16
from utils import jitter_bbox, squarify, img_pad, normalize_decoder_bbox
from probe_common import *

MEAN_RGB = np.array([123.68, 116.779, 103.939], np.float32)  # zero after VGG preprocess


def crop_with_geometry(img_path, bbox):
    """Same crop as pie_intent.load_images_and_process (no augmentation).

    Returns the 224x224 crop and the pedestrian box in that crop's coordinates.
    """
    img = Image.open(img_path).convert('RGB')
    b = squarify(jitter_bbox([list(bbox)], img_path, 'enlarge', 2)[0], 1, img.size[0])
    cb = list(map(int, b[0:4]))
    crop = img.crop(cb)
    w, h = crop.size
    r = 224.0 / max(w, h)
    ox, oy = (224 - int(w * r)) // 2, (224 - int(h * r)) // 2
    x1, y1, x2, y2 = bbox
    ped = [(x1 - cb[0]) * r + ox, (y1 - cb[1]) * r + oy,
           (x2 - cb[0]) * r + ox, (y2 - cb[1]) * r + oy]
    return img_pad(crop, mode='pad_resize', size=224), ped


def save_crop_montage(data, n=8, out='crops.png', seed=0):
    """Save n random last-frame crops side by side for eyeballing head size."""
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(data['image']), n, replace=False)
    tiles = [crop_with_geometry(data['image'][t][-1], data['bbox'][t][-1])[0] for t in ids]
    canvas = Image.new('RGB', (224 * n, 224))
    for i, t in enumerate(tiles):
        canvas.paste(t, (224 * i, 0))
    canvas.save(out)


def region_masks(ped):
    """Boolean 224x224 masks: head (top 20% of ped box), body, background."""
    x1, y1, x2, y2 = [int(round(v)) for v in ped]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, 224), min(y2, 224)
    hy = y1 + int(0.2 * (y2 - y1))
    head = np.zeros((224, 224), bool); head[y1:hy, x1:x2] = True
    box = np.zeros((224, 224), bool); box[y1:y2, x1:x2] = True
    return {'head': head, 'body': box & ~head, 'background': ~box}


def occlusion_sensitivity(model, data, n_windows=50, seq_len=15, seed=0):
    """Mean drop in P(cross) when a region is replaced by the ImageNet mean."""
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(data['image']), min(n_windows, len(data['image'])), replace=False)
    drops = {'head': [], 'body': [], 'background': []}
    for t in ids:
        imgs, boxes = data['image'][t][-seq_len:], data['bbox'][t][-seq_len:]
        crops, peds = zip(*[crop_with_geometry(p, b) for p, b in zip(imgs, boxes)])
        arr = np.stack([np.asarray(c, np.float32) for c in crops])
        dec = normalize_decoder_bbox(np.array(boxes, np.float32)[None], ['bbox'])

        def prob(a):
            f = vgg.predict(vgg16.preprocess_input(a.copy()), batch_size=seq_len, verbose=0)
            return float(model.predict([f[None], dec], verbose=0)[0, 0])

        p0 = prob(arr)
        for name in drops:
            masked = arr.copy()
            for k, ped in enumerate(peds):
                masked[k][region_masks(ped)[name]] = MEAN_RGB
            drops[name].append(p0 - prob(masked))
    for name, d in drops.items():
        print(f'{name:11s} mean drop={np.mean(d):+.4f}  mean |drop|={np.mean(np.abs(d)):.4f}')