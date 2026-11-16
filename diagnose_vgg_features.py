"""diagnose_vgg_features.py -- checks behind the VGG16 sparsity question.

A  cache consistency: recompute cached per-frame features with the same crop
   and preprocessing steps, compare with the stored .pkl files.
B  sparsity by input type at 224x224: pedestrian crops, natural-photo crops
   (same code path), whole natural photos.
C  resampling: pedestrian crops with NEAREST (pipeline) vs LANCZOS.

Run from D:\\PIE codes\\PIE in the pie_tf environment.
"""
import glob
import os
import pickle
import sys
import tempfile

import numpy as np
from PIL import Image

PIE_PATH = r"D:\PIE codes\PIE_dataset"
CODE_DIR = r"D:\PIE codes\PIE"
os.environ["PIE_PATH"] = PIE_PATH
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from keras.applications import vgg16
from keras.models import Model
from sklearn.datasets import load_sample_images

from pie_data_Hesam import PIE
from utils import jitter_bbox, squarify, img_pad

SIZE = 224
ENLARGE = 2
SQUARIFY = 1
N_CACHE = 40
N_PED = 300
N_NAT = 300
BATCH = 32
SEED = 0
TOL_CACHE = 1e-4
CACHE_ROOT = os.path.join(PIE_PATH, "data", "data", "pie", "train",
                          "features_context_pad_resize", "vgg16_none_prep1")


def crop_box(path, bbox):
    img = Image.open(path).convert("RGB")
    b = jitter_bbox([list(bbox)], path, "enlarge", ENLARGE)[0]
    b = squarify(b, SQUARIFY, img.size[0])
    return img.crop(list(map(int, b[0:4])))


def pad_resize(img, resample):
    # Same geometry as utils.img_pad(mode='pad_resize'); resample is a parameter here.
    w, h = img.size
    ratio = float(SIZE) / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    canvas = Image.new("RGB", (SIZE, SIZE))
    canvas.paste(img.resize((nw, nh), resample), ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas


def to_input(img):
    return vgg16.preprocess_input(np.array(img, dtype=np.float32))


def build_model():
    base = vgg16.VGG16(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet")
    return Model(base.input, [base.get_layer("block5_conv3").output, base.output])


def part_a(model):
    pkls = glob.glob(os.path.join(CACHE_ROOT, "set*", "video_*", "*.pkl"))
    print(f"[A] cached frame files: {len(pkls)}")
    if not pkls:
        return
    db = PIE(data_path=PIE_PATH).generate_database()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(pkls), size=min(N_CACHE, len(pkls)), replace=False)
    rel, skipped = [], 0
    for k in pick:
        p = pkls[k]
        fn = os.path.basename(p)
        frame, pid = int(fn[:5]), fn[6:-4]
        vid = os.path.basename(os.path.dirname(p))
        sid = os.path.basename(os.path.dirname(os.path.dirname(p)))
        ann = db.get(sid, {}).get(vid, {}).get("ped_annotations", {}).get(pid)
        if ann is None or frame not in ann["frames"]:
            skipped += 1
            continue
        bbox = ann["bbox"][ann["frames"].index(frame)]
        img_path = os.path.join(PIE_PATH, "images", sid, vid, f"{frame:05d}.png")
        x = to_input(pad_resize(crop_box(img_path, bbox), Image.NEAREST))[None]
        fresh = model.predict(x, verbose=0)[1][0]
        with open(p, "rb") as f:
            cached = np.squeeze(pickle.load(f))
        rel.append(float(np.abs(fresh - cached).max() / (np.abs(cached).max() + 1e-12)))

    if not rel:
        print(f"[A] nothing comparable (skipped={skipped})")
        return
    rel = np.asarray(rel)
    print(f"[A] checked={len(rel)} skipped={skipped}")
    print(f"[A] relative max error: median={np.median(rel):.2e} max={rel.max():.2e}")
    verdict = "CONSISTENT" if rel.max() < TOL_CACHE else "MISMATCH: cache was built by different code"
    print(f"[A] verdict: {verdict}")


def pedestrian_items(rng):
    seq = PIE(data_path=PIE_PATH).generate_data_trajectory_sequence(
        "train", seq_type="intention", min_track_size=15)
    items = []
    for t in rng.permutation(len(seq["image"])):
        f = int(rng.integers(len(seq["image"][t])))
        items.append((seq["image"][t][f], seq["bbox"][t][f]))
        if len(items) == N_PED:
            break
    return items


def natural_items(rng, tmpdir):
    photos = []
    for i, arr in enumerate(load_sample_images().images):
        p = os.path.join(tmpdir, f"nat{i}.png")
        Image.fromarray(arr).save(p)
        photos.append((p, arr.shape[1], arr.shape[0]))
    crops = []
    for _ in range(N_NAT):
        p, w, h = photos[int(rng.integers(len(photos)))]
        side = int(rng.integers(40, 200))
        x1 = int(rng.integers(0, w - side))
        y1 = int(rng.integers(0, h - side))
        crops.append((p, [x1, y1, x1 + side, y1 + side]))
    whole = [(p, None) for p, _, _ in photos]
    return crops, whole


def make_input(resample):
    def f(item):
        path, bbox = item
        if bbox is None:
            crop = Image.open(path).convert("RGB")
        else:
            crop = crop_box(path, bbox)
        return pad_resize(crop, resample), crop.size
    return f


def run_group(name, items, make, model):
    zc = zp = nc = npool = 0
    lo, hi = np.inf, -np.inf
    per_img, sides = [], []
    for i in range(0, len(items), BATCH):
        made = [make(it) for it in items[i:i + BATCH]]
        X = np.stack([to_input(im) for im, _ in made])
        sides += [min(s) for _, s in made]
        lo = min(lo, float(X.min()))
        hi = max(hi, float(X.max()))
        conv, pool = model.predict(X, batch_size=BATCH, verbose=0)
        zc += int((conv == 0).sum())
        nc += conv.size
        zp += int((pool == 0).sum())
        npool += pool.size
        per_img += list((pool.reshape(len(pool), -1) == 0).mean(axis=1))
    return dict(name=name, n=len(items), conv0=zc / nc, pool0=zp / npool,
                img_med=float(np.median(per_img)), lo=lo, hi=hi,
                side=float(np.median(sides)))


def print_rows(rows):
    header = (f"{'group':<36}{'n':>6}{'zero conv5_3':>14}{'zero pool5':>12}"
              f"{'med/img':>10}{'in_min':>9}{'in_max':>8}{'src_side':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<36}{r['n']:>6}{r['conv0']:>14.3f}{r['pool0']:>12.3f}"
              f"{r['img_med']:>10.3f}{r['lo']:>9.1f}{r['hi']:>8.1f}{r['side']:>10.0f}")


def main():
    rng = np.random.default_rng(SEED)
    model = build_model()
    part_a(model)

    ped = pedestrian_items(rng)
    sample = crop_box(*ped[0])
    assert np.array_equal(np.asarray(pad_resize(sample, Image.NEAREST)),
                          np.asarray(img_pad(sample, mode="pad_resize", size=SIZE))), \
        "pad_resize replica differs from utils.img_pad"
    print("\n[B] pad_resize replica matches utils.img_pad")

    with tempfile.TemporaryDirectory() as tmpdir:
        crops, whole = natural_items(rng, tmpdir)
        rows = [
            run_group("pedestrian crops (pipeline)", ped, make_input(Image.NEAREST), model),
            run_group("natural crops (same path)", crops, make_input(Image.NEAREST), model),
            run_group("natural whole photos (n=2 photos)", whole, make_input(Image.NEAREST), model),
        ]
        print()
        print_rows(rows)

        print("\n[C] resampling on pedestrian crops")
        rows_c = [
            run_group("pedestrian, NEAREST (pipeline)", ped, make_input(Image.NEAREST), model),
            run_group("pedestrian, LANCZOS", ped, make_input(Image.LANCZOS), model),
        ]
        print_rows(rows_c)


if __name__ == "__main__":
    main()