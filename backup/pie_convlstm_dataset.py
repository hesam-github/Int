"""
Dataset for the TRUE reference architecture (PIEConvLSTMEncDec): caches
raw VGG16 conv5-block features per frame, shape (512, 7, 7), UNFLATTENED
-- unlike pie_dataset.py's PIEIntentionDataset, which caches flat fc7
vectors (4096,) for the separate GRU/Transformer/S4D/Mamba four-way
comparison.

Deliberately a SEPARATE class/cache directory rather than extending
PIEIntentionDataset/BACKBONES: that dict's contract assumes a flat
per-frame feature_dim (int) and a (T, feature_dim) cache array shape.
Retrofitting spatial (C,H,W) output into that contract would either
break the existing four-way pipeline's assumptions silently or require
invasive changes to load-bearing, already-tested code. Track/window
building logic below intentionally mirrors PIEIntentionDataset's
_build_tracks/_build_windows/_balance_tracks (same verified logic, same
video-stratified balance fix already applied there) rather than a risky
shared abstraction built under time pressure.

Only intention estimation is supported here (matches
PIEConvLSTMEncDec, which is itself only a port of pie_intent.py's
train_intent() model, not the trajectory/speed prediction heads).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image

from utils import jitter_bbox, squarify, img_pad

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]
IMG_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
])


def _load_vgg16_conv5():
    """
    Matches Keras vgg16.VGG16(include_top=False): output is the raw
    conv5 block activation, (512, 7, 7) for a 224x224 input -- NO
    avgpool, NO flatten, NO fc6/fc7. torchvision's vgg16().features is
    exactly this conv stack with no avgpool/classifier attached, so no
    truncation surgery is needed here (unlike the fc7 backbone in
    pie_dataset.py, which DOES need classifier[:4] truncation).
    """
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    extractor = vgg.features   # (B,512,7,7) for 224x224 input:
                                # 5 maxpools of stride 2 -> 224/32 = 7
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


class PIEConvLSTMDataset(Dataset):
    ENLARGE_RATIO = 2
    SQUARIFY_RATIO = 1
    PAD_MODE = 'pad_resize'
    FEATURE_SHAPE = (512, 7, 7)   # (C, H, W) -- fixed for VGG16 conv5 @ 224 input

    DEFAULT_DATA_OPTS = {
        'fstride': 1,
        'data_split_type': 'default',
        'seq_type': 'intention',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'min_track_size': 0,
    }

    def __init__(self, imdb, image_set, observe_length=15,
                 seq_overlap_rate=0.5, normalize_bbox=False,
                 balance_classes=True, balance_seed=42,
                 cache_dir=None, data_opts=None, device=None):
        # NOTE on normalize_bbox default: the reference train_intent()
        # data_opts dict (train_test.py) does NOT set 'normalize_bbox' at
        # all for the intention model -- unlike train_predict()'s
        # trajectory data_opts, which explicitly sets normalize_bbox=True.
        # Default here is False to match that asymmetry; pass True
        # explicitly if you deliberately want to deviate from the
        # reference on this specific point.
        self.observe_length = observe_length
        self.seq_overlap_rate = seq_overlap_rate
        self.normalize_bbox = normalize_bbox
        self.image_set = image_set
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if cache_dir is None:
            cache_dir = os.path.join(imdb._pie_path, 'vgg_cache')
        self.cache_dir = os.path.join(cache_dir, 'vgg16_conv5')

        self.img_width = 1920
        self.img_height = 1080

        opts = dict(self.DEFAULT_DATA_OPTS)
        if data_opts is not None:
            opts.update(data_opts)
        raw_data = imdb.generate_data_trajectory_sequence(image_set, **opts)

        self.tracks = self._build_tracks(raw_data)
        if balance_classes and image_set in ('train', 'val'):
            self.tracks = self._balance_tracks(self.tracks, seed=balance_seed)
        self.samples = self._build_windows(self.tracks)

        self._ensure_cache()

        print(f"[PIEConvLSTMDataset] split={image_set}  obs={observe_length}  "
              f"overlap={seq_overlap_rate}  "
              f"balanced={balance_classes and image_set in ('train', 'val')}  "
              f"tracks={len(self.tracks)}  windows={len(self.samples)}")

    def _build_tracks(self, raw_data):
        tracks, skipped_no_image, skipped_short = [], 0, 0
        n = len(raw_data['bbox'])
        for i in range(n):
            first_path = raw_data['image'][i][0]
            if not os.path.isfile(first_path):
                skipped_no_image += 1
                continue
            if len(raw_data['bbox'][i]) < self.observe_length:
                skipped_short += 1
                continue
            tracks.append({
                'ped_id': raw_data['ped_id'][i][0][0],
                'label': raw_data['intention_binary'][i][0][0],
                'intention_prob': raw_data['intention_prob'][i][0][0],
                'bbox_seq': raw_data['bbox'][i],
                'image_seq': raw_data['image'][i],
            })
        print(f"_build_tracks: kept={len(tracks)}  skipped_no_image={skipped_no_image}  "
              f"skipped_short={skipped_short}")
        return tracks

    def _balance_tracks(self, tracks, seed):
        """
        Video-stratified undersampling of the majority label (same fix
        already applied to pie_dataset.py's _balance_tracks) -- duplicated
        here rather than imported, per the "separate, self-contained file"
        decision above.
        """
        labels = np.array([tr['label'] for tr in tracks])
        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        if n_pos == n_neg:
            print(f"_balance_tracks: already balanced (pos={n_pos} neg={n_neg})")
            return tracks
        majority_label = 1 if n_pos > n_neg else 0
        diff = abs(n_pos - n_neg)
        majority_idx = np.where(labels == majority_label)[0]

        idx_by_video = {}
        for i in majority_idx:
            vkey = '_'.join(tracks[i]['ped_id'].split('_')[:2])
            idx_by_video.setdefault(vkey, []).append(int(i))
        videos = sorted(idx_by_video.keys())
        counts = np.array([len(idx_by_video[v]) for v in videos], dtype=float)
        proportions = counts / counts.sum()
        raw = proportions * diff
        base = np.floor(raw).astype(int)
        remainder = diff - int(base.sum())
        order = np.argsort(-(raw - base))
        for k in range(remainder):
            base[order[k]] += 1

        rng = np.random.RandomState(seed)
        drop_idx = set()
        for vkey, n_drop in zip(videos, base):
            pool = idx_by_video[vkey].copy()
            rng.shuffle(pool)
            drop_idx.update(pool[:n_drop])

        balanced = [tr for i, tr in enumerate(tracks) if i not in drop_idx]
        print(f"_balance_tracks: pos={n_pos} neg={n_neg} -> dropped {diff} "
              f"majority-label(={majority_label}) tracks, video-stratified -> "
              f"kept {len(balanced)}")
        return balanced

    def _build_windows(self, tracks):
        obs, overlap = self.observe_length, self.seq_overlap_rate
        stride = obs if overlap == 0 else max(1, int((1 - overlap) * obs))
        samples = []
        for ti, tr in enumerate(tracks):
            T = len(tr['bbox_seq'])
            for start in range(0, T - obs + 1, stride):
                samples.append({'track_idx': ti, 'start': start, 'end': start + obs})
        print(f"_build_windows: obs={obs} overlap={overlap} stride={stride} "
              f"windows={len(samples)}")
        return samples

    def _get_cache_path(self, ped_id):
        parts = ped_id.split('_')
        set_name = f"set{int(parts[0]):02d}"
        video_name = f"video_{int(parts[1]):04d}"
        return os.path.join(self.cache_dir, set_name, video_name, f"{ped_id}.npy")

    def _ensure_cache(self):
        uncached = [tr for tr in self.tracks
                    if not os.path.isfile(self._get_cache_path(tr['ped_id']))]
        if not uncached:
            print(f"All {len(self.tracks)} track feature files already cached (vgg16_conv5)")
            return
        print(f"Extracting vgg16_conv5 features for {len(uncached)} uncached tracks...")
        extractor = _load_vgg16_conv5().to(self.device)
        try:
            with torch.no_grad():
                for i, tr in enumerate(uncached):
                    self._extract_single_track(tr, self._get_cache_path(tr['ped_id']), extractor)
                    if (i + 1) % 50 == 0 or (i + 1) == len(uncached):
                        print(f"{i+1}/{len(uncached)} tracks done")
        finally:
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_single_track(self, track, cache_path, extractor):
        bbox_seq, image_seq = track['bbox_seq'], track['image_seq']
        T = len(image_seq)
        C, H, W = self.FEATURE_SHAPE
        features = np.zeros((T, C, H, W), dtype=np.float32)

        for t in range(T):
            path = image_seq[t]
            full = Image.open(path).convert('RGB')
            b = list(bbox_seq[t])
            enlarged = jitter_bbox([b], path, 'enlarge', self.ENLARGE_RATIO)[0]
            squared = squarify(enlarged, self.SQUARIFY_RATIO, self.img_width)
            box_int = list(map(int, squared[0:4]))
            cropped = full.crop(box_int)
            padded = img_pad(cropped, mode=self.PAD_MODE, size=224)
            tensor = IMG_TO_TENSOR(padded).unsqueeze(0).to(self.device)
            feat = extractor(tensor)                 # (1, 512, 7, 7)
            features[t] = feat.cpu().numpy()[0]

        assert features.shape == (T, C, H, W), \
            f"Track {track['ped_id']}: expected ({T},{C},{H},{W}), got {features.shape}"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path[:-len('.npy')] + '.tmp.npy'
        np.save(tmp_path, features)
        os.replace(tmp_path, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        win = self.samples[idx]
        track = self.tracks[win['track_idx']]
        s, e = win['start'], win['end']
        cache_path = self._get_cache_path(track['ped_id'])
        try:
            features_full = np.load(cache_path)          # (T_full, 512, 7, 7)
        except Exception as err:
            raise RuntimeError(
                f"Corrupted feature cache for track '{track['ped_id']}': {cache_path}\n"
                f"({type(err).__name__}: {err}) -- delete and re-run to re-extract."
            ) from err

        assert e <= features_full.shape[0], (
            f"Window [{s}:{e}] exceeds cached track {track['ped_id']} "
            f"of length {features_full.shape[0]}."
        )

        features = features_full[s:e]                    # (obs, 512, 7, 7)
        bbox = np.asarray(track['bbox_seq'][s:e], dtype=np.float32)   # (obs, 4)
        if self.normalize_bbox:
            bbox = bbox.copy()
            bbox[:, [0, 2]] /= self.img_width
            bbox[:, [1, 3]] /= self.img_height

        return {
            'features': torch.from_numpy(features).float(),   # (obs,512,7,7)
            'bbox': torch.from_numpy(bbox).float(),            # (obs,4)
            'label': torch.tensor(track['label'], dtype=torch.long),
            'ped_id': track['ped_id'],
        }
