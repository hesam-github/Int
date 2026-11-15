import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from pie_data_Hesam import PIE

from utils import jitter_bbox, squarify, img_pad


def _load_vgg16():
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(
        vgg.features,
        vgg.avgpool,
        torch.nn.Flatten(),
        *list(vgg.classifier.children())[:4],
    )
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = True
    return feature_extractor


def _load_resnet50():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(
        *list(resnet.children())[:-1],
        torch.nn.Flatten(),
    )
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = True
    return feature_extractor


def _load_vgg16_conv():
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(
        vgg.features,
        vgg.avgpool,
        torch.nn.Flatten(),
    )
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = True
    return feature_extractor


def _load_vgg16_conv_2x2():
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(
        vgg.features,
        torch.nn.AdaptiveAvgPool2d((2, 2)),
        torch.nn.Flatten(),
    )
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = True
    return feature_extractor


BACKBONES = {
    'vgg16':          ('_load_vgg16',          4096),
    'resnet50':       ('_load_resnet50',       2048),
    'vgg16_conv':     ('_load_vgg16_conv',     25088),
    'vgg16_conv_2x2': ('_load_vgg16_conv_2x2', 2048),
}

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD  = [0.229, 0.224, 0.225]
IMG_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
])


class PIEIntentionDataset(Dataset):
    DEFAULT_DATA_OPTS = {
        'fstride':         1,
        'data_split_type': 'default',
        'seq_type':        'intention',
        'height_rng':      [0, float('inf')],
        'squarify_ratio':  0,
        'min_track_size':  0,
    }

    ENLARGE_RATIO = 2
    SQUARIFY_RATIO = 1
    PAD_MODE = 'pad_resize'

    def __init__(self, imdb, image_set, observe_length=15, backbone='vgg16',
                 seq_overlap_rate=0.5, normalize_bbox=True,
                 balance_classes=False, balance_seed=42,
                 cache_dir=None, data_opts=None, device=None):

        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Available: {list(BACKBONES.keys())}")
        self._loader_name, self.feature_dim = BACKBONES[backbone]
        self.backbone_name    = backbone
        self.observe_length   = observe_length
        self.seq_overlap_rate = seq_overlap_rate
        self.normalize_bbox   = normalize_bbox
        self.image_set        = image_set
        self.device           = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if cache_dir is None:
            cache_dir = os.path.join(imdb._pie_path, 'vgg_cache')
        self.cache_dir = os.path.join(cache_dir, backbone)

        self.img_width  = 1920
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

        print(f"[PIEIntentionDataset] split={image_set}  backbone={backbone}  "
              f"obs={observe_length}  overlap={seq_overlap_rate}  "
              f"balanced={balance_classes and image_set in ('train','val')}  "
              f"tracks={len(self.tracks)}  windows={len(self.samples)}")

    def _build_tracks(self, raw_data):
        tracks = []
        skipped_no_image = 0
        skipped_short = 0

        n = len(raw_data['bbox'])
        for i in range(n):
            first_path = raw_data['image'][i][0]
            if not os.path.isfile(first_path):
                skipped_no_image += 1
                continue
            if len(raw_data['bbox'][i]) < self.observe_length:
                skipped_short += 1
                continue

            ped_id         = raw_data['ped_id'][i][0][0]
            label          = raw_data['intention_binary'][i][0][0]
            intention_prob = raw_data['intention_prob'][i][0][0]

            tracks.append({
                'ped_id':         ped_id,
                'label':          label,
                'intention_prob': intention_prob,
                'bbox_seq':       raw_data['bbox'][i],
                'image_seq':      raw_data['image'][i],
                'occlusion_seq':  raw_data['occlusion'][i],
            })

        print(f"_build_tracks: kept= {len(tracks)}  "
              f"skipped_no_image= {skipped_no_image}  "
              f"skipped_short= {skipped_short}")
        return tracks
# pie_dataset.py -- REPLACES the existing _balance_tracks method

    def _balance_tracks(self, tracks, seed):
        """
        Undersamples the majority-label tracks so the track-level class
        distribution is exactly 50/50 -- matches PIE.balance_samples_count()
        in the reference pie_data.py, applied to train and val (never test),
        on raw tracks, before windowing.

        FIX (this version): undersampling is now STRATIFIED BY VIDEO. The
        original version drew majority-label tracks to drop via a single
        global random shuffle, with no knowledge of which video each track
        came from. Verified via execution on the real val split (obs=15,
        vgg16): this let a purely random draw disproportionately deplete
        set05 (already scarce -- 16 of 228 tracks, 7.0%) relative to set06
        (212 of 228, 93.0%), shrinking set05's share further to 5.0% after
        balancing. That is the OPPOSITE of what we want: class-balancing
        should not silently make an existing video/camera distribution shift
        worse. This version instead computes how many majority-label tracks
        to drop from EACH video proportional to that video's share of the
        majority-label pool (largest-remainder rounding so the per-video
        drop counts sum exactly to `diff`), so each video's relative
        contribution is preserved in expectation, not just in the aggregate.

        Uses an isolated np.random.RandomState(seed), not the global numpy
        RNG, so it's reproducible independent of whatever else has consumed
        the global stream (DataLoader shuffling, other dataset instantiations).
        """
        labels = np.array([tr['label'] for tr in tracks])
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        if n_pos == n_neg:
            print(f"_balance_tracks: already balanced (pos={n_pos} neg={n_neg})")
            return tracks

        majority_label = 1 if n_pos > n_neg else 0
        diff = abs(n_pos - n_neg)
        majority_idx = np.where(labels == majority_label)[0]

        # group majority-label track indices by video (set_id + video_id,
        # dropping the pedestrian id -- e.g. '1_2_50' -> '1_2').
        idx_by_video = {}
        for i in majority_idx:
            vkey = '_'.join(tracks[i]['ped_id'].split('_')[:2])
            idx_by_video.setdefault(vkey, []).append(int(i))

        videos = sorted(idx_by_video.keys())
        counts = np.array([len(idx_by_video[v]) for v in videos], dtype=float)
        proportions = counts / counts.sum()

        # Largest-remainder method: proportional allocation of `diff` drops
        # across videos, rounded to integers that sum EXACTLY to diff (plain
        # np.round per-video would not guarantee the total matches diff).
        raw = proportions * diff
        base = np.floor(raw).astype(int)
        remainder = diff - int(base.sum())
        frac = raw - base
        order = np.argsort(-frac)          # videos with largest fractional part first
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
            f"majority-label(={majority_label}) tracks, STRATIFIED BY VIDEO "
            f"across {len(videos)} videos -> kept {len(balanced)}")
        return balanced

    def _build_windows(self, tracks):
        obs = self.observe_length
        overlap = self.seq_overlap_rate
        stride = obs if overlap == 0 else int((1 - overlap) * obs)
        stride = max(1, stride)

        samples = []
        for ti, tr in enumerate(tracks):
            T = len(tr['bbox_seq'])
            for start in range(0, T - obs + 1, stride):
                samples.append({'track_idx': ti, 'start': start, 'end': start + obs})

        print(f" _build_windows: obs= {obs} overlap= {overlap}  stride= {stride}    windows= {len(samples)}  ")
        return samples

    def _get_cache_path(self, ped_id):
        parts = ped_id.split('_')
        set_name = f"set{int(parts[0]):02d}"
        video_name = f"video_{int(parts[1]):04d}"
        filename = f"{ped_id}.npy"
        return os.path.join(self.cache_dir, set_name, video_name, filename)

    def _ensure_cache(self):
        uncached = [tr for tr in self.tracks
                   if not os.path.isfile(self._get_cache_path(tr['ped_id']))]
        if not uncached:
            print(f"All {len(self.tracks)} track feature files already cached")
            return
        print(f"Extracting features for {len(uncached)} uncached tracks...")

        loader_fn = globals()[self._loader_name]
        extractor = loader_fn().to(self.device)
        try:
            with torch.no_grad():
                for i, tr in enumerate(uncached):
                    cache_path = self._get_cache_path(tr['ped_id'])
                    self._extract_single_track(tr, cache_path, extractor)
                    if (i + 1) % 50 == 0 or (i + 1) == len(uncached):
                        print(f"{i+1}/{len(uncached)} tracks done")
        finally:
            del extractor
            if torch.cuda.is_available():
                print('deleting gpu cache...')
                torch.cuda.empty_cache()

    def _extract_single_track(self, track, cache_path, extractor):
        bbox_seq = track['bbox_seq']
        image_seq = track['image_seq']
        T = len(image_seq)

        features = np.zeros((T, self.feature_dim), dtype=np.float32)

        for frame_idx in range(T):
            path = image_seq[frame_idx]
            full = Image.open(path).convert('RGB')

            b = list(bbox_seq[frame_idx])
            enlarged = jitter_bbox([b], path, 'enlarge', self.ENLARGE_RATIO)[0]
            squared = squarify(enlarged, self.SQUARIFY_RATIO, self.img_width)
            box_int = list(map(int, squared[0:4]))
            cropped = full.crop(box_int)
            padded = img_pad(cropped, mode=self.PAD_MODE, size=224)

            tensor = IMG_TO_TENSOR(padded).unsqueeze(0).to(self.device)
            feat = extractor(tensor)
            features[frame_idx] = feat.cpu().numpy()[0]

        assert features.shape == (T, self.feature_dim), \
            f"Track {track['ped_id']}: expected ({T}, {self.feature_dim}), got {features.shape}"

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
            features_full = np.load(cache_path)
        except Exception as e_:
            raise RuntimeError(
                f"Corrupted feature cache file for track '{track['ped_id']}': "
                f"{cache_path}\n({type(e_).__name__}: {e_})\n"
                f"Delete this file and re-run training; it will be re-extracted automatically."
            ) from e_

        assert e <= features_full.shape[0], (
            f"Window [{s}:{e}] exceeds cached track {track['ped_id']} of length {features_full.shape[0]}."
        )

        features = features_full[s:e]
        bbox = np.asarray(track['bbox_seq'][s:e], dtype=np.float32)

        if self.normalize_bbox:
            bbox = bbox.copy()
            bbox[:, [0, 2]] /= self.img_width
            bbox[:, [1, 3]] /= self.img_height

        return {
            'features':       torch.from_numpy(features).float(),
            'bbox':           torch.from_numpy(bbox).float(),
            'label':          torch.tensor(track['label'], dtype=torch.long),
            'intention_prob': torch.tensor(track['intention_prob'], dtype=torch.float),
            'ped_id':         track['ped_id'],
        }