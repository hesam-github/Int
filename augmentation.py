"""
augmentation.py

Purpose: geometric AND photometric AND temporal data augmentation for PIE
pedestrian-intention tracks, generated ONCE (deterministic, fixed seed)
before training so the VGG16 feature cache in
PIEIntent.load_images_and_process() is built once and reused across every
subsequent epoch and run.

v3 adds four NEW temporal/robustness augmentations on top of the original
v2 geometric+photometric ones:
  - random start-offset (teaches the model not to rely on "intention
    always happens near the middle of the clip")
  - discrete speed resampling (simulates walking faster/slower)
  - occlusion simulation via frame-freeze (simulates a real tracker
    losing the pedestrian for a couple of frames)
  - per-frame micro-jitter (removes an artificial "too smooth" signature
    that could let the model shortcut-detect synthetic tracks)

All four are cache-safe: every output frame position still maps to one
of the track's REAL (image, bbox) pairs (frozen/repeated when needed),
never to an interpolated/invented bbox -- so the VGG16 feature cache key
(image filename + synthetic ped_id) is always a correct, non-colliding
lookup.
"""

import random

from utils import PIE_IMG_WIDTH, PIE_IMG_HEIGHT


# Bump this whenever the augmentation LOGIC changes in a way that would
# alter the output for the SAME inputs. Embedded in every synthetic
# ped_id, so a logic change always produces new cache entries instead of
# silently reusing features generated under the old logic.
#
# v2 -> v3: added temporal shift, discrete speed resampling, occlusion
# simulation (frame-freeze), and per-frame micro-jitter. Any track
# feature cached under v2 never had these applied, so it MUST be treated
# as stale -- bumping the version guarantees that.
AUGMENTATION_SCHEMA_VERSION = 5

_AUG_MARKER = "__aug"


def _encode_signed(value, scale=1000):
    sign = 'p' if value >= 0 else 'n'
    return f"{sign}{int(round(abs(value) * scale)):03d}"


def _decode_signed(token, scale=1000):
    sign = 1.0 if token[0] == 'p' else -1.0
    return sign * int(token[1:]) / scale


# Single source of truth: used BOTH for the augment_fn call and for logging,
# so the log can never drift from what actually runs.
# Values mirror augment_pie_tracks() defaults (v3).
AUGMENTATION_CFG = dict(
    jitter_ratio=0.05, scale_jitter=0.05, hflip_prob=0.5,
    brightness_jitter=0.2, contrast_jitter=0.2,
    max_temporal_shift=2, occlusion_prob=0.25, occlusion_max_frames=2,
    speed_jitter=0.15, micro_jitter_px=3,
)


def _describe_aug_cfg(cfg):
    # Build a readable summary of only the ENABLED augmentation groups.
    parts = []
    if cfg['jitter_ratio'] > 0 or cfg['scale_jitter'] > 0 or cfg['hflip_prob'] > 0:
        parts.append(f"geometric(jitter={cfg['jitter_ratio']}, scale={cfg['scale_jitter']}, "
                     f"hflip={cfg['hflip_prob']})")
    if cfg['brightness_jitter'] > 0 or cfg['contrast_jitter'] > 0:
        parts.append(f"photometric(brightness={cfg['brightness_jitter']}, "
                     f"contrast={cfg['contrast_jitter']})")
    if cfg['max_temporal_shift'] > 0:
        parts.append(f"temporal-shift(max={cfg['max_temporal_shift']})")
    if cfg['speed_jitter'] > 0:
        parts.append(f"speed(+/-{cfg['speed_jitter']})")
    if cfg['occlusion_prob'] > 0:
        parts.append(f"occlusion-freeze(p={cfg['occlusion_prob']}, "
                     f"max={cfg['occlusion_max_frames']})")
    if cfg['micro_jitter_px'] > 0:
        parts.append(f"micro-jitter({cfg['micro_jitter_px']}px)")
    return " + ".join(parts) or "none"


def _make_augment_tag(jitter_ratio, scale_jitter, brightness_jitter, contrast_jitter,
                       max_temporal_shift, occlusion_prob, speed_jitter, micro_jitter_px):
    """
    Encodes every augmentation RANGE/config parameter (not the per-track
    sampled value) into a short deterministic string. A change to ANY of
    these -- old or new -- always produces a new cache filename instead
    of reusing one built under different settings.
    """
    return (f"v{AUGMENTATION_SCHEMA_VERSION}"
            f"j{int(round(jitter_ratio * 1000)):03d}"
            f"s{int(round(scale_jitter * 1000)):03d}"
            f"b{int(round(brightness_jitter * 1000)):03d}"
            f"c{int(round(contrast_jitter * 1000)):03d}"
            f"t{int(max_temporal_shift):02d}"
            f"o{int(round(occlusion_prob * 1000)):03d}"
            f"sp{int(round(speed_jitter * 1000)):03d}"
            f"m{int(micro_jitter_px):02d}")


def make_synthetic_ped_id(orig_ped_id, jitter_ratio, scale_jitter,
                           brightness_jitter, contrast_jitter,
                           max_temporal_shift, occlusion_prob, speed_jitter, micro_jitter_px,
                           copy_index, flipped,
                           brightness_factor, contrast_factor):
    """
    Builds the synthetic ped_id for one augmented copy of a track.
    Same asymmetry as before: jitter_ratio/scale_jitter/max_temporal_shift/
    occlusion_prob/speed_jitter/micro_jitter_px are CONFIG RANGES (the
    actual per-track/per-frame sampled values are baked directly into the
    returned bbox/frame-index list upstream, so they don't need to be
    recovered later). brightness_factor/contrast_factor/flipped are the
    actual PER-TRACK SAMPLED values, because they must be re-applied to
    pixels downstream in load_images_and_process().
    """
    tag = _make_augment_tag(jitter_ratio, scale_jitter, brightness_jitter, contrast_jitter,
                             max_temporal_shift, occlusion_prob, speed_jitter, micro_jitter_px)
    flip_part = "f1" if flipped else "f0"
    b_part = "br" + _encode_signed(brightness_factor - 1.0)
    c_part = "ct" + _encode_signed(contrast_factor - 1.0)
    return (f"{orig_ped_id}{_AUG_MARKER}_{tag}_{flip_part}"
            f"_{b_part}_{c_part}_{copy_index}")


def is_flip_augmented(ped_id_str):
    if _AUG_MARKER not in ped_id_str:
        return False
    return "_f1_" in ped_id_str


def get_photometric_factors(ped_id_str):
    if _AUG_MARKER not in ped_id_str:
        return 1.0, 1.0
    aug_part = ped_id_str.split(_AUG_MARKER, 1)[1]
    b_factor, c_factor = 1.0, 1.0
    for part in aug_part.split('_'):
        if part.startswith('br') and len(part) > 2:
            b_factor = 1.0 + _decode_signed(part[2:])
        elif part.startswith('ct') and len(part) > 2:
            c_factor = 1.0 + _decode_signed(part[2:])
    return b_factor, c_factor


# ----------------------------------------------------------------------
# NEW: temporal helpers. All of them work on FRAME INDICES into the
# track's own already-jittered bbox/image/label lists -- they never
# invent a bbox that doesn't already exist for some real frame, which is
# what keeps them cache-safe (see module docstring).
# ----------------------------------------------------------------------

def _resample_indices_for_speed(n_frames, speed_factor):
    """
    Returns a new list of indices into [0, n_frames) simulating playing
    the track at `speed_factor`x speed:
      speed_factor > 1.0 -> faster (fewer output positions, some real
          frames get skipped)
      speed_factor < 1.0 -> slower (more output positions, some real
          frames get repeated)
    Every returned index is a REAL frame index (nearest-neighbor
    rounding, no interpolation) -- so every output position reuses an
    EXISTING (image, bbox) pair exactly, never an invented one.
    """
    n_out = max(1, int(round(n_frames / speed_factor)))
    return [min(n_frames - 1, int(round(i * speed_factor))) for i in range(n_out)]


def _apply_occlusion_freeze(indices, occlusion_prob, max_occlusion_frames, rng):
    """
    With probability occlusion_prob, picks one point in the (already
    resampled) index list and "freezes" the next 1..max_occlusion_frames
    positions on the last-seen real index -- simulating a real tracker
    losing the pedestrian for a couple of frames and holding the last
    known box. Length is UNCHANGED (frames are frozen, not dropped), and
    every frozen position still points to a real, already-cached-able
    (image, bbox) pair (the one it's frozen on).
    """
    if len(indices) < 4 or rng.random() >= occlusion_prob:
        return indices
    out = list(indices)
    start = rng.randint(1, len(out) - 2)
    n_repeat = rng.randint(1, min(max_occlusion_frames, len(out) - start - 1))
    for i in range(start, start + n_repeat):
        out[i] = out[start - 1]
    return out


def augment_pie_tracks(beh_seq, num_augments=2,
                        jitter_ratio=0.05, scale_jitter=0.05,
                        hflip_prob=0.5,
                        brightness_jitter=0.2, contrast_jitter=0.2,
                        max_temporal_shift=2,
                        occlusion_prob=0.25, occlusion_max_frames=2,
                        speed_jitter=0.15,
                        micro_jitter_px=3,
                        min_track_len=15,
                        img_w=PIE_IMG_WIDTH, img_h=PIE_IMG_HEIGHT,
                        seed=42, target_label_type='intention_binary',
                        augment_only_label=None):
    """
    Expands a PIE-format sequence dict with `num_augments` synthetic
    copies of each eligible track, combining:
      - per-track geometric (jitter/scale/flip) -- unchanged from v2
      - per-track photometric (brightness/contrast) -- unchanged from v2
      - per-copy random start-offset (up to max_temporal_shift frames)
      - per-copy discrete speed resampling (rate in
        [1-speed_jitter, 1+speed_jitter])
      - per-copy occlusion-freeze simulation (probability occlusion_prob,
        up to occlusion_max_frames consecutive frozen frames)
      - per-frame micro-jitter (independent small pixel noise on top of
        the track-level jitter, up to micro_jitter_px in each direction)

    :param max_temporal_shift: max frames to drop from the start of a
        synthetic copy (0 disables this augmentation).
    :param occlusion_prob: probability a given synthetic copy gets an
        occlusion-freeze applied (0.0 disables this augmentation).
    :param occlusion_max_frames: max consecutive frames frozen when
        occlusion triggers.
    :param speed_jitter: max fractional speed change, e.g. 0.15 means
        each copy's speed factor is drawn from [0.85, 1.15]
        (0.0 disables this augmentation).
    :param micro_jitter_px: max independent per-frame pixel noise added
        on top of the track-level jitter (0 disables this augmentation).
    :param min_track_len: safety floor -- if start-offset + speed
        resampling would shrink a copy below this many frames, that
        copy falls back to no shift/no speed-change (occlusion and
        micro-jitter still apply), so windowing downstream never breaks.
    """
    rng = random.Random(seed)

    out = {k: list(v) for k, v in beh_seq.items()}
    n_tracks = len(beh_seq['image'])
    n_added = 0

    for t in range(n_tracks):
        if augment_only_label is not None:
            track_label = beh_seq[target_label_type][t][0][0]
            if track_label != augment_only_label:
                continue

        orig_ped_id = beh_seq['ped_id'][t][0][0]
        n_frames = len(beh_seq['image'][t])

        for k in range(num_augments):
            # --- existing geometric parameters (baked into bbox) ---
            dx_ratio = rng.uniform(-jitter_ratio, jitter_ratio)
            dy_ratio = rng.uniform(-jitter_ratio, jitter_ratio)
            scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
            flipped = rng.random() < hflip_prob

            # --- existing photometric parameters (carried via ped_id) ---
            brightness_factor = 1.0 + rng.uniform(-brightness_jitter, brightness_jitter)
            contrast_factor = 1.0 + rng.uniform(-contrast_jitter, contrast_jitter)

            # Build the track-level jittered/scaled/flipped bbox for
            # EVERY real frame first (unchanged from v2) -- temporal
            # helpers below only ever pick FROM this list by index.
            track_bboxes = []
            for b in beh_seq['bbox'][t]:
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                cx = (x1 + x2) / 2.0 + dx_ratio * w
                cy = (y1 + y2) / 2.0 + dy_ratio * h
                new_w, new_h = w * scale, h * scale

                nx1 = max(0.0, cx - new_w / 2.0)
                ny1 = max(0.0, cy - new_h / 2.0)
                nx2 = min(img_w, cx + new_w / 2.0)
                ny2 = min(img_h, cy + new_h / 2.0)

                if flipped:
                    nx1, nx2 = img_w - nx2, img_w - nx1

                track_bboxes.append([nx1, ny1, nx2, ny2])

            # --- NEW: speed resampling + start-shift, with a safe fallback ---
            speed_factor = 1.0 + rng.uniform(-speed_jitter, speed_jitter) if speed_jitter > 0 else 1.0
            resampled = _resample_indices_for_speed(n_frames, speed_factor) if speed_jitter > 0 else list(range(n_frames))

            shift = 0
            if max_temporal_shift > 0:
                max_shift_here = min(max_temporal_shift, max(0, len(resampled) - min_track_len))
                shift = rng.randint(0, max_shift_here) if max_shift_here > 0 else 0

            candidate = resampled[shift:]
            if len(candidate) < min_track_len:
                # Fallback: speed/shift would break windowing downstream --
                # disable just these two dimensions for this copy, keep
                # geometric/photometric/occlusion/micro-jitter as sampled.
                candidate = list(range(n_frames))

            # --- NEW: occlusion-freeze simulation ---
            final_indices = _apply_occlusion_freeze(
                candidate, occlusion_prob, occlusion_max_frames, rng)

            # --- NEW: per-frame micro-jitter, applied AFTER index selection ---
            # One micro-jitter offset per REAL frame index, reused on repeats,
            # so repeated frames get identical bbox == identical cache entry.
            micro_offsets = {}
            new_bboxes = []
            for idx in final_indices:
                x1, y1, x2, y2 = track_bboxes[idx]
                if micro_jitter_px > 0:
                    if idx not in micro_offsets:
                        micro_offsets[idx] = (rng.uniform(-micro_jitter_px, micro_jitter_px),
                                            rng.uniform(-micro_jitter_px, micro_jitter_px))
                    mdx, mdy = micro_offsets[idx]
                    x1 = min(max(0.0, x1 + mdx), img_w)
                    x2 = min(max(0.0, x2 + mdx), img_w)
                    y1 = min(max(0.0, y1 + mdy), img_h)
                    y2 = min(max(0.0, y2 + mdy), img_h)
                new_bboxes.append([x1, y1, x2, y2])

            synthetic_ped_id = make_synthetic_ped_id(
                orig_ped_id, jitter_ratio, scale_jitter,
                brightness_jitter, contrast_jitter,
                max_temporal_shift, occlusion_prob, speed_jitter, micro_jitter_px,
                k, flipped, brightness_factor, contrast_factor)
            new_ped_id_seq = [[synthetic_ped_id]] * len(final_indices)

            # Re-index every per-frame track by final_indices. bbox is the
            # jittered box, not the original. center is recomputed from that
            # box, including flip and micro-jitter. Schema stays 4: this does
            # not change image, bbox, or ped_id, so old VGG cache entries stay valid.
            out['image'].append([beh_seq['image'][t][i] for i in final_indices])
            out['bbox'].append(new_bboxes)
            out['ped_id'].append(new_ped_id_seq)
            if 'center' in beh_seq:
                out['center'].append([
                    [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in new_bboxes])
            for key, tracks in beh_seq.items():
                if key in ('image', 'bbox', 'ped_id', 'center'):
                    continue
                track = tracks[t]
                if not isinstance(track, (list, tuple)) or len(track) != n_frames:
                    raise ValueError(
                        f"augment_pie_tracks: {key!r} is not a per-frame track "
                        f"of length {n_frames}")
                out[key].append([track[i] for i in final_indices])
            n_added += 1

    print(f"[augment_pie_tracks] schema={AUGMENTATION_SCHEMA_VERSION} "
          f"jitter_ratio={jitter_ratio} scale_jitter={scale_jitter} "
          f"brightness_jitter={brightness_jitter} contrast_jitter={contrast_jitter} "
          f"hflip_prob={hflip_prob} max_temporal_shift={max_temporal_shift} "
          f"occlusion_prob={occlusion_prob} occlusion_max_frames={occlusion_max_frames} "
          f"speed_jitter={speed_jitter} micro_jitter_px={micro_jitter_px} seed={seed}: "
          f"{n_tracks} original tracks -> {len(out['image'])} total "
          f"tracks (+{n_added} synthetic copies)")

    return out