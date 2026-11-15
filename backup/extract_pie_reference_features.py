"""
extract_pie_reference_features.py  (robust / resumable / GPU-batched version)

New in this version - REAL GPU utilization:
    The original PIEIntent.load_images_and_process() calls
    convnet.predict() once PER FRAME with batch size 1:

        expanded_img = np.expand_dims(preprocessed_img, axis=0)
        img_features = convnet.predict(expanded_img)

    This badly underuses the GPU: each predict() call pays fixed overhead
    (Python/TF dispatch, host->device transfer) that dwarfs the actual
    compute for a single 224x224x3 image, so the GPU sits mostly idle
    between calls no matter how fast the card is.

    This version bypasses load_images_and_process()'s inner loop entirely
    and replaces it with a batched pipeline: multiple frames are
    preprocessed (CPU-bound: PIL crop/pad/resize - unavoidably per-image)
    and then run through VGG16 in ONE predict() call per batch (GPU-bound).
    This is what actually lets the GPU do parallel work instead of paying
    per-frame call overhead, and should give a large speedup over the
    frame-by-frame version, especially for the ~60k-frame test split.

    Output file paths, pickle format (unsqueezed (1,H,W,C) array, matching
    what load_images_and_process itself writes), and all the previous
    resumable/corruption-cleanup logic are unchanged, so this stays fully
    compatible with already-cached files from earlier runs.
"""

from __future__ import annotations

import os
import pickle
import time
import numpy as np
import tensorflow as tf

PIE_PATH = r'D:\PIE codes\PIE_dataset'
os.environ['PIE_PATH'] = PIE_PATH

from pie_data_Hesam import PIE
from pie_intent import PIEIntent
from keras.applications import vgg16
from keras.utils import img_to_array, load_img
from utils import jitter_bbox, squarify, img_pad, update_progress

# ── data_opts ────────────────────────────────────────────────────────────
data_opts = {
    'fstride': 1,
    'sample_type': 'all',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'data_split_type': 'default',
    'seq_type': 'intention',
    'min_track_size': 0,
    'max_size_observe': 15,
    'max_size_predict': 5,
    'seq_overlap_rate': 0.5,
    'balance': True,
    'crop_type': 'context',
    'crop_mode': 'pad_resize',
    'encoder_input_type': [],
    'decoder_input_type': ['bbox'],
    'output_type': ['intention_binary'],
}

data_type = {
    'encoder_input_type': data_opts['encoder_input_type'],
    'decoder_input_type': data_opts['decoder_input_type'],
    'output_type':        data_opts['output_type'],
}

SEQ_LENGTH  = data_opts['max_size_observe']
MAX_RETRIES = 10   # max attempts per split
RETRY_WAIT  = 30   # seconds to wait between retries
BATCH_SIZE  = 32   # frames per GPU predict() call; lowered automatically on OOM


# ── GPU setup ─────────────────────────────────────────────────────────

def configure_gpu() -> None:
    """
    Verify a GPU is visible to TensorFlow and configure it for stable use.

    Raises RuntimeError immediately if no GPU is found, so the script
    never silently spends hours running on CPU without the user knowing.
    Enables memory growth so TensorFlow allocates VRAM incrementally
    instead of reserving all of it at startup.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(
            "No GPU detected by TensorFlow. This script requires a working "
            "GPU setup (CUDA 11.2 + cuDNN 8.1 + TensorFlow 2.10 on Windows). "
            "Run:\n"
            "    python -c \"import tensorflow as tf; "
            "print(tf.config.list_physical_devices('GPU'))\"\n"
            "and confirm it returns a non-empty list before retrying."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not set memory growth for {gpu}: {e}")

    print(f"GPU configured and visible to TensorFlow: {gpus}")

    with tf.device('/GPU:0'):
        _ = tf.constant([1.0, 2.0, 3.0]) * 2
    print("GPU sanity check passed - TensorFlow is ready to compute on GPU.")


# ── low level cache-file checks ─────────────────────────────────────────

def check_pkl_exists(path: str) -> bool:
    """Fast check: file exists and is non-empty. Does not open/unpickle it."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def check_pkl_valid(path: str) -> bool:
    """Full check: file exists, can be unpickled, and holds non-empty data.
    Catches ANY unpickling failure, including module-path mismatches from
    files pickled under a different NumPy major version."""
    if not check_pkl_exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray):
            return data.size > 0
        if isinstance(data, (list, tuple)):
            return len(data) > 0
        return data is not None
    except Exception:
        return False


# ── expected cache file reconstruction ──────────────────────────────────

def compute_expected_files(save_path: str, images: list, ped_ids: list) -> set:
    """
    Rebuild the exact set of cache file paths that would be produced,
    one per (frame, pedestrian) pair, at:

        {save_path}/{set_id}/{vid_id}/{img_name}_{ped_id}.pkl
    """
    expected = set()
    for seq, pid_seq in zip(images, ped_ids):
        for imp, p in zip(seq, pid_seq):
            parts = os.path.normpath(imp).split(os.sep)
            set_id = parts[-3]
            vid_id = parts[-2]
            img_name = os.path.splitext(parts[-1])[0]
            ped_id = p[0]
            img_save_folder = os.path.join(save_path, set_id, vid_id)
            img_save_path = os.path.join(img_save_folder, f'{img_name}_{ped_id}.pkl')
            expected.add(img_save_path)
    return expected


def audit_files(expected_files: set, deep: bool = False) -> dict:
    """Compare expected_files against what's actually cached on disk."""
    check_fn = check_pkl_valid if deep else check_pkl_exists
    missing = [f for f in expected_files if not check_fn(f)]
    total = len(expected_files)
    for i, f in enumerate(expected_files):
        if not check_fn(f):
            missing.append(f)
        if deep and (i % 500 == 0 or i == total - 1):
            update_progress((i + 1) / total)
    return {
        'expected':      len(expected_files),
        'cached_valid':  len(expected_files) - len(missing),
        'missing_count': len(missing),
        'is_complete':   len(missing) == 0,
        'missing_files': missing[:20],
        '_all_missing':  missing,
    }


def remove_files(paths) -> None:
    for f in paths:
        try:
            if os.path.isfile(f):
                os.remove(f)
        except OSError:
            pass


def clean_corrupted_cache(expected_files: set) -> int:
    """
    Open every existing file in expected_files and delete any that fail to
    unpickle (truncated writes, or pickles written under a different NumPy
    major version than what's currently installed). Must run before
    extraction, since a broken pickle would otherwise be silently treated
    as "cached" by the fast existence check.
    """
    existing = [f for f in expected_files if check_pkl_exists(f)]
    if not existing:
        return 0

    print(f"Scanning {len(existing)} existing cached files for corruption...")
    corrupted = [f for f in existing if not check_pkl_valid(f)]

    if corrupted:
        print(f"Found {len(corrupted)} corrupted/unreadable cached files. Removing them "
              f"so they get regenerated cleanly...")
        remove_files(corrupted)
    else:
        print("No corrupted cached files found.")

    return len(corrupted)


# ── GPU-batched feature extraction ──────────────────────────────────────

def collect_frame_tasks(images: list, bboxes: list, ped_ids: list,
                         save_path: str, missing_only: set) -> dict:
    """
    Flatten the per-window sequences into a deduplicated dict of per-frame
    extraction tasks: {output_pkl_path: (image_path, bbox)}.

    Overlapping windows (seq_overlap_rate=0.5) share many frames, so this
    naturally deduplicates repeated (frame, pedestrian) pairs - each frame
    is only ever extracted once regardless of how many windows reference it.

    Only tasks whose output path is in missing_only are included, so
    already-cached (and already-validated) frames are never reprocessed.
    """
    tasks = {}
    for seq, bbox_seq, pid_seq in zip(images, bboxes, ped_ids):
        for imp, b, p in zip(seq, bbox_seq, pid_seq):
            parts = os.path.normpath(imp).split(os.sep)
            set_id = parts[-3]
            vid_id = parts[-2]
            img_name = os.path.splitext(parts[-1])[0]
            ped_id = p[0]
            img_save_folder = os.path.join(save_path, set_id, vid_id)
            img_save_path = os.path.join(img_save_folder, f'{img_name}_{ped_id}.pkl')
            if img_save_path not in missing_only:
                continue
            tasks[img_save_path] = (imp, b)
    return tasks


def predict_with_oom_guard(convnet, batch_input: np.ndarray) -> np.ndarray:
    """
    Run convnet.predict on batch_input, halving the batch and retrying if
    a GPU out-of-memory error occurs. This lets extraction keep running at
    the largest batch size the GPU can actually handle (important for
    smaller-VRAM cards like a GTX 1650 Ti) instead of crashing the whole
    script on a single oversized batch.
    """
    n = batch_input.shape[0]
    try:
        return convnet.predict(batch_input, batch_size=n, verbose=0)
    except tf.errors.ResourceExhaustedError:
        if n == 1:
            raise
        mid = n // 2
        print(f"  GPU out of memory at batch size {n}, splitting into "
              f"{mid} + {n - mid} and retrying...")
        first = predict_with_oom_guard(convnet, batch_input[:mid])
        second = predict_with_oom_guard(convnet, batch_input[mid:])
        return np.concatenate([first, second], axis=0)


def extract_features_batched_gpu(model, tasks: dict, batch_size: int = BATCH_SIZE) -> None:
    """
    GPU-batched replacement for PIEIntent.load_images_and_process()'s inner
    per-frame extraction loop. Collects `batch_size` frames at a time,
    preprocesses them on CPU (crop/pad/resize - inherently per-image, PIL-
    based work that can't be batched), stacks them into a single array,
    and runs ONE predict() call per batch. This is the part that actually
    saturates the GPU instead of paying per-frame call overhead.

    Each output file is saved with the same (1, H, W, C) shape that
    load_images_and_process() itself writes (batch dim of 1 kept, squeeze
    happens only on read), so cached files stay interchangeable regardless
    of which version produced them.
    """
    items = list(tasks.items())
    total = len(items)
    if total == 0:
        return

    convnet = model.context_model
    done = 0

    for start in range(0, total, batch_size):
        chunk = items[start:start + batch_size]
        batch_arrays = []
        batch_save_paths = []

        for img_save_path, (imp, b) in chunk:
            try:
                img_data = load_img(imp)
                bbox = jitter_bbox([b], imp, 'enlarge', 2)[0]
                bbox = squarify(bbox, 1, img_data.size[0])
                bbox = list(map(int, bbox[0:4]))
                cropped_image = img_data.crop(bbox)
                img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                image_array = img_to_array(img_data)
                preprocessed_img = vgg16.preprocess_input(image_array)
                batch_arrays.append(preprocessed_img)
                batch_save_paths.append(img_save_path)
            except Exception as e:
                print(f"  Skipping frame due to preprocessing error ({imp}): {e}")
                continue

        if not batch_arrays:
            continue

        batch_input = np.stack(batch_arrays, axis=0)  # (N, 224, 224, 3)
        batch_features = predict_with_oom_guard(convnet, batch_input)

        for i, img_save_path in enumerate(batch_save_paths):
            single_feature = batch_features[i:i + 1]  # keep (1, H, W, C), matches original format
            img_save_folder = os.path.dirname(img_save_path)
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder, exist_ok=True)
            with open(img_save_path, 'wb') as fid:
                pickle.dump(single_feature, fid, pickle.HIGHEST_PROTOCOL)

        done += len(chunk)
        update_progress(min(done, total) / total)


# ── main extraction routine, with resume + retry ────────────────────────

def build_and_cache_robust(imdb, model, image_set: str,
                            overlap: float, apply_balance: bool,
                            max_retries: int = MAX_RETRIES,
                            batch_size: int = BATCH_SIZE) -> bool:
    print(f"\n{'='*60}")
    print(f"SPLIT: {image_set}  overlap={overlap}  balance={apply_balance}")
    print(f"{'='*60}")

    beh_seq = imdb.generate_data_trajectory_sequence(image_set, **data_opts)
    if apply_balance:
        beh_seq = imdb.balance_samples_count(beh_seq, label_type='intention_binary')

    _, images, bboxes, ped_ids = model.get_tracks(beh_seq, data_type, SEQ_LENGTH, overlap)

    save_path = model.get_path(
        type_save='data',
        data_type='features_' + data_opts['crop_type'] + '_' + data_opts['crop_mode'],
        model_name='vgg16_none',
        data_subset=image_set,
    )
    if isinstance(save_path, tuple):
        save_path = save_path[0]

    expected_files = compute_expected_files(save_path, images, ped_ids)
    print(f"save_path : {save_path}")
    print(f"expected  : {len(expected_files)} unique frame-level cache files")

    clean_corrupted_cache(expected_files)

    complete = False
    for attempt in range(1, max_retries + 1):
        print(f"\n--- Attempt {attempt}/{max_retries} ---")

        status = audit_files(expected_files, deep=False)
        print(f"cached  : {status['cached_valid']}")
        print(f"missing : {status['missing_count']}")

        if status['is_complete']:
            print(f"{image_set} already complete (existence check) - skipping extraction.")
            complete = True
            break

        missing_set = set(status['_all_missing'])
        tasks = collect_frame_tasks(images, bboxes, ped_ids, save_path, missing_set)
        print(f"Extracting {len(tasks)} missing frames using batched GPU "
              f"inference (batch_size={batch_size})...")
        try:
            extract_features_batched_gpu(model, tasks, batch_size=batch_size)
        except Exception as e:
            print(f"extract_features_batched_gpu raised: {e}")
            print("Frames written before the failure are already on disk and will not be recomputed.")
            newly_removed = clean_corrupted_cache(expected_files)
            if newly_removed:
                print(f"Removed {newly_removed} more corrupted file(s) found mid-run; "
                      f"will retry immediately.")

        status_after = audit_files(expected_files, deep=False)
        print(f"\nPost-extraction audit: cached={status_after['cached_valid']} "
              f"missing={status_after['missing_count']}")

        if status_after['is_complete']:
            complete = True
            break

        print(f"Still incomplete ({status_after['missing_count']} missing).")
        if status_after['missing_count'] > 0:
            print(f"  Sample missing files: {status_after['missing_files'][:5]}")

        if attempt < max_retries:
            print(f"Waiting {RETRY_WAIT}s before retry...")
            time.sleep(RETRY_WAIT)

    if not complete:
        final_status = audit_files(expected_files, deep=False)
        print(f"\nFAILED after {max_retries} attempts for split={image_set}.")
        with open(f'missing_{image_set}.txt', 'w') as f:
            f.write('\n'.join(final_status['_all_missing']))
        print(f"Missing file list saved to: missing_{image_set}.txt")
        return False

    print(f"\nRunning deep validation for {image_set} (opens every cached file)...")
    deep_status = audit_files(expected_files, deep=True)
    if deep_status['is_complete']:
        print(f"{image_set} verified complete: {deep_status['expected']} valid files.")
        return True

    print(f"Deep validation found {deep_status['missing_count']} corrupted files, "
          f"removing and regenerating them...")
    corrupted_set = set(deep_status['_all_missing'])
    remove_files(corrupted_set)
    tasks = collect_frame_tasks(images, bboxes, ped_ids, save_path, corrupted_set)
    try:
        extract_features_batched_gpu(model, tasks, batch_size=batch_size)
    except Exception as e:
        print(f"extract_features_batched_gpu raised while repairing corrupted files: {e}")

    final_status = audit_files(expected_files, deep=True)
    if final_status['is_complete']:
        print(f"{image_set} COMPLETE after repairing corrupted files.")
        return True

    print(f"FAILED: {final_status['missing_count']} files still missing/invalid after repair.")
    with open(f'missing_{image_set}.txt', 'w') as f:
        f.write('\n'.join(final_status['_all_missing']))
    return False


# ── main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    configure_gpu()

    imdb = PIE(data_path=PIE_PATH)

    model = PIEIntent(
        num_hidden_units=128,
        regularizer_val=0.001,
        lstm_dropout=0.4,
        lstm_recurrent_dropout=0.2,
        convlstm_num_filters=64,
        convlstm_kernel_size=2,
    )
    model.context_model = vgg16.VGG16(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )

    results = {}

    results['train'] = build_and_cache_robust(
        imdb, model, 'train',
        overlap=data_opts['seq_overlap_rate'],
        apply_balance=True,
    )
    results['val'] = build_and_cache_robust(
        imdb, model, 'val',
        overlap=data_opts['seq_overlap_rate'],
        apply_balance=True,
    )
    results['test'] = build_and_cache_robust(
        imdb, model, 'test',
        overlap=1,
        apply_balance=False,
    )

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    all_ok = True
    for split, ok in results.items():
        status = "COMPLETE" if ok else "INCOMPLETE"
        print(f"  {split:8s}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        cache_dir = os.path.join(PIE_PATH, 'data', 'data', 'pie')
        print(f"\nAll splits complete.")
        print(f"Upload this folder to Google Drive:")
        print(f"  {cache_dir}")
        print(f"Also upload:")
        print(f"  {os.path.join(PIE_PATH, 'data_cache', 'pie_database.pkl')}")
        print(f"  {os.path.join(PIE_PATH, 'data_cache')} (all seq_*.pkl files)")
    else:
        print("\nSome splits incomplete - check missing_*.txt files.")