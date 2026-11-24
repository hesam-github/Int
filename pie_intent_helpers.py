"""
pie_intent_helpers.py

Standalone helper functions used by PIEIntent: progress printing, metrics
computation, trace/GPU logging, the streaming tf.data.Dataset builder,
and visualization output.
"""

import os
import json
import subprocess
import threading
from datetime import datetime

import gc
import numpy as np
import tensorflow as tf
import cv2

from utils import update_progress


def _safe_update_progress(progress):
    """Prints progress only from the main thread, to avoid Jupyter's
    per-thread ZMQ socket leak when called from a background thread."""
    if threading.current_thread() is threading.main_thread():
        update_progress(progress)


def compute_full_metrics(y_true, y_prob, save_path=None, threshold=0.5):
    """Computes accuracy/precision/recall/f1/roc_auc/pr_auc/nll/ece/
    brier_score/confusion_matrix at the given decision threshold, and
    optionally saves the result as JSON."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    )

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    positive_rate_gt = float(y_true.mean())
    majority_acc = max(positive_rate_gt, 1 - positive_rate_gt)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')

    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    nll = log_loss(y_true, y_prob_clipped, labels=[0, 1])

    def expected_calibration_error(y_true, y_prob, n_bins=15):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            bin_conf = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += (mask.sum() / len(y_prob)) * abs(bin_conf - bin_acc)
        return float(ece)

    ece = expected_calibration_error(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    balanced_acc = 0.5 * (float(recall) + specificity)
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    metrics = {
        'accuracy': float(accuracy), 'majority_acc': float(majority_acc),
        'precision': float(precision), 'recall': float(recall),
        'specificity': specificity, 'balanced_acc': balanced_acc,
        'f1': float(f1),
        'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc),
        'nll': float(nll), 'ece': float(ece), 'brier_score': brier_score,
        'confusion_matrix': cm.tolist(), 'n_test_samples': int(n),
        'positive_rate_gt': positive_rate_gt,
        'threshold_used': float(threshold),
    }
    
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as fid:
            json.dump(metrics, fid, indent=2)
    return metrics


CACHE_READ_WORKERS = 8
_TRACE_LOG_PATH = os.path.join(os.environ['PIE_PATH'], 'trace_main.log')
_GPU_LOG_PATH = os.path.join(os.environ['PIE_PATH'], 'trace_gpu.log')

_trace_log_fid = open(_TRACE_LOG_PATH, 'a', buffering=1, encoding='utf-8')
_trace_log_lock = threading.Lock()


def trace_log(msg):
    """Appends one timestamped line to trace_main.log, flushed and
    fsynced immediately so it survives a hard crash."""
    line = f"[{datetime.now().strftime('%H:%M:%S.%f')}] {msg}"
    with _trace_log_lock:
        _trace_log_fid.write(line + "\n")
        _trace_log_fid.flush()
        os.fsync(_trace_log_fid.fileno())


_gpu_watchdog_stop_event = threading.Event()
_gpu_watchdog_thread = None


def _gpu_watchdog_loop():
    """Polls nvidia-smi every 2s and logs utilization/memory/temperature
    to trace_gpu.log until stopped."""
    with open(_GPU_LOG_PATH, 'a', buffering=1, encoding='utf-8') as gfid:
        while not _gpu_watchdog_stop_event.is_set():
            ts = datetime.now().strftime('%H:%M:%S.%f')
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5,
                )
                line = f"[{ts}] {result.stdout.strip()}"
            except Exception as e:
                line = f"[{ts}] nvidia-smi FAILED: {type(e).__name__}: {e}"
            gfid.write(line + "\n")
            gfid.flush()
            os.fsync(gfid.fileno())
            _gpu_watchdog_stop_event.wait(2.0)


def start_gpu_watchdog():
    """Starts the GPU-polling background thread. Idempotent."""
    global _gpu_watchdog_thread
    if _gpu_watchdog_thread is not None and _gpu_watchdog_thread.is_alive():
        return
    _gpu_watchdog_stop_event.clear()
    _gpu_watchdog_thread = threading.Thread(target=_gpu_watchdog_loop, daemon=True)
    _gpu_watchdog_thread.start()
    trace_log(f"GPU watchdog started, writing to {_GPU_LOG_PATH}")


def stop_gpu_watchdog():
    """Stops the GPU-polling background thread. Safe if never started."""
    _gpu_watchdog_stop_event.set()
    trace_log("GPU watchdog stop requested")


def _windowed_frame_generator(model, images, bboxes, ped_ids, decoder_input,
                              output, save_path, batch_size, data_type_label, shuffle):
    """Yields one preprocessed (encoder_input, decoder_input, label)
    mini-batch at a time, instead of materializing the full split."""
    n = len(images)
    order = np.random.permutation(n) if shuffle else np.arange(n)
    GC_EVERY_N_BATCHES = 20
    batch_count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = order[start:end]

        batch_images = [images[i] for i in idx]
        batch_bboxes = [bboxes[i] for i in idx]
        batch_ped_ids = [ped_ids[i] for i in idx]

        batch_features = model.load_images_and_process(
            batch_images, batch_bboxes, batch_ped_ids,
            data_type=data_type_label, save_path=save_path,
        )

        batch_decoder = np.asarray(decoder_input[idx], dtype=np.float32)
        batch_output = np.asarray(output[idx], dtype=np.float32)

        assert batch_output.ndim == 2 and batch_output.shape[1] in (1, 3), (
            f"Expected batch_output shape (batch, 1) or (batch, 3), got {batch_output.shape}.")

        yield batch_features.astype(np.float32, copy=False), batch_decoder, batch_output

        batch_count += 1
        if batch_count % GC_EVERY_N_BATCHES == 0:
            gc.collect()


def build_generator_dataset(model, images, bboxes, ped_ids, decoder_input,
                            output, save_path, seq_length, feature_shape,
                            batch_size, data_type_label, shuffle,
                            private_threadpool_size=4, prefetch_size=1):
    """Wraps _windowed_frame_generator in a tf.data.Dataset with a
    restricted internal threadpool, so it can be passed to model.fit()
    or model.predict()."""
    def gen():
        for feat_batch, dec_batch, y_batch in _windowed_frame_generator(
                model, images, bboxes, ped_ids, decoder_input, output,
                save_path, batch_size, data_type_label, shuffle):
            yield (feat_batch, dec_batch), y_batch

    output_signature = (
        (
            tf.TensorSpec(shape=(None, seq_length) + feature_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None, decoder_input.shape[1], decoder_input.shape[2]), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, int(np.shape(output)[-1])), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    options = tf.data.Options()
    options.threading.private_threadpool_size = private_threadpool_size
    options.threading.max_intra_op_parallelism = private_threadpool_size
    ds = ds.with_options(options)
    ds = ds.prefetch(prefetch_size)
    return ds


def save_visualizations(vis_results, model_path, results_suffix,
                        num_correct=10, num_incorrect=10, seed=42):
    """Draws the ground-truth bbox and a prediction/label overlay on a
    random sample of correctly/incorrectly predicted tracks, and saves
    them as PNGs under model_path/visualizations<results_suffix>/."""
    import random

    if not vis_results:
        print("[visualize] vis_results is empty -- nothing to save.")
        return

    out_dir = os.path.join(model_path, f'visualizations{results_suffix}')
    os.makedirs(out_dir, exist_ok=True)

    correct_samples = []
    incorrect_samples = []
    for entry in vis_results:
        pred_prob = float(np.asarray(entry['res']).ravel()[0])
        gt = int(np.asarray(entry['target']).ravel()[0])
        pred_label = int(round(pred_prob))
        if pred_label == gt:
            correct_samples.append(entry)
        else:
            incorrect_samples.append(entry)

    rng = random.Random(seed)
    chosen_correct = rng.sample(correct_samples, min(num_correct, len(correct_samples)))
    chosen_incorrect = rng.sample(incorrect_samples, min(num_incorrect, len(incorrect_samples)))

    print(f"[visualize] saving {len(chosen_correct)} correct + "
        f"{len(chosen_incorrect)} incorrect images to {out_dir}")

    def _draw_and_save(entry, tag, idx):
        imp = entry['imp']
        bbox = entry['bbox']
        ped_id = entry['ped_id']
        pred_prob = float(np.asarray(entry['res']).ravel()[0])
        gt = int(np.asarray(entry['target']).ravel()[0])
        pred_label = int(round(pred_prob))

        img = cv2.imread(imp)
        if img is None:
            print(f"[visualize] WARNING: could not read image {imp}, skipping.")
            return

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        box_color = (0, 200, 0) if tag == 'correct' else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        text_lines = [
            f"ped_id={ped_id}",
            f"pred_prob={pred_prob:.3f} pred_label={pred_label}",
            f"gt={gt}  [{tag.upper()}]",
        ]
        text_y = max(20, y1 - 10 - 20 * (len(text_lines) - 1))
        for line in text_lines:
            cv2.putText(img, line, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, box_color, 2, cv2.LINE_AA)
            text_y += 20

        out_name = f"{tag}_{idx:03d}_ped{ped_id}.png"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, img)

    for idx, entry in enumerate(chosen_correct):
        _draw_and_save(entry, 'correct', idx)
    for idx, entry in enumerate(chosen_incorrect):
        _draw_and_save(entry, 'incorrect', idx)

    print(f"[visualize] done. {len(chosen_correct)} correct + "
        f"{len(chosen_incorrect)} incorrect images saved.")