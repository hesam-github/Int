"""
focal_loss.py

Purpose: alternative loss functions to plain binary_crossentropy for the
PIE intention model. Two options are provided:

  binary_focal_loss(gamma, alpha)
      Down-weights easy (well-classified) examples via the
      (1 - p_t)^gamma term, so the model spends more gradient "attention"
      on hard/misclassified examples. Since balance_samples_count()-style
      undersampling (or the project's synthetic-augmentation alternative)
      already fixes the class COUNT imbalance, only the EASY-EXAMPLE
      down-weighting (gamma) adds something those methods don't already
      provide -- alpha alone would just duplicate what balancing already
      does.

  weighted_binary_crossentropy(pos_weight)
      Standard BCE with a FIXED weight on the positive class. Unlike
      focal loss, this does NOT depend on the model's current confidence
      (p_t), so it has none of focal loss's confidence-compression side
      effect on the loss curve or on calibration (ECE). This matches the
      class-weighting approach used in the official PIE benchmark
      follow-up (Kotseruba et al., WACV 2021).

Both return a Keras-compatible fn(y_true, y_pred) -> scalar closure, with
a stable __name__ set for serialization/logging -- without this, every
closure instance would print as the generic "<locals>.loss_fn" and be
impossible to tell apart in logs when comparing runs with different
hyperparameters.
"""

import tensorflow as tf
from keras import backend as K


def binary_focal_loss(gamma=1.0, alpha=0.25):
    """
    :param gamma: focusing parameter (>= 0). Higher gamma = more
        aggressive down-weighting of easy examples. gamma=0 reduces to
        plain alpha-weighted cross-entropy. 2.0 is the standard default
        from the original Focal Loss paper (Lin et al., 2017), but for
        data that is already close to balanced (e.g. via
        augment_pie_tracks or balance_samples_count), a smaller value
        (1.0-1.5) is a safer starting point -- see the discussion on
        why gamma=2.0 caused rapid overfitting on this project's
        already-balanced train split.
    :param alpha: weight for the POSITIVE class (0 < alpha < 1); the
        negative class gets weight (1 - alpha). 0.25 is the paper's
        default for severely imbalanced detection data. If training
        WITHOUT prior balancing (raw class distribution), consider
        setting alpha to the positive class's share of the raw
        distribution instead, e.g.:
            alpha = neg_count / (pos_count + neg_count)
        so the loss itself compensates for the imbalance that
        undersampling/augmentation would otherwise have handled.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)

        # p_t: predicted probability of the TRUE class (not just positive).
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    focal_loss_fixed.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return focal_loss_fixed


def weighted_binary_crossentropy(pos_weight=1.0):
    """
    :param pos_weight: fixed multiplier applied to the positive-class
        term of BCE. pos_weight=1.0 reduces to plain BCE. A common
        choice is neg_count / pos_count on the RAW (unbalanced)
        distribution, so the loss compensates for imbalance without
        any data-level balancing at all.
    """
    def weighted_bce_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)

        bce = -(pos_weight * y_true * tf.math.log(y_pred)
                + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(bce)

    weighted_bce_fixed.__name__ = f"weighted_bce_pw{pos_weight}"
    return weighted_bce_fixed

def composite_intent_loss(gamma=0.0, pos_weight=1.0, lam_rank=0.0, rank_delta=0.5,
                          beta_conf=0.0, eps_smooth=0.0):
    """Switchable composite loss on the sigmoid output.

    pointwise : (soft-target) BCE, optionally focal-modulated.
    rank      : lam_rank * mean over pairs with (y_i - y_j) > rank_delta of
                softplus(-(z_i - z_j)).
    conf      : beta_conf * negative entropy of the prediction.
    eps_smooth: label smoothing toward 0.5.
    All components off except BCE reproduces plain BCE (pos_weight=1).
    """
    def composite_fixed(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        p = K.clip(tf.reshape(y_pred, [-1]), K.epsilon(), 1.0 - K.epsilon())
        t = y * (1.0 - eps_smooth) + 0.5 * eps_smooth
        ce = -(pos_weight * t * tf.math.log(p) + (1.0 - t) * tf.math.log(1.0 - p))
        if gamma > 0:
            p_t = t * p + (1.0 - t) * (1.0 - p)
            ce = ce * tf.pow(1.0 - p_t, gamma)
        total = tf.reduce_mean(ce)
        if lam_rank > 0:
            z = tf.math.log(p) - tf.math.log(1.0 - p)
            diff = z[:, None] - z[None, :]
            pair = tf.cast((y[:, None] - y[None, :]) > rank_delta, tf.float32)
            n_pairs = tf.maximum(tf.reduce_sum(pair), 1.0)
            total += lam_rank * tf.reduce_sum(tf.nn.softplus(-diff) * pair) / n_pairs
        if beta_conf > 0:
            total += beta_conf * tf.reduce_mean(
                p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p))
        return total

    parts = ["cmp"]
    if gamma > 0:
        parts.append(f"g{gamma}")
    if pos_weight != 1.0:
        parts.append(f"pw{pos_weight}")
    if lam_rank > 0:
        parts.append(f"r{lam_rank}")
        if rank_delta != 0.5:
            parts.append(f"d{rank_delta}")
    if beta_conf > 0:
        parts.append(f"c{beta_conf}")
    if eps_smooth > 0:
        parts.append(f"s{eps_smooth}")
    composite_fixed.__name__ = "_".join(parts)
    return composite_fixed

def bce_plus_focal_loss(gamma=2.0, focal_weight=1.0, pos_weight=1.0, alpha=None):
    """loss = weighted_BCE + focal_weight * focal.

    weighted_BCE uses pos_weight on the positive class (same as weighted_binary_crossentropy).
    focal = (1 - p_t)^gamma * CE, optionally multiplied by alpha_t.
    pos_weight=1.0 and alpha=None recovers plain BCE + focal.
    focal_weight=0 recovers pure weighted_BCE.
    """
    def bce_focal_fixed(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        p = K.clip(tf.reshape(y_pred, [-1]), K.epsilon(), 1.0 - K.epsilon())
        p_t = y * p + (1.0 - y) * (1.0 - p)
        # weighted BCE term
        ce = -(pos_weight * y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        # focal term (on the unweighted CE of the true class, then optionally alpha)
        focal = tf.pow(1.0 - p_t, gamma) * (-tf.math.log(p_t))
        if alpha is not None:
            focal = focal * (y * alpha + (1.0 - y) * (1.0 - alpha))
        return tf.reduce_mean(ce + focal_weight * focal)

    name = f"wbce_focal_g{gamma}_w{focal_weight}_pw{pos_weight}"
    if alpha is not None:
        name += f"_a{alpha}"
    bce_focal_fixed.__name__ = name
    return bce_focal_fixed

def class_balanced_categorical_focal(class_weights, gamma=1.0, focal_weight=1.0):
    """Weighted categorical CE plus the same CE modulated by (1 - p_t)^gamma.

    y_true and y_pred are (batch, 3): one-hot and softmax.
    class_weights has one positive number per class, in order nci, ui, ci.
    gamma=0 and focal_weight=0 with equal weights is plain categorical CE.
    """
    weights = tf.constant([float(v) for v in class_weights], dtype=tf.float32)

    def c3_focal_fixed(y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        p = K.clip(tf.cast(y_pred, tf.float32), K.epsilon(), 1.0 - K.epsilon())
        w = tf.reduce_sum(y * weights, axis=-1)
        ce = -tf.reduce_sum(y * tf.math.log(p), axis=-1)
        p_t = tf.reduce_sum(y * p, axis=-1)
        focal = tf.pow(1.0 - p_t, gamma) * ce
        return tf.reduce_mean(w * ce + focal_weight * w * focal)

    c3_focal_fixed.__name__ = (
        f"c3_wce_focal_g{gamma}_fw{focal_weight}_"
        + "_".join(f"{float(v):.3f}" for v in class_weights)
    )
    return c3_focal_fixed