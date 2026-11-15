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