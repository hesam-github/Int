"""Decoder matrix blocks and their scaling.

Columns 0:4 = main bbox (scaled by normalize_bbox in PIEIntent).
Every other selected feature is one block, in menu order (feature_spans).
Each block gets its own scaling from feature_norm: none / bbox_style / zscore.

feature_target
  decoder  extras stay in the decoder LSTM only (PIE paper)
  encoder  extras tiled over the 7x7 map as extra ConvLSTM channels
  both     tiled into ConvLSTM and also concatenated into the decoder LSTM
  (branchattn ignores this and always uses decoder: every block has its own branch)

fusion
  pie                  one ConvLSTM, repeat, concat, one LSTM, sigmoid
  pcpa                 ConvLSTM branch + bbox LSTM + ONE shared extras LSTM, then attention
  branchattn           sub-menu branch_mode, below
  branchattn_temporal  same sub-menu; temporal attention only where that mode says so

branch_mode, only when fusion is branchattn or branchattn_temporal
  static_pool        Dense on the Flatten vector, one LSTM per block, BranchAttnPool
  temporal_concat    RepeatVector of the raw Flatten, concat the whole decoder, one LSTM.
                     Same injection as fusion='pie'. Extra columns stay inside that LSTM.
  core_plus_extras   RepeatVector of the raw Flatten concatenated with bbox only, one core LSTM.
                     Each later block is its own LSTM, then BranchAttnPool.
                     Temporal attention is applied only to those extra LSTMs.

use_visual=False -> decoder-only model, no VGG16 input at all.
"""
import numpy as np
import tensorflow as tf
from keras.layers import Concatenate, ConvLSTM2D, Dense, Flatten, Input, LSTM, RepeatVector, Conv2D, TimeDistributed
from keras.models import Model
from visual_norm import ChannelStandardize, visual_op_order
from utils import PIE_IMG_WIDTH, PIE_IMG_HEIGHT
from visual_norm import ChannelStandardize
from visual_percentile import PositivePercentileScale

class SliceColumns(tf.keras.layers.Layer):
    def __init__(self, start, end=None, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = None if end is None else int(end)

    def call(self, x):
        return x[:, :, self.start:self.end]

    def get_config(self):
        config = super().get_config()
        config.update(start=self.start, end=self.end)
        return config


class BroadcastExtraChannels(tf.keras.layers.Layer):
    """(B,T,H,W,C) + (B,T,F) -> (B,T,H,W,C+F). F is constant across the map."""

    def call(self, inputs):
        maps, extra = inputs
        shape = tf.shape(maps)
        extra = extra[:, :, tf.newaxis, tf.newaxis, :]
        extra = tf.tile(extra, [1, 1, shape[2], shape[3], 1])
        return tf.concat([maps, extra], axis=-1)


class BranchAttention(tf.keras.layers.Layer):
    """Softmax over branch vectors of equal width. Output is one vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = Dense(1, name='branch_score')

    def call(self, branches):
        stacked = tf.stack(branches, axis=1)
        logits = tf.squeeze(self.score(stacked), axis=-1)
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(stacked * weights[:, :, tf.newaxis], axis=1)

    def get_config(self):
        return super().get_config()


class AdditiveTemporalAttention(tf.keras.layers.Layer):
    """(B,T,D) -> (B,D): softmax over time with an additive (tanh) score."""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.hidden = Dense(units, activation='tanh')
        self.score = Dense(1, use_bias=False)

    def call(self, seq):
        w = tf.nn.softmax(tf.squeeze(self.score(self.hidden(seq)), -1), axis=1)
        return tf.reduce_sum(seq * w[:, :, tf.newaxis], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(units=self.units)
        return config


class BranchAttnPool(tf.keras.layers.Layer):
    """Softmax over N equal-width branch vectors, one score head PER branch."""

    def __init__(self, n_branches, return_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.n_branches = int(n_branches)
        self.return_weights = return_weights
        self.scores = [Dense(1, name=f'score_{i}') for i in range(self.n_branches)]

    def call(self, branches):
        logits = tf.concat([s(b) for s, b in zip(self.scores, branches)], axis=1)
        w = tf.nn.softmax(logits, axis=1)
        pooled = tf.reduce_sum(tf.stack(branches, axis=1) * w[:, :, tf.newaxis], axis=1)
        return [pooled, w] if self.return_weights else pooled

    def get_config(self):
        config = super().get_config()
        config.update(n_branches=self.n_branches, return_weights=self.return_weights)
        return config

class Log1p(tf.keras.layers.Layer):
    """log(1 + max(x, 0)). VGG16 features are >= 0 after ReLU; the clamp only guards float noise."""

    def call(self, x):
        return tf.math.log1p(tf.maximum(x, 0.0))

    def get_config(self):
        return super().get_config()

CUSTOM_OBJECTS = {
    'SliceColumns': SliceColumns,
    'BroadcastExtraChannels': BroadcastExtraChannels,
    'BranchAttention': BranchAttention,
    'AdditiveTemporalAttention': AdditiveTemporalAttention,
    'BranchAttnPool': BranchAttnPool,
    'ChannelStandardize': ChannelStandardize,
    'Log1p': Log1p,
    'PositivePercentileScale': PositivePercentileScale,
}


def bbox_log_ratio(block):
    """Same transform as PIEIntent._normalize_decoder_bbox, for any (..., 4) raw block."""
    x = np.asarray(block, dtype=np.float32)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    w = np.maximum(x2 - x1, 1.0) / PIE_IMG_WIDTH
    h = np.maximum(y2 - y1, 1.0) / PIE_IMG_HEIGHT
    out = np.empty_like(x)
    out[..., 0] = 0.5 * (x1 + x2) / PIE_IMG_WIDTH
    out[..., 1] = y2 / PIE_IMG_HEIGHT
    out[..., 2] = np.log(h) / 5.0
    out[..., 3] = np.log(w) / 5.0
    return out


def scale_side_features(model, decoder_input, fit=False):
    """LEGACY global z-score on columns after bbox. Used only by runs without feature_norm."""
    mode = getattr(model, 'extra_norm', 'none')
    array = np.asarray(decoder_input, dtype=np.float32)
    if mode != 'zscore' or array.ndim != 3 or array.shape[-1] <= 4:
        return array
    extra = array[..., 4:]
    if fit or getattr(model, 'extra_norm_stats', None) is None:
        flat = extra.reshape(-1, extra.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1.0
        model.extra_norm_stats = {
            'mean': mean.astype(np.float64).tolist(),
            'std': std.astype(np.float64).tolist(),
        }
        print(f"[config] extra z-score fit on {flat.shape[0]} rows, "
              f"width={flat.shape[1]} (bbox columns left raw)")
    mean = np.asarray(model.extra_norm_stats['mean'], dtype=np.float32)
    std = np.asarray(model.extra_norm_stats['std'], dtype=np.float32)
    out = array.copy()
    out[..., 4:] = (extra - mean) / std
    return out


def apply_feature_norms(model, decoder_input, spans, fit=False):
    """Per-block scaling from the menu (model.feature_norm).
    Blocks named bbox* are zeroed when model.zero_decoder_bbox is set (visual-only diagnostic).
    'bbox' itself is otherwise skipped: PIEIntent._normalize_decoder_bbox already handled it.
    Runs without feature_norm (old runs) use the legacy scale_side_features()."""
    norms = getattr(model, 'feature_norm', None)
    if norms is None:
        return scale_side_features(model, decoder_input, fit=fit)

    zero_bbox = getattr(model, 'zero_decoder_bbox', False)
    array = np.asarray(decoder_input, dtype=np.float32).copy()
    stats = dict(getattr(model, 'extra_norm_stats', None) or {})
    for key, start, end in (spans or []):
        if key.startswith('bbox') and zero_bbox:
            array[..., start:end] = 0.0
            continue
        if key == 'bbox':
            continue
        mode = norms.get(key, 'none')
        if mode == 'none':
            continue
        block = array[..., start:end]
        if mode == 'bbox_style':
            array[..., start:end] = bbox_log_ratio(block)
        elif mode == 'zscore':
            if fit:
                flat = block.reshape(-1, end - start)
                std = flat.std(axis=0)
                std[std < 1e-6] = 1.0
                stats[key] = {'mean': flat.mean(axis=0).tolist(), 'std': std.tolist()}
                print(f"[config] z-score '{key}' fit on {flat.shape[0]} rows (train only)")
            elif key not in stats:
                raise RuntimeError(
                    f"no train statistics for '{key}': this model was not trained with "
                    f"that feature. Refusing to fit on val/test.")
            mean = np.asarray(stats[key]['mean'], dtype=np.float32)
            std = np.asarray(stats[key]['std'], dtype=np.float32)
            array[..., start:end] = (block - mean) / std
        elif isinstance(mode, str) and mode.startswith('x') and mode[1:].replace('.', '', 1).isdigit():
            factor = float(mode[1:])
            array[..., start:end] = block * factor
            if fit:
                print(f"[config] constant scale '{key}' x{factor}")
        else:
            raise ValueError(f"unknown feature_norm {mode!r} for {key!r}")
    model.extra_norm_stats = stats
    return array


def restore_routing(model, hyperparams):
    """Copy routing flags saved at train time onto a reloaded experiment.
    Old runs: no 'use_visual' -> True; no 'feature_norm' -> None (legacy scaling)."""
    if not hyperparams:
        return
    model.feature_target = hyperparams.get('feature_target', 'decoder')
    model.fusion = hyperparams.get('fusion', 'pie')
    model.branch_mode = hyperparams.get('branch_mode')
    model.temporal_fusion = hyperparams.get('temporal_fusion', 'pie_concat')
    model.extra_norm = hyperparams.get('extra_norm', 'none')
    model.extra_norm_stats = hyperparams.get('extra_norm_stats')
    model.feature_norm = hyperparams.get('feature_norm')
    model.use_visual = hyperparams.get('use_visual', True)
    model.capacity = hyperparams.get('capacity')
    model.visual_norm = hyperparams.get('visual_norm')
    model.percentile_scale = hyperparams.get('percentile_scale')
    model.pose_scale = hyperparams.get('pose_scale', 'bbox_relative')
    model.zero_decoder_bbox = hyperparams.get('zero_decoder_bbox', False)
    spans = hyperparams.get('feature_spans')
    model.feature_spans = [tuple(s) for s in spans] if spans else None


def _build_decoder_only(intent):
    """Decoder-only model: one LSTM over the decoder matrix. No VGG16 input."""
    width = int(intent._decoder_input_size)
    dec_in = Input(shape=(intent._decoder_seq_length, width), name='decoder_input')
    hidden = LSTM(
        units=intent._num_hidden_units,
        dropout=intent._lstm_dropout,
        recurrent_dropout=intent._lstm_recurrent_dropout,
        kernel_regularizer=intent._kernel_regularizer,
        recurrent_regularizer=intent._recurrent_regularizer,
        bias_regularizer=intent._bias_regularizer,
        activation=intent._activation,
        recurrent_activation=intent._lstm_recurrent_activation,
        name='decoder_network')(dec_in)
    output = Dense(
        intent._decoder_dense_output_size, activation='sigmoid',
        kernel_regularizer=intent._kernel_regularizer,
        bias_regularizer=intent._bias_regularizer,
        name='decoder_dense')(hidden)
    print(f"[config] decoder-only model: decoder_width={width}")
    return Model(inputs=dec_in, outputs=output, name='model')


def build_intention_model(intent):
    """Build decoder-only, pie, pcpa or branchattn. Reads routing flags from the PIEIntent instance."""
    if not getattr(intent, 'use_visual', True):
        return _build_decoder_only(intent)

    target = getattr(intent, 'feature_target', 'decoder')
    fusion = getattr(intent, 'fusion', 'pie')
    units = intent._num_hidden_units
    width = int(intent._decoder_input_size)
    extra_width = max(0, width - 4)
    if extra_width == 0 and target != 'decoder':
        print("[config] no extra columns; feature_target forced to decoder")
        target = 'decoder'
    if fusion in ('branchattn', 'branchattn_temporal') and target != 'decoder':
        print("[config] branchattn: feature_target forced to decoder (no tiling onto the VGG map)")
        target = 'decoder'
    attn_weights = None
    print(f"[config] fusion={fusion} feature_target={target} "
          f"decoder_width={width} extra_width={extra_width}")

    maps_in = Input(
        shape=(intent._sequence_length,) + intent.context_model.output_shape[1:],
        name='encoder_input')
    if getattr(intent, 'zero_visual', False):
        maps = tf.zeros_like(maps_in)
    else:
        maps = maps_in

    vn = getattr(intent, 'visual_norm', None)
    if getattr(intent, 'zero_visual', False):
        vn = None
    cap = getattr(intent, 'capacity', None) or {}

    ops = visual_op_order(vn, cap)
    print(f"[visual_pipeline] graph order: VGG -> {' -> '.join(ops) if ops else '(raw)'} -> ConvLSTM")
    for op in ops:
        if op == 'standardize':
            maps = ChannelStandardize(vn['mean'], vn['std'], name='visual_standardize')(maps)
        else:
            maps = Log1p(name='log1p_input')(maps)
    # Optional positive-percentile scaling (no centering, no clipping)
    pscale = getattr(intent, 'percentile_scale', None)
    if pscale and not getattr(intent, 'zero_visual', False):
        from visual_percentile import PositivePercentileScale
        maps = PositivePercentileScale(pscale['scales'], name='visual_percentile')(maps)
        print("[visual_pipeline] applied PositivePercentileScale")
        
    if cap.get('input_reduce'):
        maps = TimeDistributed(Conv2D(cap['input_reduce'], 1, name='input_reduce_conv'),
                            name='input_reduce')(maps)

    dec_in = Input(
        shape=(intent._decoder_seq_length, width),
        name='decoder_input')
    bbox = SliceColumns(0, 4, name='bbox_slice')(dec_in)

    if target in ('encoder', 'both') and extra_width > 0:
        extra = SliceColumns(4, None, name='extra_slice_enc')(dec_in)
        maps = BroadcastExtraChannels(name='tile_extra')([maps, extra])

    rs = bool(cap.get("return_sequences", False))
    conv = ConvLSTM2D(
        filters=intent._convlstm_num_filters,
        kernel_size=intent._convlstm_kernel_size,
        kernel_regularizer=intent._kernel_regularizer,
        recurrent_regularizer=intent._recurrent_regularizer,
        bias_regularizer=intent._bias_regularizer,
        dropout=intent._lstm_dropout,
        recurrent_dropout=intent._lstm_recurrent_dropout,
        return_sequences=rs,
        name='conv_lstm2d')(maps)
    if cap.get('encoder_reduce'):
        if rs:
            conv = TimeDistributed(
                Conv2D(cap['encoder_reduce'], 1, name='encoder_reduce_conv'),
                name='encoder_reduce')(conv)
        else:
            conv = Conv2D(cap['encoder_reduce'], 1, name='encoder_reduce')(conv)
    if rs:
        visual = TimeDistributed(Flatten(name='encoder_flatten'), name='td_flatten')(conv)
        # visual shape: (B, T, flat)
    else:
        visual = Flatten(name='encoder_flatten')(conv)

    lstm_kw = dict(
        units=units,
        dropout=intent._lstm_dropout,
        recurrent_dropout=intent._lstm_recurrent_dropout,
        kernel_regularizer=intent._kernel_regularizer,
        recurrent_regularizer=intent._recurrent_regularizer,
        bias_regularizer=intent._bias_regularizer,
        activation=intent._activation,
        recurrent_activation=intent._lstm_recurrent_activation)

    if fusion == 'pie':
        if rs:
            # visual is already (B, T, flat); no RepeatVector needed
            side = bbox if target == 'encoder' else dec_in
            fused = Concatenate(axis=2, name='concatenate')([visual, side])
        else:
            repeated = RepeatVector(intent._decoder_seq_length)(visual)
            side = bbox if target == 'encoder' else dec_in
            fused = Concatenate(axis=2, name='concatenate')([repeated, side])
        hidden = LSTM(name='decoder_network', **lstm_kw)(fused)

    elif fusion == 'pcpa':
        vis_h = Dense(units, activation='tanh', name='vis_proj',
                      kernel_regularizer=intent._kernel_regularizer)(visual)
        bbox_h = LSTM(name='bbox_branch', **lstm_kw)(bbox)
        branches = [vis_h, bbox_h]
        if extra_width > 0 and target != 'encoder':
            extra = SliceColumns(4, None, name='extra_slice_branch')(dec_in)
            branches.append(LSTM(name='extra_branch', **lstm_kw)(extra))
        hidden = BranchAttention(name='branch_attention')(branches)

    elif fusion in ('branchattn', 'branchattn_temporal'):
        spans = getattr(intent, 'feature_spans', None)
        assert spans and spans[0][0] == 'bbox', f"feature_spans missing or bbox not first: {spans}"
        temporal = fusion == 'branchattn_temporal'
        mode = getattr(intent, 'branch_mode', None) or 'static_pool'
        print(f"[config] branch_mode={mode} temporal_attention={temporal}")

        if mode == 'static_pool':
            vis_h = Dense(units, activation='tanh', name='vis_proj',
                          kernel_regularizer=intent._kernel_regularizer)(visual)
            branches = [vis_h]
            for feat_name, start, end in spans:
                part = SliceColumns(start, end, name=f'slice_{feat_name}')(dec_in)
                out = LSTM(name=f'branch_{feat_name}', return_sequences=temporal, **lstm_kw)(part)
                if temporal:
                    out = AdditiveTemporalAttention(name=f'tattn_{feat_name}')(out)
                branches.append(out)
            hidden, attn_weights = BranchAttnPool(
                len(branches), return_weights=True, name='branch_attention')(branches)
            intent.branch_names = ['visual'] + [s[0] for s in spans]

        elif mode == 'temporal_concat':
            if temporal:
                print('[warning] temporal_concat uses one LSTM; additive temporal attention is not applied')
            if extra_width > 0:
                print('[warning] temporal_concat keeps extra columns inside the one LSTM, not as branches')
            repeated = RepeatVector(intent._decoder_seq_length)(visual)
            hidden = LSTM(name='decoder_network', **lstm_kw)(
                Concatenate(axis=2, name='concatenate')([repeated, dec_in]))
            intent.branch_names = ['pie_concat']

        elif mode == 'core_plus_extras':
            repeated = RepeatVector(intent._decoder_seq_length)(visual)
            core = LSTM(name='core_decoder', **lstm_kw)(
                Concatenate(axis=2, name='core_concat')([repeated, bbox]))
            branches = [core]
            names = ['core_bbox_visual']
            for feat_name, start, end in spans:
                if feat_name == 'bbox':
                    continue
                part = SliceColumns(start, end, name=f'slice_{feat_name}')(dec_in)
                out = LSTM(name=f'branch_{feat_name}', return_sequences=temporal, **lstm_kw)(part)
                if temporal:
                    out = AdditiveTemporalAttention(name=f'tattn_{feat_name}')(out)
                branches.append(out)
                names.append(feat_name)
            if len(branches) == 1:
                hidden = core
            else:
                hidden, attn_weights = BranchAttnPool(
                    len(branches), return_weights=True, name='branch_attention')(branches)
            intent.branch_names = names

        else:
            raise ValueError(f'unknown branch_mode: {mode!r}')

    else:
        raise ValueError(f"unknown fusion: {fusion!r}")

    output = Dense(
        intent._decoder_dense_output_size, activation='sigmoid',
        kernel_regularizer=intent._kernel_regularizer,
        bias_regularizer=intent._bias_regularizer,
        name='decoder_dense')(hidden)
    model = Model(inputs=[maps_in, dec_in], outputs=output, name='model')
    if attn_weights is not None:
        intent.attention_model = Model(inputs=[maps_in, dec_in], outputs=attn_weights)
    return model