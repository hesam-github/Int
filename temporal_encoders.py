"""
temporal_encoders.py

Temporal encoders for the PIE intention model.

  convlstm2d   Exact PIE path: ConvLSTM2D (padding=same) -> Flatten.
               Context width is 7*7*filters. Comparable with old runs.
  convlstm     Matched control: ConvLSTM2D -> pool -> project to d_model.
  transformer  Pre-norm Transformer over the 15 frames.
  s4           S4, sequential scan (exact at T=15).
  s4d          Diagonal S4 (S4D-Lin).
  mamba        Selective SSM + short causal conv.
  mamba3       Complex state, trapezoidal step, MIMO rank. No conv.

Non-PIE encoders collapse each 7x7x512 frame with SpatialStem first.
The decoder is always RepeatVector + concat(decoder_input) + LSTM + sigmoid.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate, ConvLSTM2D, Conv1D, Conv2D, Dense, Dropout,
    Flatten, GlobalAveragePooling2D, Input, LSTM, LayerNormalization,
    MultiHeadAttention, RepeatVector, TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def _l2(value):
    return l2(value) if value else None


class SpatialStem(tf.keras.layers.Layer):
    """Each VGG frame (7, 7, 512) -> one d_model vector."""

    def __init__(self, d_model, l2_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.l2_value = float(l2_value)
        self.proj = Conv2D(self.d_model, 1, activation='relu',
                           kernel_regularizer=_l2(l2_value))
        self.pool = GlobalAveragePooling2D()

    def call(self, frames):
        shape = tf.shape(frames)
        flat = tf.reshape(frames, (shape[0] * shape[1], shape[2], shape[3], shape[4]))
        tokens = self.pool(self.proj(flat))
        return tf.reshape(tokens, (shape[0], shape[1], self.d_model))
    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model, l2_value=self.l2_value)
        return config

class TemporalTransformer(tf.keras.layers.Layer):
    """Pre-norm Transformer. Default mask is causal; last frame is the summary."""

    def __init__(self, d_model, num_heads=4, num_layers=2, ff_mult=4,
                 dropout=0.1, l2_value=0.0, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_mult = int(ff_mult)
        self.dropout_rate = float(dropout)
        self.causal = bool(causal)
        self.l2_value = float(l2_value)
        self.pos = None
        self.blocks = []
        key_dim = self.d_model // self.num_heads
        for _ in range(self.num_layers):
            self.blocks.append((
                LayerNormalization(epsilon=1e-5),
                MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim,
                                   dropout=self.dropout_rate,
                                   kernel_regularizer=_l2(l2_value)),
                Dropout(self.dropout_rate),
                LayerNormalization(epsilon=1e-5),
                Dense(self.ff_mult * self.d_model, activation='gelu',
                      kernel_regularizer=_l2(l2_value)),
                Dense(self.d_model, kernel_regularizer=_l2(l2_value)),
                Dropout(self.dropout_rate),
            ))

    def build(self, input_shape):
        time = int(input_shape[1])
        self.pos = self.add_weight(
            name='pos', shape=(1, time, self.d_model),
            initializer='zeros', trainable=True)
        if self.causal:
            mask = np.tril(np.ones((time, time), dtype=np.float32))
            self._mask = tf.constant(mask[None, :, :])
        else:
            self._mask = None
        super().build(input_shape)

    def call(self, tokens, training=None):
        x = tokens + self.pos
        for norm1, attn, drop1, norm2, ff1, ff2, drop2 in self.blocks:
            h = norm1(x)
            h = attn(h, h, attention_mask=self._mask, training=training)
            x = x + drop1(h, training=training)
            h = ff2(ff1(norm2(x)))
            x = x + drop2(h, training=training)
        return x[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout_rate,
            l2_value=self.l2_value,
            causal=self.causal)
        return config


def _hippo_legs(n_state):
    index = np.arange(n_state, dtype=np.float32)
    basis = np.sqrt(2.0 * index + 1.0)
    low = np.tril(np.outer(basis, basis), k=-1)
    return (-low + np.diag(-(index + 1.0))).astype(np.float32)


class S4Layer(tf.keras.layers.Layer):
    """S4 as a bilinear discretization of HiPPO-LegS, scanned one frame at a time."""

    def __init__(self, d_model, n_state=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.n_state = int(n_state)

    def build(self, input_shape):
        n = self.n_state
        self.A = tf.constant(_hippo_legs(n))
        self.delta_diag = self.add_weight(
            name='delta_diag', shape=(n,), initializer='zeros', trainable=True)
        self.log_dt = self.add_weight(
            name='log_dt', shape=(self.d_model,),
            initializer=tf.keras.initializers.RandomUniform(np.log(1e-3), np.log(1e-1)))
        b0 = np.sqrt(2.0 * np.arange(n) + 1.0).astype(np.float32)
        b0 = np.broadcast_to(b0, (self.d_model, n)).copy()
        self.B = self.add_weight(
            name='B', shape=(self.d_model, n),
            initializer=tf.keras.initializers.Constant(b0))
        self.C = self.add_weight(
            name='C', shape=(self.d_model, n),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.out = Dense(self.d_model)
        super().build(input_shape)

    def call(self, tokens):
        dt = tf.cast(tf.exp(self.log_dt), tf.complex64)
        real = -tf.exp(self.log_A_real)
        imag = tf.broadcast_to(self.A_imag[None, :], tf.shape(real))
        A = tf.complex(real, imag)
        B = tf.complex(self.B_real, self.B_imag)
        C = tf.complex(self.C_real, self.C_imag)
        a_bar = (1.0 + dt * A / 2.0) / (1.0 - dt * A / 2.0)
        b_bar = dt * B / (1.0 - dt * A / 2.0)
        series = tf.transpose(tokens, [1, 0, 2])

        def step(h_prev, x_t):
            x_c = tf.cast(x_t, tf.complex64)[:, :, None]
            return a_bar[None, :, :] * h_prev + b_bar[None, :, :] * x_c

        h0 = tf.zeros((tf.shape(tokens)[0], self.d_model, self.n_complex), dtype=tf.complex64)
        states = tf.scan(step, series, initializer=h0)
        y = 2.0 * tf.math.real(tf.reduce_sum(tf.math.conj(C)[None, None, :, :] * states, axis=-1))
        return self.out(tf.transpose(y, [1, 0, 2]))[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model, n_complex=self.n_complex)
        return config


class S4DLayer(tf.keras.layers.Layer):
    """S4D-Lin. Diagonal complex A. No low-rank HiPPO term."""

    def __init__(self, d_model, n_complex=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.n_complex = int(n_complex)

    def build(self, input_shape):
        n = self.n_complex
        imag0 = (np.pi * np.arange(n)).astype(np.float32)
        self.log_A_real = self.add_weight(
            name='log_A_real', shape=(self.d_model, n),
            initializer=tf.keras.initializers.Constant(np.log(0.5).astype(np.float32)))
        self.A_imag = self.add_weight(
            name='A_imag', shape=(n,),
            initializer=tf.keras.initializers.Constant(imag0), trainable=False)
        self.log_dt = self.add_weight(
            name='log_dt', shape=(self.d_model, 1),
            initializer=tf.keras.initializers.RandomUniform(np.log(1e-3), np.log(1e-1)))
        self.B_real = self.add_weight(name='B_real', shape=(self.d_model, n), initializer='ones')
        self.B_imag = self.add_weight(name='B_imag', shape=(self.d_model, n), initializer='zeros')
        self.C_real = self.add_weight(
            name='C_real', shape=(self.d_model, n),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.C_imag = self.add_weight(
            name='C_imag', shape=(self.d_model, n),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.out = Dense(self.d_model)
        super().build(input_shape)

    def call(self, tokens):
        dt = tf.cast(tf.exp(self.log_dt), tf.complex64)
        real = -tf.exp(self.log_A_real)
        imag = tf.broadcast_to(self.A_imag[None, :], tf.shape(real))
        A = tf.complex(real, imag)
        B = tf.complex(self.B_real, self.B_imag)
        C = tf.complex(self.C_real, self.C_imag)
        a_bar = (1.0 + dt * A / 2.0) / (1.0 - dt * A / 2.0)
        b_bar = dt * B / (1.0 - dt * A / 2.0)
        series = tf.transpose(tokens, [1, 0, 2])

    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model, n_complex=self.n_complex)
        return config



        def step(h_prev, x_t):
            x_c = tf.cast(x_t, tf.complex64)[:, :, None]
            return a_bar[None, :, :] * h_prev + b_bar[None, :, :] * x_c

        h0 = tf.zeros((tf.shape(tokens)[0], self.d_model, self.n_complex), dtype=tf.complex64)
        states = tf.scan(step, series, initializer=h0)
        y = 2.0 * tf.math.real(tf.reduce_sum(tf.math.conj(C)[None, None, :, :] * states, axis=-1))
        return self.out(tf.transpose(y, [1, 0, 2]))[:, -1, :]
    def get_config(self):
        config = super().get_config()
        config.update(d_model=self.d_model, n_complex=self.n_complex)
        return config


def _inv_softplus(values):
    return np.log(np.exp(np.asarray(values, dtype=np.float32)) - 1.0).astype(np.float32)


class MambaBlock(tf.keras.layers.Layer):
    """One Mamba block. h_t = exp(dt A) h_{t-1} + dt B x_t."""

    def __init__(self, d_model, n_state=16, expand=2, conv_size=4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.n_state = int(n_state)
        self.expand = int(expand)
        self.conv_size = int(conv_size)
        self.d_inner = self.expand * self.d_model
        self.in_proj = Dense(2 * self.d_inner, use_bias=False)
        self.conv = Conv1D(self.d_inner, self.conv_size, groups=self.d_inner,
                           padding='causal', use_bias=True)
        self.x_proj = Dense(self.d_inner + 2 * self.n_state, use_bias=False)
        dt0 = np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1), size=(self.d_inner,)))
        self.dt_proj = Dense(
            self.d_inner, use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(_inv_softplus(dt0)))
        self.out_proj = Dense(self.d_model, use_bias=False)
        self.norm = LayerNormalization(epsilon=1e-5)

    def build(self, input_shape):
        log_a = np.log(np.arange(1, self.n_state + 1, dtype=np.float32))
        log_a = np.broadcast_to(log_a, (self.d_inner, self.n_state)).copy()
        self.log_A = self.add_weight(
            name='log_A', shape=(self.d_inner, self.n_state),
            initializer=tf.keras.initializers.Constant(log_a))
        super().build(input_shape)

    def call(self, tokens):
        residual = tokens
        x = self.norm(tokens)
        x, z = tf.split(self.in_proj(x), 2, axis=-1)
        x = tf.nn.silu(self.conv(x))
        projected = self.x_proj(x)
        dt_in, B, C = tf.split(projected, [self.d_inner, self.n_state, self.n_state], axis=-1)
        dt = tf.nn.softplus(self.dt_proj(dt_in))
        A = -tf.exp(self.log_A)
        series_x = tf.transpose(x, [1, 0, 2])
        series_dt = tf.transpose(dt, [1, 0, 2])
        series_B = tf.transpose(B, [1, 0, 2])
        series_C = tf.transpose(C, [1, 0, 2])

        def step(state, elems):
            h_prev, _ = state
            x_t, dt_t, b_t, c_t = elems
            decay = tf.exp(dt_t[:, :, None] * A[None, :, :])
            h = decay * h_prev + dt_t[:, :, None] * b_t[:, None, :] * x_t[:, :, None]
            y = tf.reduce_sum(h * c_t[:, None, :], axis=-1)
            return h, y

        h0 = tf.zeros((tf.shape(x)[0], self.d_inner, self.n_state))
        y0 = tf.zeros((tf.shape(x)[0], self.d_inner))
        _, ys = tf.scan(step, (series_x, series_dt, series_B, series_C), initializer=(h0, y0))
        y = tf.transpose(ys, [1, 0, 2]) * tf.nn.silu(z)
        return residual + self.out_proj(y)
    def get_config(self):
        config = super().get_config()
        config.update(
            d_model=self.d_model,
            n_state=self.n_state,
            expand=self.expand,
            conv_size=self.conv_size)
        return config


def _rms_norm(values, eps=1e-5):
    power = tf.reduce_mean(tf.square(values), axis=-1, keepdims=True)
    return values * tf.math.rsqrt(power + eps)


class Mamba3Block(tf.keras.layers.Layer):
    """Complex selective SSM: no conv, trapezoidal step, MIMO rank."""

    def __init__(self, d_model, n_heads=4, n_state=16, mimo_rank=4, **kwargs):
        super().__init__(**kwargs)
        if int(d_model) % int(n_heads) != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_state = int(n_state)
        self.mimo_rank = int(mimo_rank)
        self.norm = LayerNormalization(epsilon=1e-5)
        packed = self.n_state * self.mimo_rank * 2 + self.mimo_rank + 1
        self.to_ssm = Dense(self.n_heads * packed, use_bias=True)
        self.out_proj = Dense(self.d_model, use_bias=False)

    def build(self, input_shape):
        self.log_A_real = self.add_weight(
            name='log_A_real', shape=(self.n_heads, self.n_state),
            initializer=tf.keras.initializers.Constant(np.log(0.5).astype(np.float32)))
        imag0 = (np.pi * np.arange(self.n_state) / self.n_state).astype(np.float32)
        imag0 = np.broadcast_to(imag0, (self.n_heads, self.n_state)).copy()
        self.A_imag = self.add_weight(
            name='A_imag', shape=(self.n_heads, self.n_state),
            initializer=tf.keras.initializers.Constant(imag0))
        self.b_bias = self.add_weight(
            name='b_bias', shape=(self.n_heads, self.n_state, self.mimo_rank), initializer='zeros')
        self.c_bias = self.add_weight(
            name='c_bias', shape=(self.n_heads, self.mimo_rank, self.n_state), initializer='zeros')
        super().build(input_shape)

    def call(self, tokens):
        residual = tokens
        x = self.norm(tokens)
        batch = tf.shape(x)[0]
        time = tf.shape(x)[1]
        packed = tf.reshape(
            self.to_ssm(x),
            (batch, time, self.n_heads,
             self.n_state * self.mimo_rank * 2 + self.mimo_rank + 1))
        b_raw, c_raw, content, dt_raw = tf.split(
            packed,
            [self.n_state * self.mimo_rank, self.n_state * self.mimo_rank, self.mimo_rank, 1],
            axis=-1)
        B = tf.reshape(b_raw, (batch, time, self.n_heads, self.n_state, self.mimo_rank))
        C = tf.reshape(c_raw, (batch, time, self.n_heads, self.mimo_rank, self.n_state))
        B = _rms_norm(B + self.b_bias[None, None, :, :, :])
        C = _rms_norm(C + self.c_bias[None, None, :, :, :])
        dt = tf.nn.softplus(dt_raw)
        A = tf.complex(-tf.exp(self.log_A_real), self.A_imag)
        decay = tf.exp(tf.cast(dt, tf.complex64) * A[None, None, :, :])
        drive = tf.einsum('btlnr,btlr->btln', B, content)
        drive = tf.cast(0.5 * dt, tf.complex64) * tf.cast(drive, tf.complex64)
        series_decay = tf.transpose(decay, [1, 0, 2, 3])
        series_drive = tf.transpose(drive, [1, 0, 2, 3])

        def step(state, elems):
            h_prev, u_prev = state
            decay_t, u_t = elems
            return decay_t * h_prev + u_t + decay_t * u_prev, u_t

        zeros = tf.zeros((batch, self.n_heads, self.n_state), dtype=tf.complex64)
        states, _ = tf.scan(step, (series_decay, series_drive), initializer=(zeros, zeros))
        h_seq = tf.transpose(states, [1, 0, 2, 3])
        read = tf.einsum('btlrn,btln->btlr', tf.cast(C, tf.float32), tf.math.real(h_seq))
        # read shape: (B, T, n_heads, mimo_rank) → flatten heads*rank, then project to d_model
        read_flat = tf.reshape(read, (batch, time, self.n_heads * self.mimo_rank))
        y = self.out_proj(read_flat)
        return residual + y

    def get_config(self):
        config = super().get_config()
        config.update(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_state=self.n_state,
            mimo_rank=self.mimo_rank)
        return config


class LastStep(tf.keras.layers.Layer):
    def call(self, sequence):
        return sequence[:, -1, :]


class MatchedConvLSTM(tf.keras.layers.Layer):
    """ConvLSTM on the grid, then pool each frame and project to d_model."""

    def __init__(self, d_model, filters=64, kernel=2, dropout=0.4,
                 recurrent_dropout=0.2, l2_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.dropout_rate = float(dropout)
        self.recurrent_dropout_rate = float(recurrent_dropout)
        self.l2_value = float(l2_value)
        self.convlstm = ConvLSTM2D(
            filters=self.filters, kernel_size=self.kernel, padding='same',
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            kernel_regularizer=_l2(self.l2_value),
            recurrent_regularizer=_l2(self.l2_value),
            bias_regularizer=_l2(self.l2_value))
        self.pool = GlobalAveragePooling2D()
        self.proj = Dense(self.d_model, kernel_regularizer=_l2(self.l2_value))
        self.last = LastStep()

    def call(self, frames, training=None):
        grid = self.convlstm(frames, training=training)
        shape = tf.shape(grid)
        pooled = self.pool(tf.reshape(grid, (shape[0] * shape[1], shape[2], shape[3], shape[4])))
        tokens = self.proj(tf.reshape(pooled, (shape[0], shape[1], self.filters)))
        return self.last(tokens)

    def get_config(self):
        config = super().get_config()
        config.update(
            d_model=self.d_model,
            filters=self.filters,
            kernel=self.kernel,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            l2_value=self.l2_value)
        return config

FACTORIZED_PREFIX = 'convfact_'


def _make_temporal_block(name, d_model, num_heads, num_layers, ff_mult, n_state,
                         dropout, recurrent_dropout, recurrent_activation, l2_value):
    """One temporal block: (N, T, d_model) -> (N, d_model), last step only.
    `name` is the base name, without the 'convfact_' prefix."""
    if name == 'lstm':
        return LSTM(
            d_model, recurrent_dropout=recurrent_dropout,
            activation='tanh', recurrent_activation=recurrent_activation,
            kernel_regularizer=_l2(l2_value), recurrent_regularizer=_l2(l2_value),
            bias_regularizer=_l2(l2_value), name='pp_lstm')
    if name == 'transformer':
        return TemporalTransformer(
            d_model, num_heads=num_heads, num_layers=num_layers, ff_mult=ff_mult,
            dropout=min(float(dropout), 0.1), l2_value=l2_value, causal=True,
            name='pp_transformer')
    if name == 's4':
        return S4Layer(d_model, n_state=n_state, name='pp_s4')
    if name == 's4d':
        return S4DLayer(d_model, n_complex=n_state, name='pp_s4d')
    if name == 'mamba':
        blocks = [MambaBlock(d_model, n_state=n_state) for _ in range(int(num_layers))]
        return tf.keras.Sequential(blocks + [LastStep()], name='pp_mamba')
    raise ValueError(f'no per-position block for {name!r}')


class PerPositionTemporal(tf.keras.layers.Layer):
    """(B, T, H, W, C) -> (B, H, W, C).

    One temporal block with SHARED weights, applied independently at each of the
    H*W grid positions: every position becomes one length-T sequence in a batch of
    B*H*W sequences. C must equal the block width (d_model).
    Not a convolutional state-space model: space is mixed only by the Conv2D that
    runs BEFORE this layer, never inside the recurrence."""

    def __init__(self, encoder_name, d_model, num_heads=4, num_layers=1, ff_mult=4,
                 n_state=16, dropout=0.4, recurrent_dropout=0.2,
                 recurrent_activation='hard_sigmoid', l2_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder_name = str(encoder_name)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_mult = int(ff_mult)
        self.n_state = int(n_state)
        self.dropout_rate = float(dropout)
        self.recurrent_dropout_rate = float(recurrent_dropout)
        self.recurrent_activation_name = str(recurrent_activation)
        self.l2_value = float(l2_value)
        self.block = _make_temporal_block(
            self.encoder_name, self.d_model, self.num_heads, self.num_layers,
            self.ff_mult, self.n_state, self.dropout_rate, self.recurrent_dropout_rate,
            self.recurrent_activation_name, self.l2_value)

    def call(self, frames, training=None):
        time, height, width, channels = frames.shape[1:]      # static dims
        if channels != self.d_model:
            raise ValueError(
                f'PerPositionTemporal: input channels {channels} != d_model {self.d_model}')
        x = tf.transpose(frames, [0, 2, 3, 1, 4])             # (B, H, W, T, C)
        x = tf.reshape(x, (-1, time, channels))               # (B*H*W, T, C)
        y = self.block(x, training=training)                  # (B*H*W, C)
        return tf.reshape(y, (-1, height, width, channels))   # (B, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            encoder_name=self.encoder_name, d_model=self.d_model,
            num_heads=self.num_heads, num_layers=self.num_layers,
            ff_mult=self.ff_mult, n_state=self.n_state,
            dropout=self.dropout_rate, recurrent_dropout=self.recurrent_dropout_rate,
            recurrent_activation=self.recurrent_activation_name, l2_value=self.l2_value)
        return config

def _apply_shared_visual_front(maps, visual_norm, capacity, percentile_scale):
    """Same channel ops as feature_routing, so option 8 honours the visual menu.
    Does not change padding and does not implement return_sequences."""
    from visual_norm import ChannelStandardize, visual_op_order
    from feature_routing import Log1p
    cap = capacity or {}
    vn = visual_norm or None
    ops = visual_op_order(vn, cap)
    print(f"[temporal visual_pipeline] VGG -> {' -> '.join(ops) if ops else '(raw)'}")
    for op in ops:
        if op == 'standardize':
            maps = ChannelStandardize(vn['mean'], vn['std'], name='visual_standardize')(maps)
        else:
            maps = Log1p(name='log1p_input')(maps)
    if percentile_scale:
        from visual_percentile import PositivePercentileScale
        maps = PositivePercentileScale(percentile_scale['scales'], name='visual_percentile')(maps)
        print("[temporal visual_pipeline] applied PositivePercentileScale")
    if cap.get('input_reduce'):
        maps = TimeDistributed(
            Conv2D(int(cap['input_reduce']), 1, name='input_reduce_conv'),
            name='input_reduce')(maps)
        print(f"[temporal] input_reduce 1x1 -> {int(cap['input_reduce'])}")
    return maps

def build_pie_temporal_model(encoder_name,
                             seq_len=15,
                             vgg_shape=(7, 7, 512),
                             decoder_dim=4,
                             d_model=128,
                             convlstm_filters=64,
                             convlstm_kernel=2,
                             n_state=64,
                             num_heads=4,
                             num_layers=2,
                             ff_mult=4,
                             mimo_rank=4,
                             lstm_units=128,
                             dropout=0.4,
                             recurrent_dropout=0.2,
                             recurrent_activation='hard_sigmoid',
                             l2_value=0.001,
                             fusion_mode='pie_concat',
                             feature_spans=None,
                             visual_norm=None,
                             capacity=None,
                             percentile_scale=None):
    """encoder_name selects only the visual temporal block. Decoder is shared.

    'convfact_*' (strategy 'factorized'): locked input dropout -> per-frame Conv2D
    (padding=same, keeps the 7x7 grid) -> ONE temporal block applied at every grid
    position with shared weights -> Flatten. Output width is 7*7*convlstm_filters,
    identical to 'convlstm2d', so only the temporal block differs."""
    if encoder_name not in ALL_ENCODER_INFO:
        raise ValueError(encoder_name)

    encoder_in = Input(shape=(seq_len,) + tuple(vgg_shape), name='encoder_input')
    decoder_in = Input(shape=(seq_len, decoder_dim), name='decoder_input')
    cap = capacity or {}
    if cap.get('return_sequences'):
        print("[temporal] return_sequences=True ignored. This decoder needs one context vector.")
    maps = _apply_shared_visual_front(encoder_in, visual_norm, cap, percentile_scale)
    in_ch = int(cap['input_reduce']) if cap.get('input_reduce') else int(vgg_shape[-1])
    spatial = (int(vgg_shape[0]), int(vgg_shape[1]), in_ch)
    print("[temporal] padding stays SAME for convlstm2d and convfact_*. "
          "feature_routing pie stays VALID. Do not compare their AUC as the same ConvLSTM.")

    if encoder_name in FACTORIZED_ENCODER_INFO:
        grid = Dropout(dropout, noise_shape=(None, 1) + spatial,
                       name='locked_input_dropout')(maps)
        grid = TimeDistributed(
            Conv2D(convlstm_filters, convlstm_kernel, padding='same', activation='relu',
                   kernel_regularizer=_l2(l2_value)),
            name='spatial_conv')(grid)
        grid = PerPositionTemporal(
            encoder_name[len(FACTORIZED_PREFIX):], d_model=convlstm_filters,
            num_heads=num_heads, num_layers=num_layers, ff_mult=ff_mult,
            n_state=n_state, dropout=dropout, recurrent_dropout=recurrent_dropout,
            recurrent_activation=recurrent_activation, l2_value=l2_value,
            name='per_position_temporal')(grid)
        if cap.get('encoder_reduce'):
            grid = Conv2D(int(cap['encoder_reduce']), 1, name='encoder_reduce')(grid)
            print(f"[temporal] encoder_reduce 1x1 -> {int(cap['encoder_reduce'])}")
        context = Flatten(name='encoder_flatten')(grid)
    elif encoder_name == 'convlstm2d':
        conv = ConvLSTM2D(
            filters=convlstm_filters, kernel_size=convlstm_kernel, padding='same',
            return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout,
            kernel_regularizer=_l2(l2_value), recurrent_regularizer=_l2(l2_value),
            bias_regularizer=_l2(l2_value), name='conv_lstm2d')(maps)
        if cap.get('encoder_reduce'):
            conv = Conv2D(int(cap['encoder_reduce']), 1, name='encoder_reduce')(conv)
            print(f"[temporal] encoder_reduce 1x1 -> {int(cap['encoder_reduce'])}")
        context = Flatten(name='encoder_flatten')(conv)
    elif encoder_name == 'convlstm':
        if cap.get('encoder_reduce'):
            print("[temporal] encoder_reduce skipped: matched convlstm pools each frame.")
        context = MatchedConvLSTM(
            d_model, filters=convlstm_filters, kernel=convlstm_kernel,
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            l2_value=l2_value)(maps)
    else:
        if cap.get('encoder_reduce'):
            print("[temporal] encoder_reduce skipped: SpatialStem already collapses 7x7.")
        tokens = SpatialStem(d_model, l2_value=l2_value)(maps)
        if encoder_name == 'transformer':
            context = TemporalTransformer(
                d_model, num_heads=num_heads, num_layers=num_layers, ff_mult=ff_mult,
                dropout=min(float(dropout), 0.1), l2_value=l2_value, causal=True)(tokens)
        elif encoder_name == 's4':
            context = S4Layer(d_model, n_state=n_state)(tokens)
        elif encoder_name == 's4d':
            context = S4DLayer(d_model, n_complex=n_state)(tokens)
        elif encoder_name == 'mamba':
            sequence = tokens
            for _ in range(int(num_layers)):
                sequence = MambaBlock(d_model, n_state=n_state)(sequence)
            context = LastStep()(sequence)
        else:
            sequence = tokens
            for _ in range(int(num_layers)):
                sequence = Mamba3Block(
                    d_model, n_heads=num_heads, n_state=n_state, mimo_rank=mimo_rank)(sequence)
            context = LastStep()(sequence)

    lstm_kw = dict(
        units=lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
        activation='tanh', recurrent_activation=recurrent_activation,
        kernel_regularizer=_l2(l2_value), recurrent_regularizer=_l2(l2_value),
        bias_regularizer=_l2(l2_value))
    mode = fusion_mode or 'pie_concat'
    if mode == 'pie_concat':
        decoded = LSTM(name='decoder_network', **lstm_kw)(
            Concatenate(axis=-1, name='concatenate')([repeated, decoder_in]))
    elif mode == 'core_plus_extras':
        from feature_routing import SliceColumns, BranchAttnPool
        bbox = SliceColumns(0, 4, name='bbox_slice')(decoder_in)
        core = LSTM(name='core_decoder', **lstm_kw)(
            Concatenate(axis=-1, name='core_concat')([repeated, bbox]))
        extra_spans = [tuple(s) for s in (feature_spans or []) if tuple(s)[0] != 'bbox']
        if not extra_spans and int(decoder_dim) > 4:
            extra_spans = [('extra', 4, int(decoder_dim))]
        if not extra_spans:
            decoded = core
        else:
            branches = [core]
            for feat_name, start, end in extra_spans:
                part = SliceColumns(int(start), int(end), name=f'slice_{feat_name}')(decoder_in)
                branches.append(LSTM(name=f'branch_{feat_name}', **lstm_kw)(part))
            decoded, _ = BranchAttnPool(
                len(branches), return_weights=True, name='branch_attention')(branches)
    elif mode == 'adaptive_branches':
        from feature_routing import SliceColumns, BranchAttnPool
        visual_vec = context  # (B, d_model)

        kin_names = {'bbox', 'velocity', 'flow_bbox', 'flow_crop', 'flow_ped'}
        print("[temporal] adaptive kin = bbox, velocity, flow_bbox, flow_crop, flow_ped. "
              "pose, scene, scene_critical, flow_rel go to the geometric branch. "
              "This split is hardcoded, not a tested scientific grouping.")
        kin_parts, geo_parts = [], []
        for name, start, end in (feature_spans or []):
            part = SliceColumns(int(start), int(end), name=f'slice_{name}')(decoder_in)
            if name in kin_names:
                kin_parts.append(part)
            else:
                geo_parts.append(part)

        # Force every branch to the same dimension as the visual vector
        target_dim = visual_vec.shape[-1]
        lstm_kwargs = {k: v for k, v in lstm_kw.items() if k != 'units'}

        branches = [visual_vec]
        if kin_parts:
            kin_in = (Concatenate(axis=-1, name='kin_concat')(kin_parts)
                      if len(kin_parts) > 1 else kin_parts[0])
            kin_raw = LSTM(lstm_units, name='kin_lstm', **lstm_kwargs)(kin_in)
            kin_vec = Dense(target_dim, name='kin_proj')(kin_raw)
            branches.append(kin_vec)
        if geo_parts:
            geo_in = (Concatenate(axis=-1, name='geo_concat')(geo_parts)
                      if len(geo_parts) > 1 else geo_parts[0])
            geo_raw = LSTM(lstm_units, name='geo_lstm', **lstm_kwargs)(geo_in)
            geo_vec = Dense(target_dim, name='geo_proj')(geo_raw)
            branches.append(geo_vec)

        if len(branches) == 1:
            decoded = visual_vec
        else:
            decoded, _ = BranchAttnPool(
                len(branches), return_weights=True, name='adaptive_branch_pool')(branches)
    else:
        raise ValueError(f'unknown temporal fusion_mode: {mode!r}')
    intention = Dense(
        1, activation='sigmoid', kernel_regularizer=_l2(l2_value),
        bias_regularizer=_l2(l2_value), name='decoder_dense')(decoded)
    return Model([encoder_in, decoder_in], intention, name='pie_' + encoder_name)


ENCODER_INFO = {
    'convlstm2d': {
        'desc': 'Exact PIE: ConvLSTM2D padding=same, then Flatten (7x7xfilters).',
        'paper': 'PIE, Rasouli et al. ICCV 2019: filters=64, kernel=2',
    },
    'convlstm': {
        'desc': 'Same ConvLSTM, then pool+project to d_model (fair-size control, not the paper).',
        'paper': 'PIE conv: filters=64, kernel=2; projection d_model=128',
    },
    'transformer': {
        'desc': 'Pre-norm Transformer. Last frame is the context vector.',
        'paper': 'Vaswani et al. 2017: d_model=512, heads=8, layers=6, ff=4*d, dropout=0.1. Short T often uses d_model=128, heads=4, layers=2',
    },
    's4': {
        'desc': 'S4 (HiPPO-LegS), exact scan. Fine at T=15.',
        'paper': 'Gu et al. ICLR 2022: state N=64. This menu default is 64',
    },
    's4d': {
        'desc': 'S4D-Lin, diagonal complex state.',
        'paper': 'Gu et al. S4D: state N=64, A_n=-0.5+i*pi*n',
    },
    'mamba': {
        'desc': 'Selective SSM + causal conv.',
        'paper': 'Gu and Dao 2023: d_state=16, expand=2, d_conv=4. Depth varies; default layers=2',
    },
    'mamba3': {
        'desc': 'Complex SSM, no conv, trapezoidal step, MIMO rank.',
        'paper': 'Defaults here: heads=4, d_state=16, mimo_rank=4, layers=2. d_model must divide by heads',
    },
}

# Mamba-3 is deliberately NOT offered in the factorized strategy, see the notes in the chat.
FACTORIZED_ENCODER_INFO = {
    'convfact_lstm': {
        'desc': 'CONTROL. Conv2D -> one shared LSTM per grid position. Same factorization as the '
                'other convfact_* options, LSTM as the temporal block.',
        'paper': 'LSTM: Hochreiter & Schmidhuber 1997. Run this FIRST and compare it with '
                 'convlstm2d (isolates the effect of factorizing) and with the other convfact_* '
                 '(isolates the effect of the temporal block).',
    },
    'convfact_transformer': {
        'desc': 'Causal pre-norm Transformer per grid position. Last token is the summary.',
        'paper': 'Vaswani et al. NeurIPS 2017 used d=512, 8 heads, 6 layers. For T=15 and d=64 '
                 '1-2 layers and 4 heads is the sensible range (author judgment, not from the paper).',
    },
    'convfact_s4': {
        'desc': 'S4 (HiPPO-LegS) per grid position, exact sequential scan.',
        'paper': 'Gu, Goel, Re. ICLR 2022 (state N=64 there). Default here is 16: memory grows '
                 'with n_state * batch * 49.',
    },
    'convfact_s4d': {
        'desc': 'S4D-Lin (diagonal complex state) per grid position.',
        'paper': 'Gu, Gupta, Goel, Re. NeurIPS 2022 (N=64 there). Default here is 16.',
    },
    'convfact_mamba': {
        'desc': 'Selective SSM + short causal conv per grid position.',
        'paper': 'Gu & Dao 2023: d_state=16, expand=2, d_conv=4. Default here is 1 layer to save memory.',
    },
}

ALL_ENCODER_INFO = {**ENCODER_INFO, **FACTORIZED_ENCODER_INFO}

STRATEGY_INFO = {
    'current': {
        'title': 'existing options (joint ConvLSTM2D, or pooled grid + temporal block)',
        'lines': [
            'convlstm2d : gates convolve the input AND the previous state, so space and time are',
            '             mixed at every step (Shi et al. NeurIPS 2015). PIE reference encoder',
            '             (Rasouli et al. ICCV 2019: 64 filters, kernel 2).',
            'others      : the 7x7 grid is collapsed by pooling BEFORE the temporal block, so they',
            '             differ from convlstm2d in TWO things at once (temporal block AND spatial',
            '             information).',
            'Use for: reproducing the paper and the baseline row. Not for isolating the encoder.',
        ],
    },
    'factorized': {
        'title': 'Conv2D keeps the grid, then ONE temporal block per grid position (shared weights)',
        'lines': [
            'frame (7,7,512) -> Conv2D(same) -> (7,7,C) -> temporal block over T at each of the 49',
            'positions -> (7,7,C) -> Flatten. Output width equals convlstm2d, so ONLY the temporal',
            'block changes between options.',
            'Precedent for factorizing space and time: (2+1)D convolutions (Tran et al. CVPR 2018),',
            'factorised encoders in ViViT (Arnab et al. ICCV 2021), divided attention in',
            'TimeSformer (Bertasius et al. ICML 2021).',
            'NOT a true convolutional SSM (ConvS5, Smith et al. NeurIPS 2023, has a convolutional',
            'state transition); do not name it that in the thesis.',
            'Recommended when the question is "which temporal encoder?": run convfact_lstm first,',
            'keep convlstm2d (strategy current) as the reference row.',
        ],
    },
}


def interactive_pick_temporal_strategy(default='current'):
    names = list(STRATEGY_INFO.keys())
    print('\n--- Temporal-encoder strategy ---')
    for i, name in enumerate(names, 1):
        info = STRATEGY_INFO[name]
        mark = ' *' if name == default else ''
        print(f'  {i}. {name}{mark}: {info["title"]}')
        for line in info['lines']:
            print(f'      {line}')
    raw = input(f'Select [1-{len(names)}] (default {default}): ').strip()
    if not raw:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(names):
            return names[idx]
    except ValueError:
        pass
    print('Invalid choice, using default')
    return default

def interactive_pick_temporal_encoder(default='convlstm2d'):
    names = list(ENCODER_INFO.keys())
    print('\n--- Temporal encoder ---')
    for i, name in enumerate(names, 1):
        info = ENCODER_INFO[name]
        mark = ' *' if name == default else ''
        print(f'  {i}. {name}{mark}')
        print(f'      {info["desc"]}')
        print(f'      paper: {info["paper"]}')
    raw = input(f'Select [1-{len(names)}] (default {default}): ').strip()
    if not raw:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(names):
            return names[idx]
    except ValueError:
        pass
    print('Invalid choice, using default')
    return default


def _ask_int(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    return int(raw) if raw else int(default)

def interactive_pick_temporal_encoder(strategy='current', default=None):
    source = FACTORIZED_ENCODER_INFO if strategy == 'factorized' else ENCODER_INFO
    names = list(source.keys())
    default = default if default in names else names[0]
    print(f'\n--- Temporal encoder ({strategy}) ---')
    for i, name in enumerate(names, 1):
        info = source[name]
        mark = ' *' if name == default else ''
        print(f'  {i}. {name}{mark}')
        print(f'      {info["desc"]}')
        print(f'      paper: {info["paper"]}')
    raw = input(f'Select [1-{len(names)}] (default {default}): ').strip()
    if not raw:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(names):
            return names[idx]
    except ValueError:
        pass
    print('Invalid choice, using default')
    return default


def _ask_int(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    return int(raw) if raw else int(default)


def interactive_pick_temporal_hyperparams(encoder_name):
    """Ask only the knobs that belong to this encoder. Enter keeps the default."""
    print(f'\n--- Hyperparameters for {encoder_name} ---')
    print(f'    paper: {ALL_ENCODER_INFO[encoder_name]["paper"]}')
    hp = {}

    if encoder_name in FACTORIZED_ENCODER_INFO:
        base = encoder_name[len(FACTORIZED_PREFIX):]
        hp['convlstm_filters'] = _ask_int('conv filters = temporal block width', 64)
        hp['convlstm_kernel'] = _ask_int('conv kernel (padding=same)', 2)
        if base == 'transformer':
            hp['num_heads'] = _ask_int('num_heads', 4)
            hp['num_layers'] = _ask_int('num_layers', 2)
            hp['ff_mult'] = _ask_int('ff_mult (paper uses 4)', 4)
            if hp['convlstm_filters'] % hp['num_heads'] != 0:
                raise ValueError('conv filters must be divisible by num_heads')
        elif base in ('s4', 's4d'):
            hp['n_state'] = _ask_int('n_state (paper 64; memory grows with it)', 16)
        elif base == 'mamba':
            hp['n_state'] = _ask_int('n_state / d_state (paper 16)', 16)
            hp['num_layers'] = _ask_int('num_layers (1 keeps memory low)', 1)
        print(f'[temporal] {encoder_name} {hp}')
        return hp

    if encoder_name in ('convlstm2d', 'convlstm'):
        hp['convlstm_filters'] = _ask_int('convlstm filters', 64)
        hp['convlstm_kernel'] = _ask_int('convlstm kernel', 2)
        if encoder_name == 'convlstm':
            hp['d_model'] = _ask_int('d_model (after pool)', 128)
    else:
        hp['d_model'] = _ask_int('d_model', 128)
        hp['num_layers'] = _ask_int('num_layers', 2)
        if encoder_name == 'transformer':
            hp['num_heads'] = _ask_int('num_heads', 4)
            hp['ff_mult'] = _ask_int('ff_mult (paper uses 4)', 4)
        if encoder_name in ('s4', 's4d'):
            hp['n_state'] = _ask_int('n_state (paper 64)', 64)
        if encoder_name == 'mamba':
            hp['n_state'] = _ask_int('n_state / d_state (paper 16)', 16)
        if encoder_name == 'mamba3':
            hp['num_heads'] = _ask_int('num_heads', 4)
            hp['n_state'] = _ask_int('n_state', 16)
            hp['mimo_rank'] = _ask_int('mimo_rank', 4)
            if hp['d_model'] % hp['num_heads'] != 0:
                raise ValueError('d_model must be divisible by num_heads')
    print(f'[temporal] {encoder_name} {hp}')
    return hp


def _print_factorized_cost(encoder_name, opts):
    """The temporal block sees batch_size * 49 sequences per step. Rough note, not measured."""
    base = encoder_name[len(FACTORIZED_PREFIX):]
    batch = opts.get('batch_size')
    if not batch:
        return
    print(f'[temporal] factorized: a batch of {batch} windows becomes {int(batch) * 49} '
          f'sequences (7*7 positions per window) for the temporal block.')
    if base in ('s4', 's4d', 'mamba') and int(batch) > 32:
        print(f'[temporal] WARNING: {base} is a tf.scan block; backprop keeps per-step states for '
              f'every one of these sequences. Not measured here. On a 4 GB GPU start with '
              f'batch_size <= 32 and raise it only if memory allows.')

def interactive_pick_temporal_fusion(default='pie_concat'):
    print('\n--- Temporal decoder fusion ---')
    print('  1. pie_concat *         repeat the visual vector, concat the whole decoder, one LSTM')
    print('  2. core_plus_extras     that LSTM sees bbox columns 0:4 only; each later span has its own LSTM')
    print('  3. adaptive_branches    visual + kinematic group + geometric group + BranchAttnPool')
    raw = input('Select [1-3] (default 1): ').strip()
    mapping = {'1': 'pie_concat', '2': 'core_plus_extras', '3': 'adaptive_branches', '': default}
    return mapping.get(raw, default)

def choose_temporal_for_menu(opts, default='convlstm2d'):
    """Call from the run menu. Writes temporal_strategy, temporal_encoder,
    temporal_hp and temporal_fusion into opts."""
    strategy = interactive_pick_temporal_strategy()
    name = interactive_pick_temporal_encoder(
        strategy, default=opts.get('temporal_encoder', default))
    hp = interactive_pick_temporal_hyperparams(name)
    if strategy == 'factorized':
        _print_factorized_cost(name, opts)
    opts['temporal_strategy'] = strategy
    opts['temporal_encoder'] = name
    opts['temporal_hp'] = hp
    opts['temporal_fusion'] = interactive_pick_temporal_fusion(
        default=opts.get('temporal_fusion') or 'pie_concat')
    print(f"[config] TEMPORAL_FUSION = {opts['temporal_fusion']}")
    return opts

def build_from_run_opts(opts, seq_len, decoder_dim, lstm_units, dropout,
                        recurrent_dropout, l2_value, vgg_shape=(7, 7, 512),
                        recurrent_activation='hard_sigmoid'):
    """Rebuild the model from the dict saved in hyperparams.json."""
    name = opts.get('temporal_encoder', 'convlstm2d')
    hp = dict(opts.get('temporal_hp') or {})
    base = name[len('convfact_'):] if str(name).startswith('convfact_') else name
    if base in ('s4', 's4d') and not str(name).startswith('convfact_'):
        n_default = 64
    else:
        n_default = 16
    return build_pie_temporal_model(
        name,
        seq_len=seq_len,
        vgg_shape=tuple(vgg_shape),
        decoder_dim=decoder_dim,
        d_model=hp.pop('d_model', 128),
        convlstm_filters=hp.pop('convlstm_filters', 64),
        convlstm_kernel=hp.pop('convlstm_kernel', 2),
        n_state=hp.pop('n_state', n_default),
        num_heads=hp.pop('num_heads', 4),
        num_layers=hp.pop('num_layers', 2),
        ff_mult=hp.pop('ff_mult', 4),
        mimo_rank=hp.pop('mimo_rank', 4),
        lstm_units=lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        recurrent_activation=recurrent_activation,
        l2_value=l2_value,
        fusion_mode=opts.get('temporal_fusion', 'pie_concat'),
        feature_spans=opts.get('feature_spans'),
        visual_norm=opts.get('visual_norm'),
        capacity=opts.get('capacity'),
        percentile_scale=opts.get('percentile_scale'))


CUSTOM_OBJECTS = {
    'SpatialStem': SpatialStem,
    'TemporalTransformer': TemporalTransformer,
    'S4Layer': S4Layer,
    'S4DLayer': S4DLayer,
    'MambaBlock': MambaBlock,
    'Mamba3Block': Mamba3Block,
    'LastStep': LastStep,
    'MatchedConvLSTM': MatchedConvLSTM,
    'PerPositionTemporal': PerPositionTemporal,     # NEW
}