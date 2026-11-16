"""
PyTorch port of the ACTUAL reference intention model from pie_intent.py
(PIEIntent.pie_convlstm_encdec), verified against the real source at
https://github.com/aras62/PIEPredict/blob/master/pie_intent.py -- not
guessed from memory or from a partial code snippet.

ARCHITECTURE (verified line-by-line against the Keras source):

    encoder_input: (B, T, 512, 7, 7)   raw VGG16 conv5 block output,
                                        NOT flattened (Keras: VGG16(include_top=False),
                                        output_shape (7,7,512); channel-last in Keras,
                                        channel-first here -- PyTorch convention)
        |
    ConvLSTM2D(filters=64, kernel_size=2, padding='valid',
               dropout=0.4, recurrent_dropout=0.2,
               kernel/recurrent/bias L2(1e-3), return_sequences=False)
        |
    encoder_output: (B, 64, 6, 6)   [7-2+1=6 with valid padding, kernel=2]
        |
    Flatten -> (B, 2304)
        |
    RepeatVector(T=15) -> (B, 15, 2304)      (encoder vector repeated per timestep)
        |
    Concatenate([repeated_encoder_vec, decoder_input], dim=-1)
        where decoder_input = bbox_seq, shape (B, 15, 4)
        -> (B, 15, 2308)
        |
    LSTM(units=128, dropout=0.4, recurrent_dropout=0.2,
         kernel/recurrent/bias L2(1e-3), return_sequences=False)
        |
    decoder_output: (B, 128)   [last timestep hidden state only]
        |
    Dense(1, activation='sigmoid')  -> (B, 1)


This file is a STANDALONE reference-paper reproduction, used only for
intention estimation (the trajectory/speed prediction heads of the
original repo -- PIEPredict, train_predict() -- are intentionally NOT
ported here; only train_intent()'s architecture is). It is NOT part of
the GRU/Transformer/S4D/Mamba four-way comparison scaffold
(base_intent_model.py): it does not subclass BaseIntentionModel, has a
different constructor contract, and uses a genuinely different input
representation (raw spatial conv5 maps + late bbox fusion at the decoder,
vs. the scaffold's flat fc7 vectors + early bbox fusion). It exists to
answer a specific question -- "does preserving spatial VGG16 features
(instead of flattening to fc7) change the ceiling we're hitting?" -- as a
controlled side experiment, not as a replacement for the four-way study.
"""
import torch
import torch.nn as nn


def _locked_dropout_mask(shape, p, device, training):
    """
    Returns a single dropout mask of the given shape, to be reused across
    ALL timesteps of one forward call (locked/variational dropout) -- NOT
    a fresh mask per timestep. Returns an all-ones tensor if not training
    or p == 0 (matches Keras: dropout layers are identity at inference).
    """
    if not training or p == 0.0:
        return torch.ones(shape, device=device)
    keep_prob = 1.0 - p
    mask = torch.bernoulli(torch.full(shape, keep_prob, device=device))
    return mask / keep_prob   # inverted dropout, scales at train time


class ConvLSTMCell(nn.Module):
    """
    Single-layer ConvLSTM cell matching Keras's ConvLSTM2D semantics.

    STILL-PRESENT BUG FOUND IN THIS FILE (separate from the lazy-decoder
    bug documented on PIEConvLSTMEncDec below, which WAS correctly fixed
    here): this class computed gates via a SINGLE Conv2d over
    concat([x_t, h_{t-1}], dim=1). That crashes -- confirmed by actually
    running this file's forward() at T=15, reproducing the exact error
    from the original traceback ("Expected size 7 but got size 6") --
    because x_t is the native 7x7 VGG conv-map size while h is already
    the 'valid'-padding-reduced 6x6 size (h is initialized at (H_out,W_out)
    BEFORE the loop even starts, so this fails from t=0, not just t>=1).

    A single conv over concatenated channels can only work if x and h
    share the same spatial size, which they structurally cannot here:
    Keras ConvLSTM2D actually uses TWO separate convolutions -- an
    input-to-state conv with the layer's own 'valid' padding (shrinking
    7x7->6x6) and a state-to-state (recurrent) conv that ALWAYS uses
    'same' padding internally regardless of the outer padding setting,
    since h must stay a CONSTANT 6x6 across all timesteps. Splitting one
    KxK conv over (in+hidden) channels into two KxK convs over the
    separate channel groups does not change the total parameter count --
    only the padding mode differs per branch.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 input_dropout=0.0, recurrent_dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        # FIX: two convs instead of one conv on concat(x, h).
        self.conv_x = nn.Conv2d(in_channels, 4 * hidden_channels,
                                kernel_size=kernel_size, padding=0, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, 4 * hidden_channels,
                                kernel_size=kernel_size, padding='same', bias=False)

    def forward(self, x_seq):
        """
        x_seq: (B, T, C_in, H, W) -> returns LAST hidden state (B, hidden, H', W')
        (return_sequences=False in the reference, so only the last state
        is needed -- computing the full sequence would be wasteful.)
        """
        B, T, C, H, W = x_seq.shape
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1
        device = x_seq.device

        h = torch.zeros(B, self.hidden_channels, H_out, W_out, device=device)
        c = torch.zeros(B, self.hidden_channels, H_out, W_out, device=device)

        # locked masks: same mask reused at every timestep this forward call
        x_mask = _locked_dropout_mask((B, C, H, W), self.input_dropout,
                                       device, self.training)
        h_mask = _locked_dropout_mask((B, self.hidden_channels, H_out, W_out),
                                       self.recurrent_dropout, device, self.training)

        for t in range(T):
            x_t = x_seq[:, t] * x_mask
            h_dropped = h * h_mask
            gates = self.conv_x(x_t) + self.conv_h(h_dropped)   # FIX: sum, not cat+single-conv
            i, f, g, o = gates.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)

        return h   # (B, hidden_channels, H_out, W_out) -- final timestep only


class LockedDropoutLSTMCell(nn.Module):
    """
    Single-layer LSTM cell with locked input + recurrent dropout, matching
    Keras LSTM(dropout=..., recurrent_dropout=...) semantics for the
    DECODER stage. Plain nn.LSTMCell has neither of these, so gates are
    implemented explicitly via two nn.Linear layers (matches Keras's
    combined kernel / recurrent_kernel weight matrices).
    """
    def __init__(self, input_size, hidden_size,
                 input_dropout=0.0, recurrent_dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x_seq):
        """
        x_seq: (B, T, input_size) -> returns LAST hidden state (B, hidden_size)
        (return_sequences=False in the reference.)
        """
        B, T, _ = x_seq.shape
        device = x_seq.device
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)

        x_mask = _locked_dropout_mask((B, x_seq.shape[-1]), self.input_dropout,
                                       device, self.training)
        h_mask = _locked_dropout_mask((B, self.hidden_size), self.recurrent_dropout,
                                       device, self.training)

        for t in range(T):
            x_t = x_seq[:, t] * x_mask
            h_dropped = h * h_mask
            gates = self.x2h(x_t) + self.h2h(h_dropped)
            i, f, g, o = gates.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)

        return h


class PIEConvLSTMEncDec(nn.Module):
    """
    Exact-architecture PyTorch port of PIEIntent.pie_convlstm_encdec().

    BUG FIX vs. an earlier draft of this file: the decoder used to be built
    LAZILY, inside forward(), on the theory that its input size (encoder
    flatten size + bbox_dim) is only "really" known once you see a real
    batch. That reasoning was wrong in a way that caused a real training
    bug: in train.py, the optimizer is created right after the model,
    BEFORE any forward() call --

        model = build_model(cfg).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), ...)

    -- so with a lazy decoder, `model.parameters()` at that point did NOT
    include the decoder's weights at all (self.decoder was still None).
    The decoder would then be constructed on the first forward() call, but
    its parameters would NEVER be registered with the optimizer, so
    optimizer.step() would silently never update them -- the decoder (and
    therefore most of the model's capacity) would never actually train,
    with no error or warning anywhere.

    The fix: the encoder's output spatial size is NOT actually data-
    dependent -- it's fully determined at construction time by
    conv_input_spatial (always 7 for a 224x224 VGG16 conv5 crop) and
    convlstm_kernel_size ('valid' convolution shrinks it by kernel_size-1).
    So the decoder is now built eagerly in __init__, and forward() just
    asserts the incoming spatial size matches what was assumed, instead of
    silently accepting whatever shape shows up first.
    """
    def __init__(self,
                 conv_in_channels=512,       # VGG16 conv5 output channels
                 convlstm_filters=64,
                 convlstm_kernel_size=2,
                 decoder_hidden=128,
                 bbox_dim=4,
                 input_dropout=0.4,
                 recurrent_dropout=0.2,
                 conv_input_spatial=7):      # VGG16 conv5 spatial size @ 224x224 input
        super().__init__()
        self.encoder = ConvLSTMCell(
            in_channels=conv_in_channels,
            hidden_channels=convlstm_filters,
            kernel_size=convlstm_kernel_size,
            input_dropout=input_dropout,
            recurrent_dropout=recurrent_dropout,
        )

        # 'valid' convolution shrinks spatial size by (kernel_size - 1).
        # This is analytically known, NOT data-dependent -- so the decoder
        # can (and must, see class docstring) be built right now.
        encoder_out_spatial = conv_input_spatial - convlstm_kernel_size + 1
        if encoder_out_spatial <= 0:
            raise ValueError(
                f"convlstm_kernel_size={convlstm_kernel_size} is too large for "
                f"conv_input_spatial={conv_input_spatial}: 'valid' padding would "
                f"produce a non-positive spatial output size."
            )
        self._conv_input_spatial = conv_input_spatial
        self._encoder_out_spatial = encoder_out_spatial
        encoder_flat_size = convlstm_filters * encoder_out_spatial * encoder_out_spatial

        self.decoder = LockedDropoutLSTMCell(
            input_size=encoder_flat_size + bbox_dim,
            hidden_size=decoder_hidden,
            input_dropout=input_dropout,
            recurrent_dropout=recurrent_dropout,
        )
        self.head = nn.Linear(decoder_hidden, 1)

    def forward(self, conv_features, bbox):
        """
        conv_features: (B, T, 512, 7, 7)  raw VGG16 conv5 output per frame
        bbox:          (B, T, 4)          decoder input (SAME T as encoder --
                                          verified, see module docstring)
        returns: (B,) raw logit (sigmoid applied by BCEWithLogitsLoss at
                 train time / by predict_proba() at inference)
        """
        B, T, C, H, W = conv_features.shape
        if H != self._conv_input_spatial or W != self._conv_input_spatial:
            raise ValueError(
                f"conv_features spatial size is ({H}x{W}), but this model was "
                f"constructed for conv_input_spatial={self._conv_input_spatial}. "
                f"Rebuild the model with the matching conv_input_spatial if "
                f"you're using a different backbone/crop size."
            )

        h_enc = self.encoder(conv_features)              # (B, filters, H', W')
        encoder_flat = h_enc.flatten(start_dim=1)         # (B, filters*H'*W')

        encoder_repeated = encoder_flat.unsqueeze(1).expand(-1, T, -1)   # (B,T,flat)
        decoder_in = torch.cat([encoder_repeated, bbox], dim=-1)         # (B,T,flat+4)

        h_dec = self.decoder(decoder_in)                  # (B, decoder_hidden)
        logit = self.head(h_dec).squeeze(-1)              # (B,)
        return logit

    @torch.no_grad()
    def predict_proba(self, conv_features, bbox):
        return torch.sigmoid(self.forward(conv_features, bbox))

    def num_params(self, trainable_only=True):
        """
        Same contract as BaseIntentionModel.num_params(), reproduced here
        because this class is a plain nn.Module (not a BaseIntentionModel
        subclass) -- train.py's main() calls model.num_params() /
        model.encoder_params() unconditionally regardless of which model
        was built, so both must exist on every model in MODEL_REGISTRY.
        """
        ps = (p for p in self.parameters()
              if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)

    def encoder_params(self):
        """
        Params outside the final classification head (self.head), i.e. the
        ConvLSTM encoder + LSTM decoder combined. Analogous in spirit to
        BaseIntentionModel.encoder_params() from the four-way scaffold,
        though this model has no separate shared input_proj layer to also
        exclude -- bbox/image features feed the encoder/decoder directly,
        not through a projection layer.
        """
        head_params = set(self.head.parameters())
        return sum(p.numel() for p in self.parameters() if p not in head_params)

    def reference_param_groups(self, weight_decay_regularized):
        """
        MISSING FROM THE ORIGINAL FILE -- train.py's optimizer-construction
        branch for model_name=='convlstm_encdec' calls this unconditionally;
        without it, model building crashes with AttributeError before a
        single epoch runs (confirmed by actually calling train.main() with
        this model).

        The reference applies L2(reg_value) as kernel_regularizer +
        recurrent_regularizer + bias_regularizer to BOTH the ConvLSTM
        encoder and the decoder LSTM in pie_intent.py -- but NOT to the
        final Dense head (pie_convlstm_encdec()'s Dense(...) call has no
        regularizer argument). Keras adds reg_value*w^2 to the loss
        (gradient contribution 2*reg_value*w); PyTorch's optimizer
        weight_decay adds weight_decay*w directly to the gradient, so pass
        weight_decay_regularized=2*reg_value here to match.
        """
        regularized = list(self.encoder.parameters()) + list(self.decoder.parameters())
        head_params = list(self.head.parameters())
        return [
            {'params': regularized, 'weight_decay': weight_decay_regularized},
            {'params': head_params, 'weight_decay': 0.0},
        ]