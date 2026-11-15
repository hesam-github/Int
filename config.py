# config.py
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Config:

    # data
    data_path:              str     = r'D:\PIE codes\PIE_dataset'
    backbone:               str     = 'vgg16'
    observe_length:         int     = 15       # Ablation point. Rebuild dataset per value
    seq_overlap_rate:       float   = 0.5
    normalize_bbox:         bool    = True

    # model: Shared across all four encoders. we Keep these identical when comparing
    hidden_dim:             int     = 128      # Reference num_hidden_units
    use_bbox:               bool    = True
    input_dropout:          float   = 0.4      # Reference lstm_dropout (input)
    head_dropout:           float   = 0.2
    num_layers:             int     = 1        # GRU-specific (change for parameter match)
    gru_dropout:            float   = 0.0

    # optimization (verified from pie_intent.py)
    batch_size:             int     = 128
    epochs:                 int     = 100
    optimizer:              str     = 'rmsprop'
    lr:                     float   = 1e-5     # Reference optimizer_params['lr']
    weight_decay:           float   = 2e-3     # ~ 2x keras L2(1e-3) gradient
    grad_clip:              float   = 0.0      # reference clip_value = 0.0 (OFF)

    rmsprop_alpha:           float   = 0.9      # == Keras RMSprop 'rho'
    rmsprop_eps:             float   = 1e-7     # == Keras RMSprop 'epsilon'

    # scheduler + early stopping
    sched_factor:           float   = 0.5      # Reference step_drop_rate
    sched_patience:         int     = 10        # Reference plateau_patience
    min_lr:                 float   = 1e-7     # Reference min_lr
    early_stop_patience:    int     = 30       # Reference used 5
  
    early_stop_min_delta:   float   = 0.0
    monitor:                str     = 'val_loss'  # 'val_auc'(max) | 'val_loss'(min)
    monitor_mode:           str     = 'min'      # val set is tiny -> AUC steadier than loss

    # misc
    seed:                   int     = 42
    num_workers:            int     = 0        # windows
    device:                 str     = 'cuda'   # auto-falls back to cpu if gpu unavailable
    model_name:             str     = 'gru'    # 'gru' | 'convlstm_encdec'

    # output naming
    # out_dir=None  -> auto-built as  runs/<model_name>_obs<observe_length>[_<timestamp>]
    # out_dir='...' -> used exactly as given (tests / deliberate overwrites)
    out_dir:                str     = None
    use_timestamp:          bool    = True

    # live training curves (Jupyter only; silently ignored in terminal runs)
    live_plot:              bool    = True

    # ── PIEConvLSTMEncDec reference model (separate from the 4-way scaffold) ──
   
    convlstm_filters:       int     = 64      # reference convlstm_num_filters
    convlstm_kernel_size:   int     = 2       # reference convlstm_kernel_size
    recurrent_dropout:      float   = 0.2     # reference lstm_recurrent_dropout
    balance_classes:        bool    = True    # video-stratified undersampling (train/val only)
    balance_seed:           int     = 42

    def __post_init__(self):
        if self.out_dir is None:
            name = f"{self.model_name}_obs{self.observe_length}"
            if self.use_timestamp:
                name += datetime.now().strftime('_%Y%m%d_%H%M%S')
            self.out_dir = f"runs/{name}"

    def run_tag(self):
        """Short identity string used in printouts and plot titles."""
        return f"{self.model_name}, obs={self.observe_length}"
