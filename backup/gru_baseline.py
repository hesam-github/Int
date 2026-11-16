""" GRU temporal encoder: the recurrent baseline of the 4-way comparison."""
import torch.nn as nn
from base_intent_model import BaseIntentionModel

class GRUIntentionModel(BaseIntentionModel):
    def __init__(self, num_layers: int = 1,gru_dropout: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gru = nn.GRU(
            input_size  = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = gru_dropout if num_layers > 1 else 0.0,
        )
    def encode(self,x):
        # x comes from forward in shape of: (B, T, hidden_dim)
        out , _ = self.gru(x)         #out: (B, T, hidden_dim) --full sequence
        return out                   # by forward takes out [:, -1, :] by _readout()
