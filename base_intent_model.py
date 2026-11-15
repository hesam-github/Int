"""
Shared base for the four way temporal-encoder comparison (GRU/ Transformer/ S4D/ Mamba).
Every subclass overrides ONLY encode(). Fusion, input projection, readout and head live 
here so the temporal module is the single different variable across all 4 models.

Dataset contract (PIEIntentionDataset.__getitem__):
    features : (B,T, 4096)          VGG16 fc7,      T = observe_length
    bbox     : (B, T, 4)            normalized to [0,1]
    label    : (B,) int64 {0,1}     -> cast to float in train.py for BCE 
Fixed-length windows -> no padding -> last-timestep readout is always valid.

NOTE: unchanged from before. PIEConvLSTMEncDec (the separate reference-paper
reproduction) deliberately does NOT subclass this -- see its own docstring.
"""

import torch
import torch.nn as nn

class BaseIntentionModel(nn.Module):
    def __init__(self,
                feature_dim:    int = 4096,
                bbox_dim:       int = 4,
                hidden_dim:     int = 128,      # reference num_hidden_units
                use_bbox:       bool = True,
                input_dropout:  float = 0.4,    # reference lstm_dropout (input)
                head_dropout:   float = 0.0):   # reference Dense has no dropout
        super().__init__()
        self.feature_dim = feature_dim
        self.bbox_dim = bbox_dim
        self.hidden_dim = hidden_dim
        self.use_bbox = use_bbox
        self.input_dim = feature_dim + (bbox_dim if use_bbox else 0)

        # Shared + constant across all 4 models -> isolates the temporal module.
        self.input_proj = nn.Linear(self.input_dim , hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout)

        # Single logit head (matches reference Dense(1) + sigmoid).
        # no sigmoid here. sigmoid is applied in BCEWithLoitsLoss at eval only
        self.head = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(hidden_dim,1),
        )
    
    # The only class that each subclass will implements by it's own class!
    def encode (self,x:torch.Tensor) ->torch.Tensor:
        """
        x: (B,T, hidden_dim) -> full state
        """
        raise NotImplementedError("Subclass must implement it's own encode().") 
    
    # fixed for all models
    def _fuse(self,features,bbox):
        if self.use_bbox:
            return torch.cat([features,bbox], dim = -1)   # (B,T ,input_dim). e.g: result:   (B, T, 4100)
        return features

    def _readout(self,seq):
        # piscks the last time step
        # turns (B,T, hidden_dim) -> (B,hidden_dim) 
        return seq[:,-1,:]        
    # train                   
    def forward(self, features , bbox = None):
        x = self._fuse(features, bbox)              # (B, T, input_dim)
        x = self.input_dropout(self.input_proj(x))  # (B, T, hidden_dim)
       
        # temporal process/ the only part that differs in models we use 
        seq = self.encode(x)                        # (B, T, hidden_dim)

        # (B, T , hidden_dim) -> (B,hidden_dim)
        h = self._readout(seq)

        # (B,1) -> (B,): raw logit  / head
        logit = self.head(h).squeeze(-1)
        return logit

    # Inference
    @torch.no_grad()
    def predict_proba(self, features, bbox = None):
        return torch.sigmoid(self.forward(features, bbox))
    
    def num_params(self, trainable_only = True):
        ps = (p for p in self.parameters()
              if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)  
    def encoder_params(self):
        """
        Params NOT in the shared scaffold = the temporal module only.
        The shared scaffold is all body but the temporal module.
        We use it to parameter-match Transformer/S4D/Mamba against GRU.
        """
        shared = set(self.input_proj.parameters()) | set(self.head.parameters())
        return sum(p.numel() for p in self.parameters() if p not in shared)
