# shared/transformer.py

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position embeddings to a sequence."""

    def __init__(self, hidden_dim: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, hidden_dim)
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    """
    Causal transformer for sequence regression.

    Each time step t attends only to positions ≤ t (upper-triangular mask),
    making the model operationally valid — it cannot see future inputs.

    Interface:
        forward(x) -> y
        x: (batch, seq_len, num_features)
        y: (batch, seq_len, num_targets)
    """

    def __init__(self, num_features: int, num_targets: int, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        hidden_dim: int      = m["hidden_dim"]
        num_layers: int      = m["num_layers"]
        num_heads: int       = m["num_heads"]
        feedforward_dim: int = m["feedforward_dim"]
        dropout: float       = m["dropout"]

        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.pos_enc    = SinusoidalPositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, num_targets)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_head.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Additive attention mask: 0 for allowed positions, -inf for future positions."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, num_features)
        T = x.size(1)
        h = self.pos_enc(self.input_proj(x))           # (B, T, hidden_dim)
        mask = self._causal_mask(T, x.device)          # (T, T)
        h = self.encoder(h, mask=mask)                 # (B, T, hidden_dim)
        return self.output_head(h)                     # (B, T, num_targets)
