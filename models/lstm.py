# models/lstm.py

import torch.nn as nn
from torch import Tensor


class LSTMModel(nn.Module):
    """
    LSTM for sequence regression — same interface as TransformerModel.

    LSTM is inherently causal: hidden state at t depends only on inputs 1..t.

    Interface:
        forward(x) -> y
        x: (batch, seq_len, num_features)
        y: (batch, seq_len, num_targets)
    """

    def __init__(self, num_features: int, num_targets: int, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        hidden_dim: int = m["hidden_dim"]
        num_layers: int = m["num_layers"]
        dropout: float  = m["dropout"]

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_dim, num_targets)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, num_features)
        out, _ = self.lstm(x)          # (B, T, hidden_dim)
        return self.output_head(out)   # (B, T, num_targets)
