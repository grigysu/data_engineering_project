"""Multivariate LSTM baseline: input is a (seq_in, n_features) window,
output is the next `seq_out` temperature values.
"""

from __future__ import annotations

import torch
from torch import nn


class WeatherLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_out: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, seq_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_in, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take the last time step's hidden state
        return self.head(last)  # (B, seq_out)
