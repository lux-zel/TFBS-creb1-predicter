"""
1D CNN for CREB1 binding prediction.

Input:  (batch, 4, L) one-hot DNA, channel-first (A, C, G, T).
Output: (batch,) raw logits (one per sequence; sigmoid applied in loss).

Architecture:
    Conv1d -> ReLU -> Conv1d -> ReLU -> MaxPool
    Conv1d -> ReLU -> Conv1d -> ReLU -> MaxPool
    AdaptiveAvgPool1d -> Flatten -> Dropout -> Linear
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Creb1CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        n_filters: int = 64,
        kernel_size: int = 11,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, n_filters, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_filters, n_filters, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            # Block 2
            nn.Conv1d(n_filters, n_filters * 2, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            # Block 3
            nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_filters * 4, n_filters * 4, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
        )

        # Length-agnostic head: average over the remaining length axis
        self.global_pool = nn.AdaptiveAvgPool1d(1)   # (B, C, L) -> (B, C, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                # (B, C, 1) -> (B, C)
            nn.Dropout(dropout),
            nn.Linear(n_filters * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 4, L) float32

        Returns
        -------
        (B,) raw logits
        """
        h = self.features(x)             # (B, C, L')
        h = self.global_pool(h)          # (B, C, 1)
        return self.classifier(h).squeeze(-1)   # (B,)


def build_model(cfg=None) -> Creb1CNN:
    """Convenience factory. Keeps notebook code stable if the class signature changes."""
    if cfg is None:
        return Creb1CNN()
    return Creb1CNN(
        n_filters=getattr(cfg, "n_filters", 64),
        kernel_size=getattr(cfg, "kernel_size", 11),
        dropout=getattr(cfg, "dropout", 0.3),
    )