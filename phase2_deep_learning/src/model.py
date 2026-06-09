"""
model.py
--------
CNN-LSTM hybrid architecture for spoken-digit classification.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~

  Input: (B, C, T)  — batch of MFCC tensors
       C = n_channels (e.g. 120 with delta features)
       T = max_len   (e.g. 128 time frames)

  ┌─────────────────────────────────────────────────────────┐
  │  CNN Feature Extractor                                  │
  │  Conv1 (C→64)  BN  ReLU  Pool                          │
  │  Conv2 (64→128) BN  ReLU  Pool                         │
  │  Conv3 (128→256) BN  ReLU  Pool  Dropout               │
  └─────────────────────────────────────────────────────────┘
              ↓  reshape → (B, T', 256)
  ┌─────────────────────────────────────────────────────────┐
  │  Bidirectional LSTM  (2 layers, hidden=256)             │
  │  + layer normalisation on final hidden state            │
  └─────────────────────────────────────────────────────────┘
              ↓  (B, 512)  [concat fwd + bwd]
  ┌─────────────────────────────────────────────────────────┐
  │  Classifier head                                        │
  │  Linear(512→256)  LayerNorm  GELU  Dropout              │
  │  Linear(256→128)  GELU  Dropout                         │
  │  Linear(128→10)                                         │
  └─────────────────────────────────────────────────────────┘

The model uses 1-D convolutions along the time axis (after treating
the MFCC coefficient dimension as channels), which is the natural
representation for CNN-over-time for audio.

Author : Phase-2 CNN-LSTM project
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    1-D Convolutional block: Conv1d → BatchNorm1d → Activation → MaxPool1d.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int
    pool_size : int
        Max-pooling kernel (and stride). Set to 1 to skip pooling.
    dropout : float
        Dropout probability applied after the activation.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        pool_size:    int = 2,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNNLSTM(nn.Module):
    """
    CNN + Bidirectional-LSTM hybrid for spoken-digit recognition.

    Parameters
    ----------
    n_channels : int
        Number of input feature channels (MFCC coefficient dimension).
    n_classes : int
        Number of output classes (10 digits).
    cnn_channels : tuple of int
        Output channel sizes for each convolutional block.
    lstm_hidden : int
        Hidden size of the LSTM (per direction).
    lstm_layers : int
        Number of stacked LSTM layers.
    lstm_dropout : float
        Dropout between LSTM layers (applied when lstm_layers > 1).
    fc_dropout : float
        Dropout in the fully-connected head.
    """

    def __init__(
        self,
        n_channels:   int         = 120,
        n_classes:    int         = 10,
        cnn_channels: tuple       = (64, 128, 256),
        lstm_hidden:  int         = 256,
        lstm_layers:  int         = 2,
        lstm_dropout: float       = 0.30,
        fc_dropout:   float       = 0.40,
    ):
        super().__init__()

        self.n_channels  = n_channels
        self.n_classes   = n_classes
        self.lstm_hidden = lstm_hidden

        # ── CNN blocks ────────────────────────────────────────────────
        cnn_layers = []
        in_ch = n_channels
        pool_sizes = [2, 2, 2]  # each halves the time dimension

        for i, (out_ch, pool) in enumerate(zip(cnn_channels, pool_sizes)):
            cnn_layers.append(
                ConvBlock(in_ch, out_ch, kernel_size=3, pool_size=pool, dropout=0.15)
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_channels = cnn_channels[-1]   # e.g. 256

        # ── Bi-LSTM ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = cnn_out_channels,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout = lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)

        # ── Classifier head ───────────────────────────────────────────
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(p=fc_dropout * 0.5),
            nn.Linear(128, n_classes),
        )

        # Weight initialisation
        self._init_weights()

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("CNNLSTM initialised — trainable parameters: {:,}".format(total))

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # LSTM orthogonal init
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, C, T)

        Returns
        -------
        logits : Tensor  (B, n_classes)
        """
        # CNN: (B, C, T) → (B, cnn_out_channels, T')
        x = self.cnn(x)                        # shape: (B, 256, T/8)

        # Prepare for LSTM: (B, T', features)
        x = x.permute(0, 2, 1)                 # (B, T', 256)

        # Bi-LSTM: (B, T', lstm_hidden*2)
        x, _ = self.lstm(x)                    # (B, T', 512)

        # Use the last time-step hidden state
        x = x[:, -1, :]                        # (B, 512)
        x = self.lstm_norm(x)

        # Classifier
        logits = self.classifier(x)            # (B, 10)
        return logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CNNLSTM("
            f"n_channels={self.n_channels}, "
            f"n_classes={self.n_classes}, "
            f"lstm_hidden={self.lstm_hidden}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = CNNLSTM(n_channels=120, n_classes=10)
    print(model)

    dummy = torch.randn(8, 120, 128)   # (batch=8, C=120, T=128)
    out   = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")   # expected: (8, 10)
