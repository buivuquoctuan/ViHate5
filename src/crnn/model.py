"""Convolutional recurrent network for lexicon-free word recognition.

Images have height 32. The CNN reduces the height to 1 and the width by 4;
its columns become time steps for two bidirectional LSTM layers. CTC is
used by the trainer, so the model returns log probabilities including blank.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRNN(nn.Module):
    """CNN -> two stacked BiLSTMs -> character log probabilities.

    The last convolution uses a (2, 1) kernel to preserve horizontal time
    steps. This is a small documented adaptation of the original CRNN CNN.
    Valid lengths are packed before both recurrent layers: the backward LSTM
    never reads the padded tail of a shorter word.
    Optional dropout acts after the CNN and between recurrent layers. Its
    default of zero preserves baseline outputs and all checkpoint weight keys.
    """

    def __init__(self, num_classes: int, hidden_size: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if num_classes < 2 or hidden_size < 1:
            raise ValueError("num_classes must be >= 2 and hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = float(dropout)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (2, 1)), nn.ReLU(inplace=True),
        )
        # Stateless module keeps every original CNN/RNN state-dict key intact.
        self.feature_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    @staticmethod
    def output_lengths(widths: Tensor) -> Tensor:
        """Return CPU int64 CTC lengths for the two horizontal pooling layers."""
        return torch.div(widths.detach().to(device="cpu", dtype=torch.long), 4,
                         rounding_mode="floor")

    def forward(self, images: Tensor, widths: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1:3] != (1, 32):
            raise ValueError("images must have shape [batch, 1, 32, padded_width]")
        if widths.ndim != 1 or widths.numel() != images.shape[0]:
            raise ValueError("widths must contain one valid image width per sample")
        if widths.dtype not in (torch.int32, torch.int64):
            raise ValueError("widths must be an integer tensor")
        if torch.any(widths < 4) or torch.any(widths > images.shape[-1]):
            raise ValueError("valid image widths must be between 4 and padded_width")
        lengths = self.output_lengths(widths)
        features = self.feature_dropout(self.cnn(images)).squeeze(2).permute(2, 0, 1).contiguous()
        packed = pack_padded_sequence(features, lengths, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        sequence, _ = pad_packed_sequence(encoded, total_length=features.shape[0])
        return self.classifier(sequence).log_softmax(dim=2), lengths
