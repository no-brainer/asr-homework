import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hw_asr.base import BaseModel
from hw_asr.model.utils import get_same_padding


class ResidualBlock(nn.Module):
    # https://arxiv.org/pdf/1603.05027.pdf

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualBlock, self).__init__()

        padding = get_same_padding(kernel)

        curr_channels = in_channels
        layers = []
        for _ in range(2):
            layers.extend([
                nn.LayerNorm(n_feats),
                nn.GELU(),
                nn.Conv2d(curr_channels, out_channels, kernel, stride, padding=padding),
                nn.Dropout(dropout),
            ])
            curr_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class CRNN(BaseModel):

    def __init__(
            self,
            n_feats,
            n_class,
            n_cnn_layers,
            n_rnn_layers,
            rnn_dim,
            stride=2,
            dropout=0.1,
            *args,
            **kwargs
    ):
        super(CRNN, self).__init__(n_feats, n_class, *args, **kwargs)
        n_feats = n_feats // 2
        # cnn for extracting hierarchic features
        self.cnn = nn.Conv2d(1, 32, kernel_size=3, stride=stride, padding=get_same_padding(3))

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualBlock(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Sequential(
            nn.Linear(n_feats * 32, rnn_dim),
            nn.GELU(),
        )
        self.birnn_layers = nn.LSTM(rnn_dim, hidden_size=rnn_dim, bidirectional=True,
                                    num_layers=n_rnn_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * rnn_dim, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        # initially spectrogram is (batch, time, feature)
        spectrogram = spectrogram.unsqueeze(1)
        out = self.cnn(spectrogram)
        out = self.rescnn_layers(out)

        out = out.transpose(1, 2)  # (batch, channels, time, feats) -> (batch, time, channels, feats)
        sizes = out.size()
        out = out.view(sizes[0], sizes[1], sizes[2] * sizes[3])  # (batch, time, feats)

        out = self.fully_connected(out)
        out_packed = pack_padded_sequence(
            out, kwargs["spectrogram_length"], batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.birnn_layers(out_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        return self.classifier(out)
