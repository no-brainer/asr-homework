from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hw_asr.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, rnn_dim, rnn_layers, hidden_dim, hidden_layers, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        is_bidirectional = kwargs.get("bidirectional", False)
        dropout = kwargs.get("dropout", 0.1)

        self.rnn = nn.GRU(
            n_feats, hidden_size=rnn_dim, num_layers=rnn_layers,
            bidirectional=is_bidirectional, batch_first=True
        )

        cur_dim = 2 * rnn_dim
        layers = []
        if hidden_layers > 1:
            for _ in range(hidden_layers - 1):
                layers.extend([
                    nn.Linear(cur_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                cur_dim = hidden_dim

        layers.append(nn.Linear(cur_dim, n_class))

        self.classifier = Sequential(*layers)

    def forward(self, spectrogram, *args, **kwargs):
        # (batch, time, feature)
        spectrogram = pack_padded_sequence(
            spectrogram, kwargs["spectrogram_length"], batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(spectrogram)
        out_padded, _ = pad_packed_sequence(out, batch_first=True)
        return {"logits": self.classifier(out_padded)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
