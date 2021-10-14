import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    for key in ["duration", "text", "audio_path"]:
        result_batch[key] = [item[key] for item in dataset_items]

    for key in ["audio", "spectrogram", "text_encoded"]:
        vals = []
        for item in dataset_items:
            val = item[key]
            if key == "spectrogram":
                val = val.transpose(1, 2)
            val = val.squeeze(0)
            vals.append(val)
        result_batch[key] = pad_sequence(vals, batch_first=True, padding_value=-1)
        result_batch[f"{key}_length"] = torch.IntTensor([item.size(0) for item in vals])

    return result_batch
