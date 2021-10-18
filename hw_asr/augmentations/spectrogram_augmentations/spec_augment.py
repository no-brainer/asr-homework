import torch
from torch.nn import Sequential
import torchaudio


from hw_asr.augmentations.base import AugmentationBase


class SpecAugment(AugmentationBase):
    def __init__(self, *args, **kwargs):
        max_freq_mask = kwargs.get("max_freq_mask", 20)
        max_time_mask = kwargs.get("max_time_mask", 100)

        self._aug = Sequential(
            torchaudio.transforms.FrequencyMasking(max_freq_mask),
            torchaudio.transforms.TimeMasking(max_time_mask),
        )

    def __call__(self, data: torch.Tensor):
        return self._aug(data)
