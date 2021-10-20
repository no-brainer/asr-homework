import torch
from torch.nn import Sequential
import torchaudio


from hw_asr.augmentations.base import AugmentationBase


class SpecAugment(AugmentationBase):
    def __init__(self, *args, **kwargs):
        max_freq_mask = kwargs.get("max_freq_mask", 20)
        max_time_mask = kwargs.get("max_time_mask", 100)

        self.freq_mask = torchaudio.transforms.FrequencyMasking(max_freq_mask)
        self.time_mask = torchaudio.transforms.TimeMasking(max_time_mask)

    def __call__(self, data: torch.Tensor):
        val = data.mean()
        data = self.freq_mask(data, val)
        return self.time_mask(data, val)
