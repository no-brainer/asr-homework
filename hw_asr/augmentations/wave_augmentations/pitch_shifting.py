import random

import librosa
import torch

from hw_asr.augmentations.base import AugmentationBase


class PitchShifting(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.min_shift = kwargs.get("min_shift", -5)
        self.max_shift = kwargs.get("max_shift", 5)

    def __call__(self, data: torch.Tensor):
        shift = random.randint(self.min_shift, self.max_shift)
        x = librosa.effects.pitch_shift(data.numpy().squeeze(), shift)
        return torch.from_numpy(x)
