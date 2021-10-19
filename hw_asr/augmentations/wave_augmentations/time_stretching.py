import random

import librosa
import torch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.min_stretch = kwargs.get("min_stretch", 0.75)
        self.max_stretch = kwargs.get("max_stretch", 1.25)

    def __call__(self, data: torch.Tensor):
        stretch_coef = self.min_stretch + (self.max_stretch - self.min_stretch) * random.random()
        x = librosa.effects.time_stretch(data.numpy().squeeze(), stretch_coef)
        return torch.from_numpy(x)
