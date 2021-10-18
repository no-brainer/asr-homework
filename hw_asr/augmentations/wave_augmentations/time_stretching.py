import random

import librosa
import torch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: torch.Tensor):
        stretch_coef = 0.5 + 1.5 * random.random()
        x = librosa.effects.time_stretch(data.numpy().squeeze(), stretch_coef)
        return torch.from_numpy(x)
