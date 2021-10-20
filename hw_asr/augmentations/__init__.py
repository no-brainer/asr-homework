from collections import Callable
from typing import List

import hw_asr.augmentations.spectrogram_augmentations
import hw_asr.augmentations.wave_augmentations
from hw_asr.augmentations.random_apply import RandomApply
from hw_asr.augmentations.sequential import SequentialAugmentation
from hw_asr.utils.parse_config import ConfigParser


def from_configs(configs: ConfigParser):
    wave_augs = []
    if "augmentations" in configs.config and "wave" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["wave"]:
            aug_prob = aug_dict.pop("aug_prob", None)
            wave_aug = configs.init_obj(aug_dict, hw_asr.augmentations.wave_augmentations)
            if aug_prob is not None:
                wave_aug = RandomApply(wave_aug, aug_prob)
            wave_augs.append(wave_aug)

    spec_augs = []
    if "augmentations" in configs.config and "spectrogram" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["spectrogram"]:
            aug_prob = aug_dict.pop("aug_prob", None)
            spec_aug = configs.init_obj(aug_dict, hw_asr.augmentations.spectrogram_augmentations)
            if aug_prob is not None:
                spec_aug = RandomApply(spec_aug, aug_prob)
            spec_augs.append(spec_aug)
    return _to_function(wave_augs), _to_function(spec_augs)


def _to_function(augs_list: List[Callable]):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return SequentialAugmentation(augs_list)
