import os
import random

import gdown
import shutil
import torch
import torchaudio

from hw_asr.augmentations.base import AugmentationBase


class BackgroundNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.sound_folder = "./data/background_sounds"

        if not os.path.exists(self.sound_folder):
            os.mkdir(self.sound_folder)

            gdrive_id = "1HUWvM0UZO34iltVQz-pLeTRpZgpQZH-7"
            zip_path = os.path.join(self.sound_folder, "noise.zip")
            print("Downloading background sounds.")
            gdown.download(id=gdrive_id, output=zip_path)
            print("Unzipping background sounds.")
            shutil.unpack_archive(zip_path, self.sound_folder, "zip")
            os.remove(zip_path)
        else:
            print("Background sounds are loaded.")

        self.background_index = [
            filename for filename in os.listdir(self.sound_folder)
            if filename.endswith(".wav")
        ]

        self.sr = kwargs.get("sr", 16_000)

    def _load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor.squeeze(0)

    def __call__(self, data: torch.Tensor):
        sound_path = os.path.join(
            self.sound_folder,
            random.choice(self.background_index)
        )
        background_noise = self._load_audio(sound_path)
        data = data.squeeze(0)
        # offset for noise
        offset = random.randint(0, int(0.75 * data.shape[0]))
        background_noise = background_noise[:data.shape[0] - offset]

        noize_level = torch.Tensor([random.randint(0, 40)])

        noize_energy = torch.norm(background_noise)
        audio_energy = torch.norm(data)

        alpha = (audio_energy / noize_energy) * torch.pow(10, -noize_level / 20)

        data[offset:offset + background_noise.shape[0]] += alpha * background_noise
        return torch.clamp(data, -1, 1).unsqueeze(0)


