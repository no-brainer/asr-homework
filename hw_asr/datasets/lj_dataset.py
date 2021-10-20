import json
import logging
import os
import shutil
from pathlib import Path
import re

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


DATASET_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJDataset(BaseDataset):

    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        self.pattern = re.compile("[^a-z ]")
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading the LJ dataset")
        download_file(DATASET_URL, arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"lj_index_{part}.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _normalize_text(self, text):
        text = text.lower()
        return self.pattern.sub("", text)

    def _create_index(self, part):
        index = []

        metadata_path = self._data_dir / "metadata.csv"

        if not metadata_path.exists():
            self._load_part()

        wav_dir = Path(self._data_dir / "wavs")
        with open(metadata_path, "r") as f:
            for i, line in enumerate(f):
                if (i % 10 == 0) != (part == "test"):
                    continue

                utterance_id, text, _ = line.strip().split("|")
                wav_path = wav_dir / f"{utterance_id}.wav"
                t_info = torchaudio.info(str(wav_path))
                length = t_info.num_frames / t_info.sample_rate
                index.append({
                    "path": str(wav_path.absolute().resolve()),
                    "text": self._normalize_text(text),
                    "audio_len": length,
                })
        return index
