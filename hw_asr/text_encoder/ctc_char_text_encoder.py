import gzip
import os
import shutil
from typing import List, Tuple
import wget

import kenlm
import numpy as np
from pyctcdecode import build_ctcdecoder
import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        kenlm_path, unigram_list = self._prepare_kenlm()
        self.ctc_decoder = build_ctcdecoder(
            [""] + alphabet,  # pyctcdecoder only works with "" as blank
            kenlm_path,
            unigram_list
        )

    def _prepare_kenlm(self):
        # based on https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/01_pipeline_nemo.ipynb
        lm_folder = "./language_models"
        if not os.path.exists(lm_folder):
            os.mkdir(lm_folder)

        lm_gzip_path = os.path.join(lm_folder, "3-gram.pruned.1e-7.arpa.gz")
        if not os.path.exists(lm_gzip_path):
            print("Downloading pruned 3-gram model.")
            lm_url = "http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
            lm_gzip_path = wget.download(lm_url, out=lm_gzip_path)
            print("Downloaded the 3-gram language model.")
        else:
            print("Pruned .arpa.gz already exists.")

        uppercase_lm_path = os.path.join(lm_folder, "3-gram.pruned.1e-7.arpa")
        if not os.path.exists(uppercase_lm_path):
            with gzip.open(lm_gzip_path, "rb") as f_zipped:
                with open(uppercase_lm_path, "wb") as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            print("Unzipped the 3-gram language model.")
        else:
            print("Unzipped .arpa already exists.")

        lm_path = os.path.join(lm_folder, "3-gram.lowercase.pruned.1e-7.arpa")
        if not os.path.exists(lm_path):
            with open(uppercase_lm_path, "r") as f_upper:
                with open(lm_path, "w") as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower().replace("'", ""))
        print("Converted language model file to lowercase.")

        vocab_path = os.path.join(lm_folder, "librispeech-vocab.txt")
        if not os.path.exists(vocab_path):
            print("Downloading librispeech vocab.")
            vocab_url = "http://www.openslr.org/resources/11/librispeech-vocab.txt"
            wget.download(vocab_url, out=vocab_path)

        with open(vocab_path) as f:
            unigram_list = [t.lower() for t in f.read().strip().split("\n")]

        return lm_path, unigram_list

    def ctc_decode(self, inds: List[int]) -> str:
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()
        chars = []
        for ind in inds:
            if len(chars) and chars[-1] == self.ind2char[ind]:
                continue
            if len(chars) and chars[-1] == self.EMPTY_TOK:
                chars.pop()
            chars.append(self.ind2char[ind])

        if chars[-1] == self.EMPTY_TOK:
            chars.pop()

        return "".join(chars)

    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        if isinstance(probs, torch.Tensor):
            probs = probs.numpy()

        probs = np.log(np.clip(probs, 1e-15, 1))
        beams = self.ctc_decoder.decode_beams(probs, beam_width=beam_size, token_min_logp=-15.)
        hypos = []
        for beam in beams:
            hypos.append((
                beam[0],  # hypothesis
                beam[-1]  # log prob
            ))
        return hypos
