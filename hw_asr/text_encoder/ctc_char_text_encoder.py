from typing import List, Tuple

import numpy as np
from pyctcdecode import build_ctcdecoder
import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str], *args, **kwargs):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.ctc_decoder = build_ctcdecoder(
            [""] + alphabet,  # pyctcdecoder only works with "" as blank
            kwargs.get("lm_path")
        )

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
        probs = np.log(np.clip(probs.numpy(), 1e-15, 1))
        beams = self.ctc_decoder.decode_beams(probs, beam_width=beam_size)
        hypos = []
        for beam in beams:
            hypos.append((
                beam[0],  # hypothesis
                beam[-1]  # log prob
            ))
        return hypos
