import os
from typing import List, Tuple

import gdown
import torch
import youtokentome as yttm

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class BPECharTextEncoder(CTCCharTextEncoder):

    def __init__(self):
        model_path = "./hw_asr/language_models/bpe.model"
        if not os.path.exists(model_path):
            print("Downloading BPE model")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdrive_id = "19Or72sk4S_33hwZBhig92-erHYsfUZJi"
            gdown.download(id=gdrive_id, output=model_path)

        self.bpe = yttm.BPE(model=model_path)

        alphabet = self.bpe.vocab()[1:]  # first token is <PAD>, we repurpose it
        super().__init__(alphabet)

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return torch.IntTensor(self.bpe.encode([text], output_type=yttm.OutputType.ID))

    def ctc_decode(self, inds: List[int]) -> str:
        # first we filter ctc tokens
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()

        new_inds = []
        for ind in inds:
            if len(new_inds) and new_inds[-1] == ind:
                continue
            if len(new_inds) and new_inds[-1] == self.char2ind[self.EMPTY_TOK]:
                new_inds.pop()
            new_inds.append(ind)

        if new_inds[-1] == self.char2ind[self.EMPTY_TOK]:
            new_inds.pop()

        return self.bpe.decode(new_inds)[0]

    def ctc_beam_search(self, probs: torch.Tensor,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        raise NotImplemented("TODO")
