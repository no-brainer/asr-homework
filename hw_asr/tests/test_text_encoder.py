import unittest

import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    @staticmethod
    def build_perfect_text_probs(text, text_encoder):
        probs = torch.zeros((len(text), len(text_encoder.char2ind)))
        for i, char in enumerate(text):
            probs[i][text_encoder.char2ind[char]] = 1.
        return probs

    def test_beam_search(self):
        true_text = "bugz bunny"
        text_encoder = CTCCharTextEncoder(list(set(true_text)))

        bad_probs = self.build_perfect_text_probs("bun^ny bun^ny", text_encoder)
        good_probs = self.build_perfect_text_probs("bugz^^ bun^ny", text_encoder)
        probs = 0.51 * bad_probs + 0.49 * good_probs

        # without LM
        decoded_beams = text_encoder.ctc_beam_search(probs, beam_size=20)
        self.assertIn(decoded_beams[0][0], "bunny bunny")
