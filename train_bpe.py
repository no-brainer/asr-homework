import argparse
import os
import tempfile

import youtokentome as yttm


def normalize_text(text):
    # librispeech has some codes as first word in each line
    text_partition = text.split(maxsplit=1)
    if len(text_partition) > 1:
        text = text_partition[1]

    return text.lower().replace("'", "")


def prep_corpus(corpus_file, path):
    for dir_path, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue

            full_path = os.path.join(dir_path, filename)
            with open(full_path, "r") as f:
                for line in f:
                    corpus_file.write(normalize_text(line))


def train_bpe_tokenizer(corpus_file, vocab_size, save_file):
    yttm.BPE.train(
        data=corpus_file,
        vocab_size=vocab_size,
        model=save_file
    )


def main(args):
    with tempfile.TemporaryDirectory() as temp_dir:
        corpus_path = os.path.join(temp_dir, "corpus.txt")
        with open(corpus_path, "w") as corpus_file:
            prep_corpus(corpus_file, args.path)

        train_bpe_tokenizer(
            corpus_path,
            args.vocab_size,
            args.out
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Prepare BPE tokenizer")
    args.add_argument(
        "-p",
        "--path",
        default="./data/datasets/librispeech/train-clean-100",
        type=str,
        help="path to dataset root"
    )
    args.add_argument(
        "-o",
        "--out",
        default="./hw_asr/language_models/bpe.model",
        type=str,
        help="path to saved bpe tokenizer"
    )
    args.add_argument(
        "-s",
        "--vocab_size",
        default=500,
        type=int,
        help="amount of tokens used by bpe"
    )

    args = args.parse_args()
    main(args)
