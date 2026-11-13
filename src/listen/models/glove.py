"""Load pretrained GloVe word vectors from a local text file."""

import numpy as np


def load_glove(path):
    word_vectors = {}

    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue

            # Each line starts with a word, followed by its vector.
            word = parts[0]
            word_vectors[word] = np.array(parts[1:], dtype=np.float32)

    return word_vectors