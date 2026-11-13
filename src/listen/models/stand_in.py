"""Stand-in model for building the app before any real model is trained.

It follows the MultiLabelModel interface but knows nothing about law. Each word gets
a fixed random score per category, and a story's score is built from its words, so:
  - the same story always gets the same probabilities,
  - removing a word changes the scores (so occlusion highlighting can be tested),
  - very short stories get low scores (so "need more detail" can be tested).
Its outputs mean nothing. Do not report any number it produces.
"""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from . import MultiLabelModel


class StandInModel(MultiLabelModel):
    def __init__(self, n_categories: int = 20):
        self.n_categories = n_categories

    def fit(self, texts: Sequence[str], Y: np.ndarray, mask: np.ndarray) -> "StandInModel":
        # Nothing to learn. Only record how many categories the labels have.
        self.n_categories = Y.shape[1]
        return self

    def word_scores(self, word: str) -> np.ndarray:
        # Turn the word into a fixed seed, so it always gets the same random scores.
        seed = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=self.n_categories)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        scores = np.zeros((len(texts), self.n_categories))

        for row, text in enumerate(texts):
            words = re.findall(r"\b\w+\b", text.lower())
            if not words:
                continue

            # Scaled so the spread is the same for short and long stories.
            combined = np.sum([self.word_scores(w) for w in words], axis=0) / np.sqrt(len(words))

            # Fewer than 20 words pulls every score down.
            length_bonus = 2.0 * min(1.0, len(words) / 20)
            scores[row] = combined - 2.5 + length_bonus

        # Squash into probabilities between 0 and 1.
        return 1 / (1 + np.exp(-scores))