"""Average pretrained word vectors, then fit logistic regression per category."""

import re

import numpy as np

from .tfidf import TfidfLogReg


class AverageWordVectors:
    """Turn each story into the average of its known word vectors."""

    def __init__(self, word_vectors):
        if not word_vectors:
            raise ValueError("Provide at least one word vector.")

        self.word_vectors = word_vectors
        self.vector_size = len(next(iter(word_vectors.values())))

    def fit_transform(self, texts):
        # The pretrained vectors are already fixed.
        return self.transform(texts)

    def transform(self, texts):
        features = np.zeros((len(texts), self.vector_size))

        for row, story in enumerate(texts):
            vectors = []
            for word in re.findall(r"\b\w+\b", story.lower()):
                if word in self.word_vectors:
                    vectors.append(self.word_vectors[word])

            # A story with no recognized words keeps its zero vector.
            if vectors:
                features[row] = np.mean(vectors, axis=0)

        return features


class EmbeddingLogReg(TfidfLogReg):
    """Use the same logistic regression with averaged word vectors."""

    def __init__(self, word_vectors):
        self.word_vectors = word_vectors

    def make_vectorizer(self):
        return AverageWordVectors(self.word_vectors)