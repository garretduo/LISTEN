"""TF-IDF features with one logistic regression per legal category."""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import MultiLabelModel


class TfidfLogReg(MultiLabelModel):
    def fit(self, texts: Sequence[str], Y: np.ndarray, mask: np.ndarray) -> "TfidfLogReg":
        # Learn the vocabulary and word weights from the training stories.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        features = self.vectorizer.fit_transform(texts)
        self.classifiers = []

        for category in range(Y.shape[1]):
            # A story can have a known label for one category and an unknown label for another.
            known = mask[:, category] == 1
            labels = Y[known, category]

            if labels.size == 0:
                raise ValueError(f"Category column {category} has no known training labels.")

            if np.unique(labels).size == 1:
                # Logistic regression needs both classes. Use the observed label if only one exists.
                self.classifiers.append(float(labels[0]))
            else:
                classifier = LogisticRegression(max_iter=1000)
                classifier.fit(features[known], labels)
                self.classifiers.append(classifier)

        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        scores = np.empty((features.shape[0], len(self.classifiers)))

        for category, classifier in enumerate(self.classifiers):
            if isinstance(classifier, float):
                scores[:, category] = classifier
            else:
                # Column 1 is the probability that the category applies.
                scores[:, category] = classifier.predict_proba(features)[:, 1]

        return scores