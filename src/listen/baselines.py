"""Models 1 and 2: the reference baselines."""

import numpy as np

from .models import MultiLabelModel


class ClassPriorBaseline(MultiLabelModel):
    """Score each category by how often it appears in the training set."""

    def fit(self, texts, Y, mask):
        self.category_scores = np.empty(Y.shape[1])

        for category in range(Y.shape[1]):
            # Skip stories whose label for this category is unknown.
            known = mask[:, category] == 1
            labels = Y[known, category]

            if labels.size == 0:
                raise ValueError(f"Category column {category} has no known training labels.")

            # The average of the 0s and 1s is the fraction marked yes.
            self.category_scores[category] = labels.mean()

        return self

    def predict_proba(self, texts):
        # Give every new story the same category scores.
        return np.tile(self.category_scores, (len(texts), 1))


class RuleBasedBaseline(MultiLabelModel):
    """Model 2: hand-written regular expressions built from the LIST category definitions.

    Not implemented yet.
    """