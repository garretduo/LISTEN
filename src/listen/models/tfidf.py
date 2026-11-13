"""Model 3: TF-IDF features with one-vs-rest logistic regression (the main baseline).

Unigram and bigram TF-IDF features. Each category's logistic regression is fit
only on training rows where that category's label is known (mask == 1).
"""

from . import MultiLabelModel


class TfidfLogReg(MultiLabelModel):
    """Not implemented yet. Inherits fit and predict_proba, which raise NotImplementedError."""
