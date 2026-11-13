"""Model 4: averaged pretrained word embeddings with logistic regression.

Differs from Model 3 (listen.models.tfidf) in exactly one component: the feature
representation. The classifier must stay the same one-vs-rest logistic regression,
fit per category only on rows where the label is known. The embedding source is
not chosen yet.
"""

from . import MultiLabelModel


class EmbeddingLogReg(MultiLabelModel):
    """Not implemented yet. Inherits fit and predict_proba, which raise NotImplementedError."""
