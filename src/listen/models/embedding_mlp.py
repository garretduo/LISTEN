"""Model 5 (the proposed model): averaged pretrained word embeddings with an MLP.

Differs from Model 4 (listen.models.embedding_lr) in exactly one component: the
classifier. The features must stay the same averaged embeddings. The feedforward
network's loss counts only labeled cells:

    L_i = -sum_c m_ic * [y_ic * log p_ic + (1 - y_ic) * log(1 - p_ic)]

where m_ic is 1 when story i has a label for category c and 0 when it was omitted.
"""

from . import MultiLabelModel


class EmbeddingMLP(MultiLabelModel):
    """Not implemented yet. Inherits fit and predict_proba, which raise NotImplementedError."""
