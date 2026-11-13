"""Average pretrained word vectors, then score categories with a neural network."""

import torch
from torch import nn

from . import MultiLabelModel
from .embedding_lr import AverageWordVectors


class EmbeddingMLP(MultiLabelModel):
    """Train a small neural network using only known labels."""

    def __init__(self, word_vectors, hidden_size=32, epochs=30, seed=0):
        self.vectorizer = AverageWordVectors(word_vectors)
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, texts, Y, mask):
        features = torch.as_tensor(self.vectorizer.transform(texts), dtype=torch.float32)
        labels = torch.as_tensor(Y, dtype=torch.float32)
        known = torch.as_tensor(mask == 1)

        if not known.any():
            raise ValueError("The training set has no known labels.")

        torch.manual_seed(self.seed)
        self.network = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, labels.shape[1]),
        )

        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        loss_function = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            raw_scores = self.network(features)

            # Select known cells first so missing labels never enter the loss.
            loss = loss_function(raw_scores[known], labels[known])
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, texts):
        features = torch.as_tensor(self.vectorizer.transform(texts), dtype=torch.float32)
        self.network.eval()

        with torch.no_grad():
            # Each category gets its own probability.
            return torch.sigmoid(self.network(features)).numpy()