"""Model interface shared by all five models (baselines included).

Every model outputs one probability per category. A per-category threshold, tuned
in listen.thresholds, turns those probabilities into labels.
"""

from collections.abc import Sequence

import numpy as np


class MultiLabelModel:
    """Common interface so the results driver and the app treat every model alike."""

    def fit(self, texts: Sequence[str], Y: np.ndarray, mask: np.ndarray) -> "MultiLabelModel":
        """Fit on training stories.

        Y is (n_stories, 20) with 0/1 labels. mask is (n_stories, 20) with 1 where
        the label is known and 0 where the cell was omitted. Omitted means unknown,
        not negative, so fitting must use only the cells where mask is 1.
        """
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n_stories, 20) array of per-category probabilities."""
        raise NotImplementedError
