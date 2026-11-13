"""Per-category threshold tuning.

Not implemented yet. One threshold per category, tuned on the validation split to
maximize F1, using only the rows where that category's label is known. The test
set must never be used here.
"""

import numpy as np


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a (20,) array of thresholds, one per category, chosen on validation data."""
    raise NotImplementedError


def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Turn (n_stories, 20) probabilities into 0/1 predictions."""
    raise NotImplementedError
