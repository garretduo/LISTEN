"""Per-category threshold tuning.
"""

import numpy as np


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a (n_categories,) array of thresholds chosen on validation data.

    Because each threshold is one of the category's observed scores, apply_thresholds (which
    predicts "yes" at score >= threshold) reproduces the F1-maximizing cut
    """
    thresholds = np.full(y_prob.shape[1], 0.5)
    for c in range(y_prob.shape[1]):
        known = mask[:, c] == 1
        y, p = y_true[known, c], y_prob[known, c]
        if y.sum() == 0:
            continue
        order = np.argsort(-p, kind="stable")
        y, p = y[order], p[order]
        # Predicting the top k stories gives F1 = 2*TP / (k + number of positives).
        f1 = 2 * np.cumsum(y) / (np.arange(1, len(p) + 1) + y.sum())
        f1[:-1][p[:-1] == p[1:]] = -1  # cut only between different scores
        thresholds[c] = p[np.argmax(f1)]
    return thresholds


def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Turn (n_stories, n_categories) probabilities into 0/1 predictions (score >= threshold)."""
    return (y_prob >= thresholds).astype(np.int8)
