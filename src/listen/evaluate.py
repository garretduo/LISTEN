"""Metrics and bootstrap intervals.

Not implemented yet. Macro-F1 is the headline metric. Per-label accuracy is not
reported, because most categories are absent from most stories and predicting
nothing scores high. Every metric counts only cells where the label is known
(mask == 1).
"""

import numpy as np


def per_category_prf(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Precision, recall and F1 for each of the 20 categories."""
    raise NotImplementedError


def macro_micro_f1(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Return (macro_f1, micro_f1)."""
    raise NotImplementedError


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray, k: int = 3):
    """Recall at k: how many true labels appear among the k highest-probability categories."""
    raise NotImplementedError


def bootstrap_ci(metric, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, seed: int):
    """95% bootstrap confidence interval for metric, resampling over stories."""
    raise NotImplementedError
