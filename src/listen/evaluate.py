"""Metrics and bootstrap intervals.

Notes: 
1. Per-label accuracy is not reported, because most categories are absent from most stories and predicting nothing scores high. 
2. Every metric counts only cells where the label is known (mask == 1). 
3. A category's recall and F1 are NaN ("n/a") when the part has no known positive story for it.
4. Macro-F1 averages the 16 categories where F1 is defined.
"""

import numpy as np


def _ratio(a, b):
    """a / b, NaN where b is 0."""
    return np.where(b > 0, a / np.maximum(b, 1), np.nan)


def _counts(y_true, y_pred, mask):
    """True positives, false positives and false negatives per category, over known cells."""
    known = mask == 1
    return [((y_true == t) & (y_pred == p) & known).sum(0) for t, p in ((1, 1), (0, 1), (1, 0))]


def per_category_prf(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Return (precision, recall, f1), each a (n_categories,) array.

    Precision is NaN when nothing was predicted for the category; recall and F1 are NaN
    when it has no known positive story.
    """
    tp, fp, fn = _counts(y_true, y_pred, mask)
    f1 = np.where(tp + fn > 0, _ratio(2 * tp, 2 * tp + fp + fn), np.nan)
    return _ratio(tp, tp + fp), _ratio(tp, tp + fn), f1


def macro_micro_f1(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Return (macro_f1, micro_f1) over the columns given.

    Macro averages the per-category F1 values that are defined. Micro pools all known
    cells. Pass only the scored columns (everything but `evaluation.excluded_categories`).
    """
    tp, fp, fn = _counts(y_true, y_pred, mask)
    macro = np.nanmean(per_category_prf(y_true, y_pred, mask)[2])
    return float(macro), float(_ratio(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum()))


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray, k: int = 3):
    """Recall at k: share of a story's known positive categories among its k top-scored ones.

    Ranks all categories, averages over stories with at least one known positive, and
    breaks ties by category order so repeated runs agree.
    """
    positive = (y_true == 1) & (mask == 1)
    top = np.argsort(-y_prob, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(positive, top, axis=1).sum(1)
    total = positive.sum(1)
    return float(np.mean(hits[total > 0] / total[total > 0]))


def bootstrap_ci(metric, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, seed: int,
                 n_boot: int = 1000):
    """95% percentile bootstrap interval (low, high), resampling stories.

    metric(y_true, y_pred, mask) returns a float; pass probabilities as y_pred for
    recall_at_k. A resample where the metric is NaN is ignored.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        values.append(metric(y_true[rows], y_pred[rows], mask[rows]))
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)
