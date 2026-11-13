"""Tests for listen.evaluate and listen.thresholds on small hand-checked examples."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from listen.data import load_config
from listen.evaluate import bootstrap_ci, macro_micro_f1, per_category_prf, recall_at_k
from listen.thresholds import apply_thresholds, tune_thresholds

ROOT = Path(__file__).resolve().parent.parent


def test_prf_matches_sklearn_on_known_cells():
    rng = np.random.default_rng(0)
    y, p, m = (rng.integers(0, 2, (200, 3)) for _ in range(3))
    got = per_category_prf(y, p, m)
    for c in range(3):
        k = m[:, c] == 1
        want = precision_recall_fscore_support(y[k, c], p[k, c], average="binary")[:3]
        np.testing.assert_allclose([g[c] for g in got], want)


def test_macro_micro_and_mask():
    y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
    p = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
    ones = np.ones_like(y)
    macro, micro = macro_micro_f1(y, p, ones)
    assert (macro, micro) == (0.75, 4 / 6)  # category F1s are 0.5 and 1.0
    hidden = ones.copy()
    hidden[1:3, 0] = 0  # the two missed positives are unknown, so they cannot count
    assert macro_micro_f1(y, p, hidden) == (1.0, 1.0)


def test_f1_is_nan_without_positives_and_macro_skips_it():
    y = np.array([[1, 0], [0, 0]])
    p = np.array([[1, 1], [0, 0]])  # a false alarm in the all-negative category
    ones = np.ones_like(y)
    assert np.isnan(per_category_prf(y, p, ones)[2][1])
    assert macro_micro_f1(y, p, ones)[0] == 1.0


def test_recall_at_k():
    y = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]])
    prob = np.array([[0.9, 0.1, 0.8, 0.7]] * 3)
    mask = np.ones_like(y)
    mask[2, 1] = 0  # row 2: only category 0 is a known positive
    # Row 0 finds 1 of 2, row 1 has no positives (ignored), row 2 finds 1 of 1.
    assert recall_at_k(y, prob, mask, k=2) == 0.75


def test_thresholds():
    y = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]])
    prob = np.array([[0.9, 0.5, 0.4], [0.8, 0.5, 0.4], [0.3, 0.5, 0.4], [0.2, 0.5, 0.4], [0.9, 0.5, 0.4]])
    mask = np.ones_like(y)
    mask[4, 0] = 0  # an unknown cell with a high score must not influence the cut
    t = tune_thresholds(y, prob, mask)
    assert t[0] == 0.8  # best F1 predicts the top two scores
    assert t[1] == 0.5  # no positives: fallback
    assert t[2] == 0.4  # constant scores: predict yes for every story
    assert apply_thresholds(prob, t)[:, 2].all()
    assert apply_thresholds(prob, t)[:, 0].tolist() == [1, 1, 0, 0, 1]


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, (300, 4)), rng.integers(0, 2, (300, 4))
    m = np.ones_like(y)
    metric = lambda t, q, k: macro_micro_f1(t, q, k)[0]
    low, high = bootstrap_ci(metric, y, p, m, seed=0)
    assert (low, high) == bootstrap_ci(metric, y, p, m, seed=0)
    assert low <= metric(y, p, m) <= high


def test_excluded_categories_are_real_columns():
    categories = json.loads((ROOT / "results/splits/best_guess_binary.meta.json").read_text())["categories"]
    excluded = load_config()["evaluation"]["excluded_categories"]
    assert set(excluded) <= set(categories) and len(categories) - len(excluded) == 16
