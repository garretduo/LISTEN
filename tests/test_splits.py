"""Tests for listen.splits using synthetic data only.

No story text from the Learned Hands corpus is used here: every text is random words
from a made-up vocabulary and every id is random hex. The tests need only numpy,
pytest and iterative-stratification, and run in a few seconds.
"""

import doctest

import numpy as np
import pytest

from listen import splits
from listen.splits import GROUPING_VERSION, assign_parts, near_duplicate_groups

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
N_CATEGORIES = 20
VOCAB = [f"w{i}" for i in range(300)]


def _hex_ids(rng, n):
    """Return n unique 24-character lowercase hex ids."""
    ids = set()
    while len(ids) < n:
        ids.add("".join(rng.choice(list("0123456789abcdef"), size=24)))
    ids = sorted(ids)
    rng.shuffle(ids)  # So the smallest id is not tied to row position.
    return [str(i) for i in ids]


def _jaccard(text_a, text_b, k=5):
    """Exact Jaccard of k-token shingle sets, computed independently of the module."""
    def shingles(text):
        tokens = text.lower().split()
        return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}

    a, b = shingles(text_a), shingles(text_b)
    return len(a & b) / len(a | b)


def _edit(tokens, positions, tag):
    """Copy of tokens with the given positions replaced by words that appear nowhere else."""
    out = list(tokens)
    for p in positions:
        out[p] = f"{tag}x{p}"
    return out


@pytest.fixture(scope="module")
def corpus():
    """About 300 synthetic stories with 10 planted near-duplicate groups.

    Returns a dict with texts, ids, the expected group id per row, the planted groups
    as sets of ids, the ids of the short stories, and a label matrix with ~50% NaN.
    """
    rng = np.random.default_rng(12345)
    texts, planted = [], []  # planted: list of lists of row indices

    def fresh_tokens(length):
        return [str(w) for w in rng.choice(VOCAB, size=length)]

    # Ten planted groups of size 2 or 3. Every copy is derived from the original with
    # 1-2 changed words far apart, so each copy has Jaccard >= 0.8 with the original
    # (checked below) while two copies can fall under 0.8 against each other. The group
    # is then held together only by the connected-component rule.
    for g in range(10):
        original = fresh_tokens(180)
        rows = [len(texts)]
        texts.append(" ".join(original))
        for c in range(2 if g % 2 == 0 else 1):
            positions = [20 + 7 * c, 120 + 9 * c]
            rows.append(len(texts))
            texts.append(" ".join(_edit(original, positions, f"g{g}c{c}")))
        planted.append(rows)

    # Unrelated stories, plus two identical 4-token stories (below k, so they must stay
    # alone even though they are exact copies).
    short_tokens = ["alpha", "beta", "gamma", "delta"]
    short_rows = [len(texts), len(texts) + 1]
    texts += [" ".join(short_tokens)] * 2
    while len(texts) < 300:
        texts.append(" ".join(fresh_tokens(int(rng.integers(60, 200)))))

    ids = _hex_ids(rng, len(texts))
    expected = list(ids)
    for rows in planted:
        smallest = min(ids[r] for r in rows)
        for r in rows:
            expected[r] = smallest

    Y = (rng.random((len(texts), N_CATEGORIES)) < 0.15).astype(float)
    Y[rng.random(Y.shape) < 0.5] = np.nan

    return {
        "texts": texts,
        "ids": ids,
        "expected": expected,
        "planted": planted,
        "short_rows": short_rows,
        "Y": Y,
    }


def test_grouping_version_is_one():
    assert GROUPING_VERSION == 1


def test_planted_copies_are_similar_enough(corpus):
    """Guards the fixture: every copy must reach Jaccard 0.8 with its original."""
    texts = corpus["texts"]
    for rows in corpus["planted"]:
        for r in rows[1:]:
            assert _jaccard(texts[rows[0]], texts[r]) >= 0.8


def test_groups_equal_planted_groups(corpus):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    assert groups == corpus["expected"]


def test_unrelated_stories_stay_alone(corpus):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    planted_rows = {r for rows in corpus["planted"] for r in rows}
    for r, (group, story_id) in enumerate(zip(groups, corpus["ids"])):
        if r not in planted_rows:
            assert group == story_id


def test_group_id_is_smallest_id_and_not_the_first_row(corpus):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    first_row_is_smallest = 0
    for rows in corpus["planted"]:
        member_ids = [corpus["ids"][r] for r in rows]
        assert {groups[r] for r in rows} == {min(member_ids)}
        first_row_is_smallest += member_ids[0] == min(member_ids)
    # The fixture shuffles ids, so "smallest id" is not the same as "first row".
    assert first_row_is_smallest < len(corpus["planted"])


def test_story_under_k_tokens_stays_alone(corpus):
    """Two identical 4-token stories must not link: they have no shingles."""
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    for r in corpus["short_rows"]:
        assert groups[r] == corpus["ids"][r]
    assert corpus["texts"][corpus["short_rows"][0]] == corpus["texts"][corpus["short_rows"][1]]


def test_exact_k_tokens_story_can_match(corpus):
    """A story of exactly k tokens has one shingle, so identical copies link."""
    groups = near_duplicate_groups(["a b c d e", "a b c d e"], ["bb", "aa"])
    assert groups == ["aa", "aa"]


def test_case_and_punctuation_do_not_matter():
    texts = ["One two, three! four five six seven", "one TWO three four five six seven"]
    assert near_duplicate_groups(texts, ["y", "x"]) == ["x", "x"]


def test_threshold_boundary_is_inclusive():
    # With k=1 the shingle sets are {a,b,c,d} and {a,b,c,d,e}: Jaccard is exactly 0.8.
    texts = ["a b c d", "a b c d e"]
    assert near_duplicate_groups(texts, ["x", "w"], k=1, threshold=0.8) == ["w", "w"]
    assert near_duplicate_groups(texts, ["x", "w"], k=1, threshold=0.81) == ["x", "w"]


def test_dissimilar_copy_is_not_linked():
    rng = np.random.default_rng(7)
    original = [str(w) for w in rng.choice(VOCAB, size=100)]
    heavily_edited = _edit(original, range(5, 100, 10), "zz")  # 10 changed words
    texts = [" ".join(original), " ".join(heavily_edited)]
    assert _jaccard(*texts) < 0.8
    assert near_duplicate_groups(texts, ["b", "a"]) == ["b", "a"]


def test_grouping_is_independent_of_row_order(corpus):
    texts, ids = corpus["texts"], corpus["ids"]
    baseline = dict(zip(ids, near_duplicate_groups(texts, ids)))
    order = np.random.default_rng(99).permutation(len(texts))
    shuffled_texts = [texts[i] for i in order]
    shuffled_ids = [ids[i] for i in order]
    shuffled = near_duplicate_groups(shuffled_texts, shuffled_ids)
    assert dict(zip(shuffled_ids, shuffled)) == baseline


def test_grouping_empty_input():
    assert near_duplicate_groups([], []) == []


def test_grouping_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        near_duplicate_groups(["a b c d e"], ["x", "y"])


def test_grouping_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        near_duplicate_groups(["a b c d e", "f g h i j"], ["x", "x"])


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"threshold": 0.0}, {"threshold": 1.5}])
def test_grouping_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        near_duplicate_groups(["a b c d e"], ["x"], **kwargs)


def _synthetic_groups(n, rng, n_multi=100):
    """Group ids for n stories: n_multi groups of 2-3 stories, the rest singletons."""
    ids = _hex_ids(rng, n)
    groups = list(ids)
    row = 0
    for _ in range(n_multi):
        size = int(rng.integers(2, 4))
        for r in range(row, row + size):
            groups[r] = ids[row]
        row += size
    return ids, groups


def _assert_groups_whole(parts, groups):
    for g in set(groups):
        member_parts = {p for p, gg in zip(parts, groups) if gg == g}
        assert len(member_parts) == 1


def test_parts_are_valid_and_keep_groups_whole(corpus):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    parts = assign_parts(corpus["Y"], groups, RATIOS, seed=0)
    assert parts.shape == (len(groups),)
    assert set(parts.tolist()) == {"train", "val", "test"}  # Disjoint by construction.
    _assert_groups_whole(parts, groups)
    # Planted groups are the point of the whole exercise: assert them explicitly.
    for rows in corpus["planted"]:
        assert len({parts[r] for r in rows}) == 1


def test_part_sizes_on_a_large_set():
    rng = np.random.default_rng(2024)
    n = 3000
    _, groups = _synthetic_groups(n, rng)
    Y = (rng.random((n, N_CATEGORIES)) < rng.uniform(0.01, 0.3, N_CATEGORIES)).astype(float)
    Y[rng.random(Y.shape) < 0.5] = np.nan
    for seed in range(2):
        parts = assign_parts(Y, groups, RATIOS, seed=seed)
        _assert_groups_whole(parts, groups)
        for name, share in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
            assert abs(int((parts == name).sum()) - round(share * n)) <= 15


def test_same_seed_is_identical_and_seeds_differ(corpus):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    first = assign_parts(corpus["Y"], groups, RATIOS, seed=0)
    again = assign_parts(corpus["Y"], groups, RATIOS, seed=0)
    other = assign_parts(corpus["Y"], groups, RATIOS, seed=1)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_parts_are_independent_of_row_order(corpus):
    ids, Y = corpus["ids"], corpus["Y"]
    groups = near_duplicate_groups(corpus["texts"], ids)
    baseline = dict(zip(ids, assign_parts(Y, groups, RATIOS, seed=3)))
    for shuffle_seed in (0, 1):
        order = np.random.default_rng(shuffle_seed).permutation(len(ids))
        parts = assign_parts(Y[order], [groups[i] for i in order], RATIOS, seed=3)
        assert dict(zip([ids[i] for i in order], parts)) == baseline


def test_nan_is_treated_as_zero_and_y_is_not_modified(corpus):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    Y = corpus["Y"].copy()
    assert np.isnan(Y).any()
    with_nan = assign_parts(Y, groups, RATIOS, seed=2)
    assert np.array_equal(np.isnan(Y), np.isnan(corpus["Y"]))  # Caller's NaNs survive.
    filled = assign_parts(np.nan_to_num(Y, nan=0.0), groups, RATIOS, seed=2)
    assert np.array_equal(with_nan, filled)


def test_group_labels_are_the_union_over_members():
    """A positive on any member makes the whole group positive for stratification."""
    rng = np.random.default_rng(5)
    n = 120
    ids = _hex_ids(rng, n)
    # Groups of two: rows (0,1), (2,3), ... share a group id.
    groups = [ids[2 * (r // 2)] for r in range(n)]
    Y = (rng.random((n, N_CATEGORIES)) < 0.2).astype(float)
    # Splitting each pair's labels differently between members must not change parts
    # as long as the union is the same.
    Y_moved = Y.copy()
    Y_moved[0::2] = np.maximum(Y[0::2], Y[1::2])
    Y_moved[1::2] = 0.0
    assert np.array_equal(
        assign_parts(Y, groups, RATIOS, seed=0),
        assign_parts(Y_moved, groups, RATIOS, seed=0),
    )


@pytest.mark.parametrize(
    "ratios, message",
    [
        ({"train": 0.8, "val": 0.15, "test": 0.05}, "must equal"),  # val != test
        ({"train": 0.76, "val": 0.12, "test": 0.12}, r"1 - 2/n_splits"),  # wrong train
        ({"train": 0.8, "val": 0.105, "test": 0.105}, "sum to 1"),  # wrong sum
        ({"train": 0.8, "val": 0.1}, "exactly the keys"),  # missing key
        ({"train": 1.0, "val": 0.0, "test": 0.0}, "positive"),  # no test part
        ({"train": 0.0, "val": 0.5, "test": 0.5}, "at least 3"),  # two folds only
    ],
)
def test_bad_ratios_raise(corpus, ratios, message):
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    with pytest.raises(ValueError, match=message):
        assign_parts(corpus["Y"], groups, ratios, seed=0)


def test_other_valid_ratios_are_accepted():
    rng = np.random.default_rng(11)
    n = 200
    ids = _hex_ids(rng, n)
    Y = (rng.random((n, N_CATEGORIES)) < 0.3).astype(float)
    parts = assign_parts(Y, ids, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    # Stratified folds can differ from the exact size by a story or two.
    assert abs(int((parts == "test").sum()) - 40) <= 2
    assert abs(int((parts == "val").sum()) - 40) <= 2
    assert abs(int((parts == "train").sum()) - 120) <= 4


def test_assign_parts_rejects_bad_inputs(corpus):
    Y = corpus["Y"]
    groups = near_duplicate_groups(corpus["texts"], corpus["ids"])
    with pytest.raises(ValueError, match="one entry per row"):
        assign_parts(Y, groups[:-1], RATIOS, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        assign_parts(Y[:, 0], groups, RATIOS, seed=0)
    with pytest.raises(ValueError, match="integer"):
        assign_parts(Y, groups, RATIOS, seed=None)
    with pytest.raises(ValueError, match="distinct groups"):
        assign_parts(Y[:5], groups[:5], RATIOS, seed=0)


def test_importing_module_does_not_import_iterstrat():
    """near_duplicate_groups must work in an environment without iterative-stratification."""
    import subprocess
    import sys
    from pathlib import Path

    # The child process does not inherit pytest's pythonpath setting, so point it at
    # the directory that contains the listen package.
    src_dir = str(Path(splits.__file__).resolve().parents[1])
    code = (
        f"import sys; sys.path.insert(0, {src_dir!r}); import listen.splits as s; "
        "assert 'iterstrat' not in sys.modules; "
        "assert s.near_duplicate_groups(['a b c d e'], ['x']) == ['x']"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_docstring_examples_run():
    results = doctest.testmod(splits)
    assert results.attempted >= 2
    assert results.failed == 0
