"""Near-duplicate grouping and group-aware iterative-stratified train/val/test parts.

Purpose
-------
This module turns the cleaned story texts and the label matrix into the 80/10/10
train/val/test assignment that every reported result depends on. It has two public
functions, used in this order:

1. ``near_duplicate_groups(texts, ids)`` finds stories that are reposts of each other
   and gives each family one shared group id.
2. ``assign_parts(Y, groups, ratios, seed)`` splits at the *group* level with
   iterative stratification and returns one part name per story.

The module is pure computation. It reads no files, writes no files and logs nothing.
Importing it needs only ``numpy``; ``iterstrat`` (iterative-stratification) is
imported inside ``assign_parts``, so code that only groups stories, or only imports
the module, never needs that package.

Inputs and outputs
------------------
* ``near_duplicate_groups``: cleaned texts and unique story ids, in any order; returns
  a list of group ids aligned with the input rows.
* ``assign_parts``: the (n, 20) label matrix (NaN = omitted), the group id of each row
  and the ratios from ``configs/default.yaml``; returns an array of the strings
  ``"train"``, ``"val"`` and ``"test"``, one per row, in input order.

Why near-duplicate stories are kept in one part
-----------------------------------------------
About 3% of the stories are reposts with edited openings (52 groups holding 107
stories, largest group 4, in the best-guess binary file; measured 2026-11-13 with
the default ``k`` and ``threshold``). If one copy were in train
and another in test, a model could score well on the test copy by memorising its
twin, and the reported macro-F1 would be optimistic. Putting every copy of a story in
the same part removes that leak. The price is small: stratification and the 80/10/10
sizes are computed on groups, so the story-level part sizes can differ from the exact
ratios by a few stories.

Why groups are sorted before splitting
--------------------------------------
``MultilabelStratifiedKFold`` shuffles and breaks ties by *row position*, so the same
data in a different row order can give a different split. To make the result a
function of the stories rather than of the file order, ``assign_parts`` sorts the
distinct group ids in plain string order and splits that canonical list. Group ids
are the smallest story ``_id`` of the group (see ``near_duplicate_groups``), so a
story's part depends only on the set of (group id, group labels) pairs, the ratios and
the seed. Reordering the same rows (shuffling the CSV, or loading it in another order)
gives every story the same part as before. Adding or removing stories changes the set of
groups and their labels, so it can move surviving stories to a different part; the
frozen split files are only valid for the exact story set they were made from.

Why NaN is filled with 0 here, and only here
--------------------------------------------
Iterative stratification needs a complete 0/1 label matrix; ``iterstrat`` raises
``ValueError`` on NaN. An omitted cell means "label unknown", not "negative"
(README, "Partially labeled stories"). ``assign_parts`` therefore treats an omitted
cell as 0 *for the stratification step only*, on its own private copy. It never
changes the caller's ``Y``. The consequence is that stratification balances the
labels that are present (1.0) across parts and ignores which cells are unknown, so
the share of unknown cells per category can differ between parts. Everything after
the split (models, thresholds, metrics) must keep using the original ``Y`` and its
mask. Do not copy the 0-fill anywhere else.

How the folds map to parts
--------------------------
With ``n_splits = round(1 / ratios["test"])`` (10 for 80/10/10), fold 0 is the test
part, fold 1 is the validation part and folds 2 and up form the training part. That
works because the ratios are required to be ``test == val == 1 / n_splits`` and
``train == 1 - 2 / n_splits``.

Changing the algorithm
----------------------
Frozen split files depend on every detail here: the tokenisation, ``k``, the threshold,
the sorting rule, the fold-to-part mapping and the pinned ``iterstrat`` version. If you
change any of them, bump ``GROUPING_VERSION``, regenerate the frozen splits with
``PYTHONPATH=src python scripts/make_splits.py --force`` (the project's ``make splits``
target refuses to overwrite and cannot pass ``--force``), and add an entry to
``docs/decisions.md`` so earlier and later results are not compared by mistake.
The cleaning step that produces ``texts`` also feeds the grouping, so changing it
counts as a change to this algorithm too.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

# Version of the grouping and splitting algorithm. Callers should store it with the
# frozen splits so a mismatch is detected instead of silently mixing results from two
# algorithms.
GROUPING_VERSION = 1

# Tolerance for comparing split ratios, which arrive as floats from YAML.
_RATIO_TOLERANCE = 1e-9

# Part names, in fold order: fold 0 is the test part, fold 1 the validation part and
# folds 2 and up the training part (see the module docstring).
_TEST, _VAL, _TRAIN = "test", "val", "train"


def near_duplicate_groups(
    texts: Sequence[str],
    ids: Sequence[str],
    k: int = 5,
    threshold: float = 0.8,
) -> list[str]:
    """Group stories that are near-duplicates (reposts) and return one group id per story.

    Two stories are linked when the Jaccard similarity of their sets of k-token
    shingles is at least ``threshold``. Groups are the connected components of that
    link relation, so A~B and B~C put A, B and C together even when A and C alone fall
    below the threshold.

    Tokens are ``re.findall(r"\\w+", text.lower())``. A shingle is a tuple of ``k``
    consecutive tokens, and a story's shingle set ignores repeats and order. A story
    with fewer than ``k`` tokens has an empty shingle set, so it never matches anything
    (not even an identical short story) and stays alone.

    Args:
        texts: Story texts, one per row, ALREADY CLEANED by the project's text cleaner.
            This function lower-cases and tokenises but does no other cleaning, so
            uncleaned text (for example with usernames or links left in) can change
            which stories link.
        ids: Story ids (``_id``), one per row, aligned with ``texts``. Must be unique.
            Compared as plain strings, so they should be the 24-character lowercase
            hex ids from the CSV.
        k: Shingle length in tokens. Must be at least 1.
        threshold: Minimum Jaccard similarity for a link, in (0, 1].

    Returns:
        A list of ``len(texts)`` strings aligned with the input rows. Entry ``i`` is
        the group id of row ``i``: the smallest id (plain string order) among all rows
        in that row's group. A story with no near-duplicate is its own group, so its
        entry equals its own id. The result does not depend on the order of the rows:
        each story gets the same group id however the input is shuffled.

    Raises:
        ValueError: If ``len(texts) != len(ids)`` (pass one id per text), if ``ids``
            contains a repeated id (ids must identify stories uniquely; drop or fix
            the repeats), if ``k < 1`` or if ``threshold`` is not in (0, 1].

    Example:
        >>> texts = ["a b c d e f g", "a b c d e f h", "x y z x y z x"]
        >>> near_duplicate_groups(texts, ["b", "a", "c"], k=3, threshold=0.5)
        ['a', 'a', 'c']
    """
    if len(texts) != len(ids):
        raise ValueError(
            f"texts and ids must have the same length, got {len(texts)} texts and "
            f"{len(ids)} ids; pass one id per text, in the same order."
        )
    if len(set(ids)) != len(ids):
        raise ValueError(
            "ids must be unique, but at least one id appears more than once; "
            "deduplicate the rows or use the unique '_id' column."
        )
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")

    n = len(texts)

    # One shingle set per story. A story shorter than k tokens gets an empty set,
    # which is how short stories are kept out of every comparison below.
    shingles: list[set[tuple[str, ...]]] = []
    for text in texts:
        tokens = re.findall(r"\w+", text.lower())
        shingles.append(
            {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}
        )

    # Union-find over row indices. The smaller index always becomes the root, so the
    # structure is deterministic; the final components do not depend on union order.
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # Path compression keeps later finds short.
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    # Inverted index shingle -> earlier rows containing it. Rows are visited once, in
    # order, and each row is compared only with earlier rows that share at least one
    # shingle, so every candidate pair is seen exactly once. The shared-shingle count
    # per pair gives the exact Jaccard; common shingles are not capped or skipped,
    # because dropping them could break a true near-duplicate link.
    index: dict[tuple[str, ...], list[int]] = {}
    for i in range(n):
        shingle_set = shingles[i]
        if not shingle_set:
            continue
        shared_with: Counter[int] = Counter()
        for shingle in shingle_set:
            earlier_rows = index.get(shingle)
            if earlier_rows:
                shared_with.update(earlier_rows)
            else:
                index[shingle] = earlier_rows = []
            earlier_rows.append(i)
        size_i = len(shingle_set)
        for j, shared in shared_with.items():
            union_size = size_i + len(shingles[j]) - shared
            if shared / union_size >= threshold:
                union(j, i)

    # A group id is the smallest id in the component, not the id of the union-find
    # root: the root is the smallest ROW index, which need not hold the smallest id.
    smallest_id: dict[int, str] = {}
    for i in range(n):
        root = find(i)
        if root not in smallest_id or ids[i] < smallest_id[root]:
            smallest_id[root] = ids[i]
    return [smallest_id[find(i)] for i in range(n)]


def assign_parts(
    Y: np.ndarray,
    groups: Sequence[str],
    ratios: Mapping[str, float],
    seed: int,
) -> np.ndarray:
    """Assign every story to "train", "val" or "test" by iterative stratification on groups.

    Whole groups (see ``near_duplicate_groups``) are assigned together, so no
    near-duplicate crosses parts. Groups are stratified by the union of their members'
    labels. Omitted (NaN) cells count as 0 for this step only; ``Y`` is not modified.
    The result for a story does not depend on the order of the input rows, and the same
    inputs and seed always give the same output.

    Args:
        Y: Label matrix of shape ``(n_stories, n_categories)``, normally ``(n, 20)``.
            Cells are 0.0, 1.0 or NaN (omitted, meaning unknown). Any value above 0
            counts as a positive label; NaN and values at or below 0 count as 0.
        groups: ``n_stories`` group ids aligned with the rows of ``Y``, normally the
            output of ``near_duplicate_groups``. Stories with equal ids stay together.
            Ids are compared and sorted as plain strings.
        ratios: Mapping with exactly the keys ``"train"``, ``"val"`` and ``"test"``,
            for example ``{"train": 0.8, "val": 0.1, "test": 0.1}``. Let
            ``n_splits = round(1 / ratios["test"])``. Required: ``val == test`` and
            ``train == 1 - 2 / n_splits`` and ``train + val + test == 1``, each within
            1e-9, with ``n_splits >= 3``.
        seed: Integer seed for the shuffle inside the stratifier. Use the values in
            ``configs/default.yaml``. A different seed gives a different split.

    Returns:
        A numpy array of dtype ``<U5`` and shape ``(n_stories,)`` holding the strings
        ``"train"``, ``"val"`` or ``"test"``, in the same order as the rows of ``Y``.
        Fold 0 of the stratified k-fold is the test part, fold 1 is val and the other
        folds are train. Part sizes follow the ratios at the group level, so story
        counts can be off by a few stories when groups hold more than one story.

    Raises:
        ValueError: If ``ratios`` does not have exactly the keys train, val and test
            (fix the config), if ``ratios["test"]`` is not positive, if
            ``ratios["val"]`` differs from ``ratios["test"]`` (the fold scheme gives
            val and test the same size; set them equal), if ``ratios["train"]`` is not
            ``1 - 2 / n_splits`` (for example 0.8 for a 0.1 test ratio), if the
            ratios do not sum to 1, if ``n_splits`` is below 3 (use a test ratio of at
            most 1/3), if ``Y`` is not 2-D, if ``len(groups) != len(Y)``, if ``seed``
            is not an integer, or if there are fewer distinct groups than
            ``n_splits`` (supply more stories or a larger test ratio).
        ImportError: If the pinned ``iterative-stratification`` package is missing
            (install it with ``pip install -r requirements.txt``).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> Y = (rng.random((40, 3)) < 0.4).astype(float)
        >>> Y[rng.random(Y.shape) < 0.5] = np.nan  # omitted cells are allowed
        >>> groups = [f"g{i:02d}" for i in range(40)]
        >>> parts = assign_parts(Y, groups, {"train": 0.8, "val": 0.1, "test": 0.1}, 0)
        >>> sorted(set(parts.tolist())), int((parts == "test").sum())
        (['test', 'train', 'val'], 4)
    """
    n_splits = _validate_ratios(ratios)

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(
            f"Y must be 2-D (n_stories, n_categories), got {Y.ndim}-D with shape "
            f"{Y.shape}."
        )
    if len(groups) != Y.shape[0]:
        raise ValueError(
            f"groups must have one entry per row of Y, got {len(groups)} groups for "
            f"{Y.shape[0]} rows."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(
            f"seed must be an integer, got {type(seed).__name__}; a missing seed would "
            "make the split non-reproducible."
        )

    # np.unique sorts the distinct ids in plain string order. That canonical order is
    # what makes the result independent of the input row order. `row_group` maps each
    # story to its position in that order, and is also used to expand the result back.
    group_ids, row_group = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    row_group = row_group.reshape(-1)
    n_groups = len(group_ids)
    if n_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} distinct groups for {n_splits}-fold "
            f"stratification, got {n_groups}; supply more stories or a larger test "
            "ratio."
        )

    # Omitted cells become 0 here and only here (iterstrat rejects NaN). This builds a
    # new array, so the caller's Y keeps its NaNs. A group is positive for a category
    # when any member is, so one stratification row stands for the whole group.
    positives = (np.nan_to_num(Y, nan=0.0) > 0).astype(np.int8)
    group_labels = np.zeros((n_groups, Y.shape[1]), dtype=np.int8)
    np.maximum.at(group_labels, row_group, positives)

    # Imported here so that importing this module never requires iterstrat.
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(seed)
    )
    # Each group lands in exactly one fold's "test" indices; record that fold number.
    fold_of_group = np.full(n_groups, -1, dtype=int)
    for fold, (_, fold_groups) in enumerate(
        splitter.split(np.zeros(n_groups), group_labels)
    ):
        fold_of_group[fold_groups] = fold

    part_of_group = np.full(n_groups, _TRAIN, dtype="<U5")
    part_of_group[fold_of_group == 0] = _TEST
    part_of_group[fold_of_group == 1] = _VAL
    return part_of_group[row_group]


def _validate_ratios(ratios: Mapping[str, float]) -> int:
    """Check the split ratios and return ``n_splits``, the number of folds to use.

    Args:
        ratios: The mapping passed to ``assign_parts``.

    Returns:
        ``round(1 / ratios["test"])``, an integer of at least 3.

    Raises:
        ValueError: With a message naming the offending ratio and how to fix it.
    """
    if set(ratios) != {_TRAIN, _VAL, _TEST}:
        raise ValueError(
            f"ratios must have exactly the keys 'train', 'val' and 'test', got "
            f"{sorted(ratios)}; fix the 'splits' section of configs/default.yaml."
        )
    train, val, test = (float(ratios[name]) for name in (_TRAIN, _VAL, _TEST))
    if test <= 0:
        raise ValueError(
            f"ratios['test'] must be positive, got {test}; use for example 0.1."
        )
    n_splits = round(1 / test)
    if n_splits < 3:
        raise ValueError(
            f"ratios['test'] = {test} gives {n_splits} folds, but val, test and a "
            "non-empty train part need at least 3; use a test ratio of at most 1/3."
        )
    if abs(val - test) > _RATIO_TOLERANCE:
        raise ValueError(
            f"ratios['val'] ({val}) must equal ratios['test'] ({test}): the fold "
            "scheme gives val and test one fold each. Set the two equal."
        )
    expected_train = 1 - 2 / n_splits
    if abs(train - expected_train) > _RATIO_TOLERANCE:
        raise ValueError(
            f"ratios['train'] ({train}) must be 1 - 2/n_splits = {expected_train} for "
            f"n_splits = round(1 / test) = {n_splits}; change train or test so that "
            "they match (for example 0.8 / 0.1 / 0.1)."
        )
    if abs(train + val + test - 1.0) > _RATIO_TOLERANCE:
        raise ValueError(
            f"ratios must sum to 1, got {train + val + test}; with n_splits = "
            f"{n_splits} the test and val ratios must each be exactly 1/{n_splits}."
        )
    return n_splits
