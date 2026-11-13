"""Tests of the committed frozen splits in results/splits/ (and, where possible, the data).

The frozen files are written by `make splits` and committed to git, so these tests
guard what every teammate and every reported result depends on.

* Part (a) reads only the frozen files, so it runs on a fresh clone without `data/`.
* Part (b) also needs the real CSV (`make data`), because it checks the split against
  the labels. It asserts on counts, ids and shapes only and never prints story text.
* Part (c) recomputes the split and compares it with the files. Recomputing depends on
  the library versions, so it skips unless they match the ones recorded in the files.

Each test skips (with the command to run) when what it needs is absent.
"""

import csv
import re
from importlib import metadata

import numpy as np
import pytest

from listen import data

PARTS = ("train", "val", "test")
SIZE_TOLERANCE = 15  # stories; group-level splitting makes part sizes inexact


def ids_in_part(frozen, seed, name):
    """The ids that the frozen files put in part `name` for `seed`."""
    return [i for i, part in zip(frozen.ids, frozen.parts[seed]) if part == name]


# --------------------------------------------------------------------------------
# (a) The committed files on their own
# --------------------------------------------------------------------------------

def test_frozen_seeds_equal_config_seeds(config, frozen):
    assert frozen.meta["seeds"] == config["seeds"]
    assert list(frozen.parts) == config["seeds"]  # the CSV's seed_<n> columns, in order


def test_frozen_meta_describes_the_files(config, frozen):
    meta = frozen.meta
    assert meta["data_set"] == config["data"]["default_set"]
    assert meta["file"] == config["data"]["files"][meta["data_set"]]["name"]
    assert meta["n_stories"] == len(frozen.ids) == config["data"]["files"][meta["data_set"]]["rows"]
    assert len(meta["categories"]) == 20
    assert meta["ratios"] == {name: config["splits"][name] for name in PARTS}
    assert len(meta["sha256"]) == 64
    assert frozen.ids == sorted(frozen.ids) and len(set(frozen.ids)) == len(frozen.ids)


def test_every_seed_puts_every_story_in_exactly_one_part(frozen):
    everything = set(frozen.ids)
    for seed, labels in frozen.parts.items():
        assert len(labels) == len(frozen.ids)
        assert set(labels) <= set(PARTS)
        by_part = {name: set(ids_in_part(frozen, seed, name)) for name in PARTS}
        assert by_part["train"].isdisjoint(by_part["val"])
        assert by_part["train"].isdisjoint(by_part["test"])
        assert by_part["val"].isdisjoint(by_part["test"])
        assert by_part["train"] | by_part["val"] | by_part["test"] == everything


def test_no_near_duplicate_group_is_split_across_parts(frozen):
    groups = frozen.meta["grouping"]["groups"]
    assert all(len(group) > 1 for group in groups)
    row_of = {story_id: i for i, story_id in enumerate(frozen.ids)}
    for seed, labels in frozen.parts.items():
        for group in groups:
            assert len({labels[row_of[story_id]] for story_id in group}) == 1, (
                f"seed {seed}: a group of {len(group)} stories is split across parts"
            )


def test_different_seeds_give_different_splits(frozen):
    seeds = list(frozen.parts)
    for i, first in enumerate(seeds):
        for second in seeds[i + 1:]:
            assert frozen.parts[first] != frozen.parts[second]
            assert ids_in_part(frozen, first, "test") != ids_in_part(frozen, second, "test")


def test_part_sizes_are_close_to_80_10_10(config, frozen):
    n = frozen.meta["n_stories"]
    expected = {name: round(config["splits"][name] * n) for name in PARTS}  # 2767/346/346 for 3,459
    for seed, labels in frozen.parts.items():
        for name in PARTS:
            size = labels.count(name)
            assert abs(size - expected[name]) <= SIZE_TOLERANCE, (
                f"seed {seed}: {name} has {size} stories, expected about {expected[name]}"
            )


# --------------------------------------------------------------------------------
# (b) The frozen split against the real data
# --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def splits(config, dataset, frozen):
    """get_split for every seed, computed once. Skips if the data or the files are absent."""
    return {seed: data.get_split(seed, config) for seed in config["seeds"]}


def test_get_split_ids_equal_the_frozen_ids(dataset, frozen, splits):
    assert dataset.ids == frozen.ids
    for seed, split in splits.items():
        for name in PARTS:
            assert getattr(split, name).ids == ids_in_part(frozen, seed, name)


def test_masks_equal_the_label_mask_and_y_is_zero_where_unknown(dataset, splits):
    row_of = {story_id: i for i, story_id in enumerate(dataset.ids)}
    for seed, split in splits.items():
        for name in PARTS:
            part = getattr(split, name)
            rows = [row_of[story_id] for story_id in part.ids]
            np.testing.assert_array_equal(part.mask, data.label_mask(dataset.Y[rows]))
            assert part.Y.dtype == np.int8 and part.mask.dtype == np.int8
            assert part.Y.shape == part.mask.shape == (len(rows), 20)
            assert (part.Y[part.mask == 0] == 0).all()
            known = part.mask == 1
            np.testing.assert_array_equal(part.Y[known], dataset.Y[rows][known].astype(np.int8))
            assert part.texts == [dataset.texts[r] for r in rows]


def test_categories_equal_the_csv_header(config, dataset, frozen, splits):
    csv.field_size_limit(2**31 - 1)
    with open(data.data_path(config), newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert len(header) == 22
    assert dataset.categories == header[2:] == frozen.meta["categories"]
    assert all(split.categories == header[2:] for split in splits.values())


def test_every_category_with_a_positive_has_one_in_train(dataset, splits):
    positives = (dataset.Y == 1.0).sum(axis=0)  # labeled positives per category; NaN counts as no
    for seed, split in splits.items():
        in_train = split.train.Y.sum(axis=0)  # Y is 0 where unknown, so this counts labeled positives
        lacking = [c for c in range(20) if positives[c] > 0 and in_train[c] == 0]
        assert not lacking, f"seed {seed}: no labeled positive in train for categories {lacking}"


def test_every_category_with_ten_positives_is_in_every_part(dataset, splits):
    positives = (dataset.Y == 1.0).sum(axis=0)
    for seed, split in splits.items():
        for name in PARTS:
            in_part = getattr(split, name).Y.sum(axis=0)
            lacking = [c for c in range(20) if positives[c] >= 10 and in_part[c] == 0]
            assert not lacking, f"seed {seed}: no labeled positive in {name} for categories {lacking}"


def test_cleaned_texts_have_no_links_or_user_names(dataset):
    patterns = {
        "link": re.compile(r"https?://"),
        "www": re.compile(r"www\."),
        "user": re.compile(r"(?<![\w/])/?[ur]/\w"),
    }
    # Counts only: the offending text must never appear in a failure message.
    counts = {
        name: sum(1 for text in dataset.texts if pattern.search(text))
        for name, pattern in patterns.items()
    }
    assert counts == {"link": 0, "www": 0, "user": 0}, f"stories with a leftover pattern: {counts}"


# --------------------------------------------------------------------------------
# (c) Drift: does recomputing still give the committed files?
# --------------------------------------------------------------------------------

def test_frozen_splits_match_a_fresh_computation(config, dataset, frozen):
    pytest.importorskip("iterstrat")
    # Only the three libraries that can change a split are compared. The Python patch
    # version cannot change the result and would make this test skip for no reason.
    recorded = frozen.meta["versions"]
    libraries = ("numpy", "scikit-learn", "iterative-stratification")
    installed = {name: metadata.version(name) for name in libraries}
    differing = {
        name: (recorded.get(name), version)
        for name, version in installed.items()
        if recorded.get(name) != version
    }
    if differing:
        pytest.skip(f"library versions differ from the frozen ones (frozen, installed): {differing}")
    assert data.check_frozen_splits(config) == []
