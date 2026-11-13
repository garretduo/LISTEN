"""Tests for listen.data on synthetic data (always runnable, no data/ folder needed).

Every test builds a small random data set in pytest's tmp_path (see conftest.py), so no
story text is stored, printed or quoted here. The tests cover loading and validation,
the label mask, the index-based make_splits, and the freeze / get_split / check cycle
with its failure modes.
"""

import copy
import csv
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from listen import data
from listen.text import CLEANING_VERSION, clean_text

SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def read_csv(path):
    """Return (header, rows) of a CSV file."""
    csv.field_size_limit(2**31 - 1)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def write_csv(path, header, rows):
    """Write a CSV with the same dialect as the synthetic generator."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def edit_csv(path, change):
    """Read a CSV, let change(header, rows) modify it in place, and write it back."""
    header, rows = read_csv(path)
    change(header, rows)
    write_csv(path, header, rows)


def edit_meta(config, change):
    """Read the frozen meta JSON, let change(meta) modify it, and write it back."""
    _, meta_path = data.frozen_paths(config)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    change(meta)
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def tiny_config(synthetic, directory):
    """A copy of the synthetic config whose data folder is `directory` (for edited CSVs)."""
    config = copy.deepcopy(synthetic.config)
    config["paths"]["data_dir"] = str(directory)
    return config


@pytest.fixture
def variant(synthetic, tmp_path):
    """Return a function that writes an edited copy of the synthetic CSV and its config."""

    def make(change):
        folder = tmp_path / "variant"
        folder.mkdir(exist_ok=True)
        header, rows = read_csv(synthetic.path)
        change(header, rows)
        write_csv(folder / "synthetic.csv", header, rows)
        return tiny_config(synthetic, folder)

    return make


@pytest.fixture(scope="module")
def shared(tmp_path_factory, make_synthetic):
    """A synthetic data set that has been frozen once. Read-only for the tests."""
    base = tmp_path_factory.mktemp("shared")
    synthetic = make_synthetic(base)
    data.freeze_splits(synthetic.config)
    return SimpleNamespace(synthetic=synthetic, base=base, config=synthetic.config)


@pytest.fixture
def clone(shared, tmp_path):
    """A private, writable copy of the frozen synthetic data set, for tests that tamper."""
    destination = tmp_path / "clone"
    shutil.copytree(shared.base, destination)
    config = copy.deepcopy(shared.config)
    config["paths"]["data_dir"] = str(destination / "data")
    config["paths"]["splits_dir"] = str(destination / "splits")
    return SimpleNamespace(config=config, data=destination / "data" / "synthetic.csv")


# --------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------

def test_the_real_config_names_the_frozen_folder():
    config = data.load_config()
    assert config["paths"]["splits_dir"] == "results/splits"
    csv_path, meta_path = data.frozen_paths(config)
    assert csv_path.name == "best_guess_binary.csv"
    assert meta_path.name == "best_guess_binary.meta.json"


def test_load_dataset_matches_the_file(synthetic):
    # The generator writes rows shuffled, so sorting is a real test.
    assert synthetic.ids != sorted(synthetic.ids)
    ds = data.load_dataset(synthetic.config)
    n = len(synthetic.ids)

    assert ds.ids == sorted(synthetic.ids)
    assert len(ds.texts) == n
    assert ds.categories == synthetic.categories
    assert ds.Y.shape == (n, 20) and ds.Y.dtype == np.float64
    assert ds.mask.shape == (n, 20) and ds.mask.dtype == np.int8

    row_of = {story_id: i for i, story_id in enumerate(synthetic.ids)}
    for k, story_id in enumerate(ds.ids):
        j = row_of[story_id]
        # Cleaned exactly once; the expectation is whatever clean_text does.
        assert ds.texts[k] == clean_text(synthetic.texts[j])
        np.testing.assert_array_equal(ds.Y[k], synthetic.Y[j])  # NaN equals NaN here


def test_empty_cells_are_nan_and_mask_is_the_complement(synthetic):
    ds = data.load_dataset(synthetic.config)
    assert np.isnan(ds.Y).any() and (ds.mask == 1).any()  # the data set has both kinds
    np.testing.assert_array_equal(ds.mask, (~np.isnan(ds.Y)).astype(np.int8))
    assert set(np.unique(ds.Y[ds.mask == 1])) <= {0.0, 1.0}
    assert int(ds.mask.sum()) + int(np.isnan(ds.Y).sum()) == ds.Y.size


def test_exact_cell_semantics(synthetic, tmp_path):
    categories = synthetic.categories
    rows = [
        ["b" * 24, "text b", "1.0", "0.0", ""] + [""] * 17,
        ["a" * 24, "text a", "", "", "1.0"] + ["0.0"] * 17,
    ]
    folder = tmp_path / "tiny"
    folder.mkdir()
    write_csv(folder / "synthetic.csv", ["_id", "full_text"] + categories, rows)
    ds = data.load_dataset(tiny_config(synthetic, folder))
    assert ds.ids == ["a" * 24, "b" * 24]  # sorted, not in file order
    assert np.isnan(ds.Y[0, :2]).all() and ds.Y[0, 2] == 1.0
    assert ds.Y[1, 0] == 1.0 and ds.Y[1, 1] == 0.0 and np.isnan(ds.Y[1, 2])
    assert ds.mask[0, :3].tolist() == [0, 0, 1]
    assert ds.mask[1, :3].tolist() == [1, 1, 0]


@pytest.mark.parametrize("bad", ["0.5", "1", "yes", " ", "nan", "-1.0", "1.00"])
def test_bad_label_cell_raises_and_never_quotes_the_text(synthetic, variant, bad):
    def change(header, rows):
        rows[3][2 + 5] = bad

    with pytest.raises(ValueError) as error:
        data.load_dataset(variant(change))
    message = str(error.value)
    assert "data row 3" in message and synthetic.categories[5] in message
    assert all(text[:30] not in message for text in synthetic.texts)


def test_duplicate_ids_raise(variant):
    def change(header, rows):
        rows[7][0] = rows[2][0]

    with pytest.raises(ValueError, match="unique"):
        data.load_dataset(variant(change))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda h, r: h.__setitem__(0, "id"), "first columns"),
        (lambda h, r: h.__setitem__(1, "body"), "first columns"),
        (lambda h, r: (h.__setitem__(0, h[1]), h.__setitem__(1, "_id")), "first columns"),
        (lambda h, r: (h.pop(), [row.pop() for row in r]), "21 columns"),
        (lambda h, r: (h.append("Extra"), [row.append("") for row in r]), "23 columns"),
        (lambda h, r: r[4].pop(), "data row 4"),
    ],
)
def test_bad_header_or_row_length_raises(variant, change, message):
    with pytest.raises(ValueError, match=message):
        data.load_dataset(variant(change))


def test_missing_csv_raises_and_points_to_make_data(tmp_path, synthetic):
    config = tiny_config(synthetic, tmp_path / "nothing_here")
    with pytest.raises(FileNotFoundError, match="make data"):
        data.load_dataset(config)


def test_absolute_paths_in_the_config_are_used_as_they_are(synthetic):
    # ROOT / absolute path == absolute path; the synthetic config relies on this.
    assert data.data_path(synthetic.config) == synthetic.path
    assert Path(synthetic.config["paths"]["splits_dir"]).is_absolute()
    assert data.frozen_paths(synthetic.config)[0].parent == Path(synthetic.config["paths"]["splits_dir"])


# --------------------------------------------------------------------------------
# label_mask and make_splits
# --------------------------------------------------------------------------------

def test_label_mask():
    Y = np.array([[0.0, 1.0, np.nan], [np.nan, np.nan, 0.0]])
    before = Y.copy()
    mask = data.label_mask(Y)
    assert mask.dtype == np.int8
    assert mask.tolist() == [[1, 1, 0], [0, 0, 1]]
    np.testing.assert_array_equal(Y, before)  # input untouched


def test_make_splits_returns_disjoint_ascending_cover(synthetic):
    ds = data.load_dataset(synthetic.config)
    train, val, test = data.make_splits(ds.Y, synthetic.config, seed=0)
    for idx in (train, val, test):
        assert np.issubdtype(idx.dtype, np.integer)
        assert (np.diff(idx) > 0).all()
    everything = np.concatenate([train, val, test])
    assert sorted(everything.tolist()) == list(range(len(ds.ids)))
    assert abs(len(val) - 24) <= 3 and abs(len(test) - 24) <= 3
    # Same call, same answer; another seed, another split.
    again = data.make_splits(ds.Y, synthetic.config, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip((train, val, test), again))
    other = data.make_splits(ds.Y, synthetic.config, seed=1)
    assert not np.array_equal(test, other[2])


def test_make_splits_honors_groups(synthetic):
    ds = data.load_dataset(synthetic.config)
    groups = [str(i // 3) for i in range(len(ds.ids))]  # rows 0-2, 3-5, ... go together
    parts = data.make_splits(ds.Y, synthetic.config, seed=2, groups=groups)
    part_of = {}
    for name, idx in zip(data.PARTS, parts):
        part_of.update({int(i): name for i in idx})
    assert len(part_of) == len(ds.ids)
    for group in set(groups):
        members = [i for i, g in enumerate(groups) if g == group]
        assert len({part_of[i] for i in members}) == 1


def test_make_splits_rejects_bad_input(synthetic):
    ds = data.load_dataset(synthetic.config)
    with pytest.raises(ValueError, match="one entry per row"):
        data.make_splits(ds.Y, synthetic.config, seed=0, groups=["a", "b"])
    with pytest.raises(ValueError, match="2-D"):
        data.make_splits(ds.Y[:, 0], synthetic.config, seed=0)


# --------------------------------------------------------------------------------
# The frozen files
# --------------------------------------------------------------------------------

def test_frozen_csv_format(shared):
    csv_path, _ = data.frozen_paths(shared.config)
    raw = csv_path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    header, rows = read_csv(csv_path)
    assert header == ["_id"] + [f"seed_{s}" for s in SEEDS]
    ids = [row[0] for row in rows]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) == len(shared.synthetic.ids)
    assert all(cell in ("train", "val", "test") for row in rows for cell in row[1:])


def test_frozen_meta_content(shared):
    csv_path, meta_path = data.frozen_paths(shared.config)
    text = meta_path.read_text(encoding="utf-8")
    meta = json.loads(text)
    assert text == json.dumps(meta, sort_keys=True, indent=2) + "\n"  # deterministic layout
    assert "\r" not in text

    assert set(meta) == {
        "data_set", "file", "sha256", "n_stories", "categories", "seeds", "ratios", "method",
        "cleaning_version", "grouping", "versions", "created",
    }
    assert meta["data_set"] == "synthetic" and meta["file"] == "synthetic.csv"
    assert meta["sha256"] == hashlib.sha256(shared.synthetic.path.read_bytes()).hexdigest()
    assert meta["n_stories"] == len(shared.synthetic.ids)
    assert meta["categories"] == shared.synthetic.categories
    assert meta["seeds"] == SEEDS
    assert meta["ratios"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert isinstance(meta["method"], str) and "iterative stratification" in meta["method"]
    assert meta["cleaning_version"] == CLEANING_VERSION
    assert set(meta["versions"]) == {"python", "numpy", "scikit-learn", "iterative-stratification"}
    assert len(meta["created"]) == 10 and meta["created"][4] == "-"

    grouping = meta["grouping"]
    assert grouping["k"] == 5 and grouping["threshold"] == 0.8 and isinstance(grouping["version"], int)
    groups = grouping["groups"]
    assert groups, "the synthetic data has planted near-duplicates, so some group must be listed"
    assert all(len(g) > 1 and g == sorted(g) for g in groups)
    assert groups == sorted(groups, key=lambda g: g[0])
    # Every planted pair must have been found and put in one group.
    for first, second in shared.synthetic.planted:
        assert any(first in g and second in g for g in groups)


@pytest.mark.parametrize("seed", SEEDS)
def test_get_split_parts(shared, seed):
    synthetic = shared.synthetic
    ds = data.load_dataset(shared.config)
    split = data.get_split(seed, shared.config)
    assert split.seed == seed and split.categories == ds.categories

    parts = [split.train, split.val, split.test]
    all_ids = [i for part in parts for i in part.ids]
    assert sorted(all_ids) == ds.ids  # disjoint and covering
    assert len(set(all_ids)) == len(all_ids)
    assert abs(len(split.val.ids) - 24) <= 6 and abs(len(split.test.ids) - 24) <= 6
    row_of = {story_id: i for i, story_id in enumerate(ds.ids)}
    for part in parts:
        n = len(part.ids)
        assert part.ids == sorted(part.ids)
        assert part.Y.shape == part.mask.shape == (n, 20)
        assert part.Y.dtype == np.int8 and part.mask.dtype == np.int8
        assert len(part.texts) == n
        assert (part.Y[part.mask == 0] == 0).all()
        rows = [row_of[i] for i in part.ids]
        np.testing.assert_array_equal(part.mask, data.label_mask(ds.Y[rows]))
        np.testing.assert_array_equal(part.Y[part.mask == 1], ds.Y[rows][part.mask == 1].astype(np.int8))
        assert part.texts == [ds.texts[r] for r in rows]
    assert len(synthetic.ids) == len(all_ids)


def test_near_duplicates_are_never_split_across_parts(shared):
    groups = json.loads(data.frozen_paths(shared.config)[1].read_text())["grouping"]["groups"]
    for seed in SEEDS:
        split = data.get_split(seed, shared.config)
        part_of = {i: name for name in data.PARTS for i in getattr(split, name).ids}
        for group in groups:
            assert len({part_of[i] for i in group}) == 1


def test_seeds_give_different_splits(shared):
    tests = [tuple(data.get_split(s, shared.config).test.ids) for s in SEEDS]
    assert len(set(tests)) == len(SEEDS)


def test_numpy_integer_seed_is_accepted(shared):
    split = data.get_split(np.int64(1), shared.config)
    assert split.seed == 1 and type(split.seed) is int


@pytest.mark.parametrize("seed", [5, -1, True, "0", 1.0, None, np.True_])
def test_bad_seed_raises_and_lists_valid_seeds(shared, seed):
    with pytest.raises(ValueError, match=r"\[0, 1, 2, 3, 4\]"):
        data.get_split(seed, shared.config)


def test_dataclasses_are_frozen_and_comparable(shared):
    split = data.get_split(0, shared.config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        split.seed = 3
    assert split.train == split.train  # identity equality; no "ambiguous truth value" error
    assert split.train != data.get_split(0, shared.config).train


def test_get_split_without_frozen_files_points_to_make_splits(synthetic):
    with pytest.raises(FileNotFoundError, match="make splits"):
        data.get_split(0, synthetic.config)
    with pytest.raises(FileNotFoundError, match="make splits"):
        data.check_frozen_splits(synthetic.config)


def test_a_single_missing_frozen_file_is_reported(clone):
    _, meta_path = data.frozen_paths(clone.config)
    meta_path.unlink()
    with pytest.raises(FileNotFoundError, match=r"make splits"):
        data.get_split(0, clone.config)


def test_get_split_without_data_points_to_make_data(clone):
    clone.data.unlink()
    with pytest.raises(FileNotFoundError, match="make data"):
        data.get_split(0, clone.config)


def test_changed_id_after_freezing_raises(clone):
    def change(header, rows):
        rows[0][0] = "f" * 24

    edit_csv(clone.data, change)
    with pytest.raises(ValueError, match="changed"):
        data.get_split(0, clone.config)


def test_removed_story_after_freezing_raises(clone):
    edit_csv(clone.data, lambda header, rows: rows.pop())
    with pytest.raises(ValueError, match="changed"):
        data.get_split(0, clone.config)


def test_changed_categories_raise(clone):
    edit_meta(clone.config, lambda meta: meta["categories"].reverse())
    with pytest.raises(ValueError, match="category"):
        data.get_split(0, clone.config)


def test_changed_label_bytes_only_warn(shared, clone):
    expected = data.get_split(2, shared.config)

    def change(header, rows):
        rows[0][-1] = "0.0" if rows[0][-1] == "1.0" else "1.0"

    edit_csv(clone.data, change)
    with pytest.warns(UserWarning, match="bytes differ"):
        split = data.get_split(2, clone.config)
    assert split.train.ids == expected.train.ids
    assert split.val.ids == expected.val.ids
    assert split.test.ids == expected.test.ids


def test_cleaning_version_and_ratio_mismatches_warn(shared, clone):
    edit_meta(clone.config, lambda meta: meta.update(cleaning_version=CLEANING_VERSION + 1))
    with pytest.warns(UserWarning, match="cleaning version"):
        data.get_split(0, clone.config)

    other = copy.deepcopy(shared.config)
    other["splits"] = {"train": 0.6, "val": 0.2, "test": 0.2}
    with pytest.warns(UserWarning, match="config"):
        data.get_split(0, other)


def test_config_seed_missing_from_the_frozen_files_raises(synthetic):
    config = copy.deepcopy(synthetic.config)
    config["seeds"] = [0, 1]
    data.freeze_splits(config)
    config["seeds"] = [0, 1, 2]
    assert len(data.get_split(1, config).train.ids) > 0
    with pytest.raises(ValueError, match="not in the frozen files"):
        data.get_split(2, config)


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda h, r: r[3].__setitem__(1, "banana"), "part names"),
        (lambda h, r: r[3].pop(), "part names"),
        (lambda h, r: h.__setitem__(1, "fold_0"), "header"),
    ],
)
def test_malformed_frozen_csv_raises(clone, tamper, message):
    csv_path, _ = data.frozen_paths(clone.config)
    edit_csv(csv_path, tamper)
    with pytest.raises(ValueError, match=message):
        data.get_split(0, clone.config)


def test_malformed_frozen_meta_raises(clone):
    _, meta_path = data.frozen_paths(clone.config)
    meta_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        data.get_split(0, clone.config)
    meta_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the keys"):
        data.get_split(0, clone.config)


def test_get_split_never_imports_iterstrat_or_splits(shared, tmp_path):
    # A fresh interpreter in which importing iterstrat fails: get_split must still work,
    # which is what lets teammates with any library versions read the frozen split.
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(shared.config))
    src = Path(data.__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import json, sys
        sys.modules["iterstrat"] = None
        sys.modules["iterstrat.ml_stratifiers"] = None
        sys.path.insert(0, sys.argv[1])
        from listen.data import get_split
        split = get_split(0, json.load(open(sys.argv[2])))
        assert len(split.train.ids) > 0
        assert "listen.splits" not in sys.modules
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(src), str(config_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip() == "ok"


# --------------------------------------------------------------------------------
# freeze_splits and check_frozen_splits
# --------------------------------------------------------------------------------

def test_freeze_refuses_to_overwrite_without_force(clone, monkeypatch):
    csv_path, meta_path = data.frozen_paths(clone.config)
    before = (csv_path.read_bytes(), meta_path.read_bytes())
    with pytest.raises(FileExistsError, match="--force"):
        data.freeze_splits(clone.config)
    assert (csv_path.read_bytes(), meta_path.read_bytes()) == before  # nothing changed

    monkeypatch.setattr(data, "_today", lambda: "2001-02-03")
    data.freeze_splits(clone.config, force=True)
    assert json.loads(meta_path.read_text())["created"] == "2001-02-03"
    assert csv_path.read_bytes() == before[0]


def test_freeze_with_only_one_file_present_still_refuses(clone):
    _, meta_path = data.frozen_paths(clone.config)
    meta_path.unlink()
    with pytest.raises(FileExistsError):
        data.freeze_splits(clone.config)
    data.freeze_splits(clone.config, force=True)
    assert meta_path.is_file()


def test_freezing_twice_gives_identical_bytes(synthetic, monkeypatch):
    monkeypatch.setattr(data, "_today", lambda: "2026-01-01")
    csv_path, meta_path = data.frozen_paths(synthetic.config)
    data.freeze_splits(synthetic.config)
    first = (csv_path.read_bytes(), meta_path.read_bytes())
    data.freeze_splits(synthetic.config, force=True)
    assert (csv_path.read_bytes(), meta_path.read_bytes()) == first


def test_freeze_creates_the_splits_folder(synthetic):
    folder = Path(synthetic.config["paths"]["splits_dir"])
    assert not folder.exists()
    data.freeze_splits(synthetic.config)
    assert folder.is_dir()


def test_freeze_needs_a_splits_dir_and_valid_seeds(synthetic):
    config = copy.deepcopy(synthetic.config)
    del config["paths"]["splits_dir"]
    with pytest.raises(ValueError, match="splits_dir"):
        data.freeze_splits(config)
    config = copy.deepcopy(synthetic.config)
    config["seeds"] = [0, 0]
    with pytest.raises(ValueError, match="seeds"):
        data.freeze_splits(config)
    with pytest.raises(FileNotFoundError, match="make data"):
        data.freeze_splits(tiny_config(synthetic, Path(synthetic.config["paths"]["data_dir"]) / "none"))


def test_check_is_empty_right_after_freezing(shared):
    assert data.check_frozen_splits(shared.config) == []


def test_check_reports_a_tampered_frozen_csv(clone):
    csv_path, _ = data.frozen_paths(clone.config)

    def change(header, rows):
        rows[0][1] = "test" if rows[0][1] != "test" else "train"

    edit_csv(csv_path, change)
    differences = data.check_frozen_splits(clone.config)
    assert differences and any("seed 0" in line for line in differences)
    assert not any("seed 1" in line for line in differences)


def test_check_reports_tampered_meta_and_data(clone):
    edit_meta(clone.config, lambda meta: meta.update(n_stories=meta["n_stories"] + 1))
    assert any("n_stories" in line for line in data.check_frozen_splits(clone.config))

    def change(header, rows):
        rows[0][0] = "f" * 24

    edit_csv(clone.data, change)
    differences = data.check_frozen_splits(clone.config)
    assert any("story ids" in line for line in differences)
    assert any("sha256" in line for line in differences)


def test_check_reports_changed_groups_and_cleaning_version(clone):
    def change(meta):
        meta["grouping"]["groups"].pop()
        meta["cleaning_version"] += 1

    edit_meta(clone.config, change)
    differences = data.check_frozen_splits(clone.config)
    assert any("groups" in line for line in differences)
    assert any("cleaning_version" in line for line in differences)


def test_check_warns_about_library_versions_but_does_not_count_them(clone):
    edit_meta(clone.config, lambda meta: meta["versions"].update({"numpy": "0.0.1"}))
    with pytest.warns(UserWarning, match="Library versions differ"):
        assert data.check_frozen_splits(clone.config) == []


def test_check_is_quiet_when_versions_match(shared):
    with warnings.catch_warnings():
        # Only this project's warning is an error; library deprecation noise is not ours.
        warnings.filterwarnings("error", message="Library versions differ")
        assert data.check_frozen_splits(shared.config) == []
