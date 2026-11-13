"""Shared test fixtures: the project config, synthetic data sets and the real frozen files.

Nothing here quotes or stores story text. The synthetic data sets are made of random
words and random hex ids, written into pytest's tmp_path. The fixtures that need the
real data (``dataset``) or the committed frozen files (``frozen``) skip when those are
absent instead of failing, so a fresh clone without `make data` still runs the rest.

``listen.data`` is imported inside the fixtures, not at the top, so a problem in that
module cannot stop other test files from being collected.
"""

import csv
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# Synthetic data set shape. 240 stories is small enough to freeze in about a second but
# big enough for 10-fold stratification with every category having positives.
N_SYNTHETIC = 240
N_PAIRS = 10  # near-duplicate pairs planted in the synthetic data
N_CATEGORIES = 20


@dataclass
class Synthetic:
    """A synthetic data set on disk plus what was written, in file order.

    Attributes:
        config: Settings dict whose paths point into the temporary folder.
        path: The CSV that was written.
        ids: Story ids in the order written (shuffled, so not sorted).
        texts: Raw story texts in the order written (random words, no real text).
        Y: float array (n, 20), NaN where the cell was written empty.
        categories: The 20 category names of the header.
        planted: Pairs of ids that were made near-duplicates of each other.
    """

    config: dict
    path: Path
    ids: list
    texts: list
    Y: np.ndarray
    categories: list
    planted: list


def write_synthetic_dataset(directory, n_stories=N_SYNTHETIC, seed=20260101):
    """Write a synthetic Learned-Hands-shaped CSV under ``directory`` and return it.

    The file has the real layout: ``_id``, ``full_text`` and 20 category columns with
    cells "0.0", "1.0" or "" (about 50% empty, about 10% of all cells equal to 1).
    Stories are 100 to 140 random words (some commas and newlines, to exercise CSV
    quoting). ``N_PAIRS`` stories get a near-duplicate: most copies differ in one word
    (Jaccard of 5-token shingles about 0.9) and two are exact copies. Rows are written
    in shuffled order so tests can check that loading sorts by id.

    Args:
        directory: Existing folder (for example pytest's ``tmp_path``). ``data/`` and
            ``splits/`` are used below it.
        n_stories: Total number of stories, planted copies included.
        seed: Seed of the random generators.

    Returns:
        A ``Synthetic`` describing what was written.
    """
    directory = Path(directory)
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    vocabulary = sorted(
        {"".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 9))) for _ in range(500)}
    )
    spare_words = [f"zz{w}" for w in vocabulary[:50]]  # used for edits; not in the vocabulary

    def story():
        words = rng.choices(vocabulary, k=rng.randint(100, 140))
        pieces = []
        for i, word in enumerate(words):
            if i % 17 == 16:
                word += ","
            pieces.append(word)
            pieces.append("\n" if i % 41 == 40 else " ")
        return "".join(pieces[:-1])

    n_base = n_stories - N_PAIRS
    ids = set()
    while len(ids) < n_stories:
        ids.add(f"{rng.getrandbits(96):024x}")
    ids = list(ids)
    rng.shuffle(ids)

    texts = [story() for _ in range(n_base)]
    labels = nprng.random((n_base, N_CATEGORIES))
    Y = np.where(labels < 0.5, np.nan, np.where(labels < 0.6, 1.0, 0.0))  # 50% empty, 10% ones

    planted = []
    for pair, base in enumerate(range(N_PAIRS)):
        copy = texts[base]
        if pair >= 2:  # two exact copies, the rest differ by one word in the middle
            words = copy.split(" ")
            words[len(words) // 2] = spare_words[pair]
            copy = " ".join(words)
        texts.append(copy)
        Y = np.vstack([Y, Y[base]])
        planted.append((ids[base], ids[n_base + pair]))

    categories = [f"Category {i:02d}" for i in range(N_CATEGORIES)]
    positives = (Y == 1.0).sum(axis=0)
    assert positives.min() >= 10, "synthetic labels too sparse for 10-fold stratification"

    # Shuffle the written order: the loader must sort by id, not trust the file.
    order = list(range(n_stories))
    rng.shuffle(order)
    ids = [ids[i] for i in order]
    texts = [texts[i] for i in order]
    Y = Y[order]
    planted = [tuple(pair) for pair in planted]

    data_dir = directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "synthetic.csv"
    cell_text = {0.0: "0.0", 1.0: "1.0"}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["_id", "full_text"] + categories)
        for story_id, text, row in zip(ids, texts, Y):
            writer.writerow([story_id, text] + ["" if np.isnan(v) else cell_text[v] for v in row])

    config = {
        "seeds": [0, 1, 2, 3, 4],
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "paths": {
            "data_dir": str(data_dir),
            "splits_dir": str(directory / "splits"),
        },
        "data": {
            "default_set": "synthetic",
            "id_column": "_id",
            "text_column": "full_text",
            "files": {"synthetic": {"name": "synthetic.csv", "rows": n_stories}},
        },
    }
    return Synthetic(config, path, ids, texts, Y, categories, planted)


@pytest.fixture(scope="session")
def config():
    """The project's real settings (configs/default.yaml), via listen.data.load_config."""
    from listen.data import load_config

    return load_config()


@pytest.fixture(scope="session")
def make_synthetic():
    """The write_synthetic_dataset function, for fixtures that need a wider scope."""
    return write_synthetic_dataset


@pytest.fixture
def synthetic(tmp_path):
    """A fresh synthetic data set in tmp_path (see write_synthetic_dataset)."""
    return write_synthetic_dataset(tmp_path)


@pytest.fixture(scope="session")
def dataset(config):
    """The real Dataset, loaded once. Skips when the data folder has no CSV yet."""
    from listen.data import data_path, load_dataset

    if not data_path(config).is_file():
        pytest.skip("run `make data` first")
    return load_dataset(config)


@dataclass
class FrozenFiles:
    """The committed frozen files, read directly (no listen.data parsing).

    Attributes:
        meta: The parsed meta JSON.
        ids: Story ids of the CSV rows.
        parts: seed -> list of part names aligned with ``ids``.
    """

    meta: dict
    ids: list
    parts: dict


@pytest.fixture(scope="session")
def frozen(config):
    """The real committed frozen files. Skips when they are absent (`make splits`)."""
    from listen.data import frozen_paths

    csv_path, meta_path = frozen_paths(config)
    if not (csv_path.is_file() and meta_path.is_file()):
        pytest.skip("run `make splits`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    seeds = [int(name.removeprefix("seed_")) for name in rows[0][1:]]
    ids = [row[0] for row in rows[1:]]
    parts = {seed: [row[1 + k] for row in rows[1:]] for k, seed in enumerate(seeds)}
    return FrozenFiles(meta=meta, ids=ids, parts=parts)
