"""Loading the Learned Hands data, the label mask, and the frozen train/val/test splits.

Purpose
-------
Everything that touches the raw CSV or decides which story is in train, val or test
lives here. Teammates should not read the CSV themselves and should not split the
data themselves: call ``get_split(seed)`` and fit on ``train``, tune on ``val`` and
touch ``test`` once, at the very end (README, "Splits and metrics").

Data flow
---------
1. ``load_dataset`` reads the raw CSV in ``data/`` (CC BY-NC-SA, gitignored) and
   returns a ``Dataset``: rows sorted by ``_id``, every story cleaned once with
   ``listen.text.clean_text``, labels as floats with NaN for omitted cells, and the
   0/1 label mask.
2. ``freeze_splits`` runs once (``make splits``, or
   ``PYTHONPATH=src python scripts/make_splits.py --force`` to overwrite). It finds
   near-duplicate stories (``listen.splits.near_duplicate_groups``), assigns each
   group to train, val or test for every seed in ``configs/default.yaml``
   (``listen.splits.assign_parts``), and writes two small files into
   ``paths.splits_dir`` (``results/splits``): ``<data set>.csv`` (one part name per
   story and seed) and ``<data set>.meta.json`` (how, when and from which data they
   were made). They hold ids and part names only, never story text, and are
   committed to git so every teammate gets the same split.
3. ``get_split`` loads the dataset and the frozen files, checks that they describe
   the same stories, and returns a ``Split`` of three ``Part`` objects. It never
   imports ``iterstrat`` or recomputes anything, so the split does not change with
   the installed library versions.

Why the split is frozen instead of recomputed: the stratification step depends on the
installed versions of iterative-stratification, scikit-learn and numpy, so recomputing
on each machine could silently give teammates different splits and make results
incomparable. (The near-duplicate grouping is plain Python and does not depend on
library versions, only on the cleaning, the shingle length and the threshold.) The
protocol is recorded in docs/decisions.md, in the entries "near-duplicate stories are
kept in the same part", "grouped iterative stratification, and the fold mapping",
"seeds 0 to 4 are frozen", "frozen split files" and "Y encoding, zero-filled labels
plus a mandatory mask". ``check_frozen_splits`` (``make_splits.py --check``)
recomputes the split and reports any difference from the files, which is how drift is
detected.

Invariants
----------
* Rows are sorted by ``_id`` in plain string order, in ``Dataset`` and in every
  ``Part``, so a row's position never depends on the CSV's row order.
* An empty label cell means "unknown", not "no" (README, "Partially labeled
  stories"). In a ``Dataset`` it is NaN in ``Y`` and 0 in ``mask``. In a ``Part`` it
  is 0 in ``Y`` (only because integer arrays cannot hold NaN) and 0 in ``mask``, so
  that 0 in ``Y`` still means "unknown" wherever ``mask`` is 0. Always use ``mask``;
  never treat ``Y == 0`` as a negative label.
* Texts are already cleaned. Do not call ``clean_text`` on them again: it is not
  guaranteed to be idempotent (see ``listen.text``).
* The test part is for the final evaluation only. Fit on train, tune thresholds and
  choose models on val, and look at test once after every choice is frozen.
* Only the seeds in ``configs/default.yaml`` (``config["seeds"]``) have frozen splits.
* The omitted-cell 0-fill exists only inside ``listen.splits.assign_parts``, for the
  stratification step. Nothing here fills NaN with 0 except the integer ``Part.Y``
  described above.

Changing anything the frozen files depend on
--------------------------------------------
The frozen files depend on the data file, ``listen.text.clean_text`` (through the
near-duplicate groups), the grouping and splitting algorithm, the split ratios and
the seeds. After changing any of them:

1. bump ``CLEANING_VERSION`` in ``listen/text.py`` if the cleaning output changed, or
   ``GROUPING_VERSION`` in ``listen/splits.py`` if the grouping or splitting changed;
2. regenerate with ``PYTHONPATH=src python scripts/make_splits.py --force`` (the
   ``make splits`` target refuses to overwrite and cannot pass ``--force``);
3. commit the new files together with an entry in ``docs/decisions.md`` (see "frozen
   split files" there), and re-run every result that used the old splits.

Run ``PYTHONPATH=src python scripts/make_splits.py --check`` to see whether the
committed files still match what the current code computes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import numbers
import platform
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from listen.text import CLEANING_VERSION, clean_text

__all__ = [
    "Dataset",
    "Part",
    "Split",
    "load_config",
    "load_dataset",
    "label_mask",
    "make_splits",
    "get_split",
    "freeze_splits",
    "check_frozen_splits",
    "data_path",
    "frozen_paths",
]

# Repo root: src/listen/data.py -> parents[0] = listen, [1] = src, [2] = repo root.
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "default.yaml"

N_CATEGORIES = 20  # label columns after the id and text columns
PARTS = ("train", "val", "test")

# Near-duplicate grouping settings recorded in the frozen meta file. They are fixed
# here (not in the config) because they define the frozen protocol; changing them
# means a GROUPING_VERSION bump and a new freeze (see the module docstring).
GROUPING_K = 5
GROUPING_THRESHOLD = 0.8

# The only cell values the label columns may hold. An empty cell is unknown, so it
# maps to NaN and never to 0.
_CELL_VALUES = {"": np.nan, "0.0": 0.0, "1.0": 1.0}

# Meta-file keys that readers rely on; a file without them is rejected up front with
# one clear message instead of a KeyError somewhere deep.
_META_KEYS = (
    "data_set", "file", "sha256", "n_stories", "categories", "seeds", "ratios",
    "method", "cleaning_version", "grouping", "versions", "created",
)


# --------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------

# eq=False on all three containers: the generated __eq__ would compare the numpy
# arrays with ==, which raises "truth value of an array is ambiguous" (and the
# generated __hash__ would fail on lists). Identity comparison is the honest default.
@dataclass(frozen=True, eq=False)
class Dataset:
    """The whole cleaned data set, before any split.

    The dataclass is frozen, so the fields cannot be reassigned. The arrays and lists
    inside are ordinary mutable objects: copy them before modifying in place.

    Attributes:
        ids: ``n`` story ids (the ``_id`` strings; in the real file they are
            24-character lowercase hex), sorted in plain string order. Unique.
        texts: ``n`` cleaned story texts, aligned with ``ids``. Already passed through
            ``clean_text`` once; do not clean again.
        Y: Labels, ``float64`` array of shape ``(n, 20)``. Values are 0.0, 1.0 or NaN.
            NaN means the cell was empty in the CSV, i.e. the label is unknown, not
            negative.
        mask: ``int8`` array of shape ``(n, 20)``. 1 where the label is known (``Y`` is
            not NaN), 0 where it is unknown. Equal to ``label_mask(Y)``.
        categories: The 20 category names, in the CSV's column order, which is also
            the column order of ``Y`` and ``mask``.
    """

    ids: list[str]
    texts: list[str]
    Y: np.ndarray
    mask: np.ndarray
    categories: list[str]


@dataclass(frozen=True, eq=False)
class Part:
    """One part of a split: the train, val or test stories.

    Rows are in ``_id`` order. The dataclass is frozen (fields cannot be reassigned);
    the arrays and lists inside are ordinary mutable objects.

    Attributes:
        ids: ``n`` story ids, ascending in plain string order.
        texts: ``n`` cleaned texts, aligned with ``ids``. Already cleaned; do not
            clean again.
        Y: ``int8`` array of shape ``(n, 20)`` with values 0 or 1. It is 0 wherever
            ``mask`` is 0, and there the 0 means UNKNOWN, not "no". Only the cells
            where ``mask == 1`` are real labels.
        mask: ``int8`` array of shape ``(n, 20)``. 1 where the label is known, 0 where
            it is unknown. Pass it as the ``mask`` argument of
            ``MultiLabelModel.fit`` and use it when tuning thresholds and scoring.
    """

    ids: list[str]
    texts: list[str]
    Y: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True, eq=False)
class Split:
    """One frozen 80/10/10 split of the data, for one seed.

    Attributes:
        seed: The seed this split belongs to (one of ``config["seeds"]``).
        train: Stories to fit models on.
        val: Stories to tune thresholds and make model choices on.
        test: Stories for the final evaluation only. Look at them once, after all
            model choices are frozen.
        categories: The 20 category names, in the column order of every ``Y`` and
            ``mask`` in the three parts.
    """

    seed: int
    train: Part
    val: Part
    test: Part
    categories: list[str]


# --------------------------------------------------------------------------------
# Config and paths
# --------------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict:
    """Read the project settings (``configs/default.yaml`` unless a path is given).

    Args:
        path: Path of a YAML file. ``None`` means ``configs/default.yaml`` in the repo
            root, wherever the program is run from.

    Returns:
        The parsed settings as a dict: ``seeds``, ``splits``, ``paths`` and ``data``.

    Raises:
        FileNotFoundError: If the file does not exist (check the path).
        ValueError: If the file does not hold a YAML mapping (fix the file).
        yaml.YAMLError: If the file is not valid YAML (fix the syntax).
    """
    config_path = CONFIG_PATH if path is None else Path(path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must hold a YAML mapping, got {type(config).__name__}.")
    return config


def _resolve_config(config: dict | None) -> dict:
    """Return ``config``, or the default settings when it is None."""
    return load_config() if config is None else config


def data_path(config: dict | None = None) -> Path:
    """Return the path of the CSV chosen by ``config["data"]["default_set"]``.

    Args:
        config: Settings dict, or ``None`` for ``configs/default.yaml``. The path is
            ``paths.data_dir`` joined with the file name of ``data.default_set``. A
            relative ``data_dir`` is relative to the repo root; an absolute one is
            used as it is.

    Returns:
        The path. It is not checked for existence.

    Raises:
        KeyError: If the config lacks ``paths.data_dir``, ``data.default_set`` or the
            matching entry of ``data.files`` (fix the config).
    """
    config = _resolve_config(config)
    name = config["data"]["files"][config["data"]["default_set"]]["name"]
    return ROOT / config["paths"]["data_dir"] / name


def frozen_paths(config: dict | None = None) -> tuple[Path, Path]:
    """Return the paths of the two frozen split files for the configured data set.

    Args:
        config: Settings dict, or ``None`` for ``configs/default.yaml``. The files are
            in ``paths.splits_dir`` (relative to the repo root, or absolute) and are
            named after ``data.default_set``.

    Returns:
        ``(csv_path, meta_path)``: ``<splits_dir>/<default_set>.csv`` and
        ``<splits_dir>/<default_set>.meta.json``. Neither is checked for existence.

    Raises:
        ValueError: If the config has no ``paths.splits_dir`` (add
            ``splits_dir: results/splits`` under ``paths:`` in the config).
        KeyError: If the config lacks ``data.default_set``.
    """
    config = _resolve_config(config)
    try:
        splits_dir = ROOT / config["paths"]["splits_dir"]
    except KeyError:
        raise ValueError(
            "The config has no paths.splits_dir. Add `splits_dir: results/splits` "
            "under `paths:` in configs/default.yaml."
        ) from None
    key = config["data"]["default_set"]
    return splits_dir / f"{key}.csv", splits_dir / f"{key}.meta.json"


def _config_seeds(config: dict) -> list[int]:
    """Return config["seeds"] as a list of ints, after checking it is sane.

    Raises:
        ValueError: If the seeds are not a non-empty list of distinct integers.
    """
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, (list, tuple))
        or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            f"config['seeds'] must be a non-empty list of distinct integers, got {seeds!r}."
        )
    return list(seeds)


# --------------------------------------------------------------------------------
# Loading the data
# --------------------------------------------------------------------------------

def label_mask(Y: np.ndarray) -> np.ndarray:
    """Return the 0/1 mask of known labels: 1 where ``Y`` has a value, 0 where it is NaN.

    An omitted CSV cell is unknown, not negative (README, "Partially labeled
    stories"), so fits, thresholds and metrics must use only the cells where the
    mask is 1.

    Args:
        Y: Label array of any shape (normally ``(n_stories, 20)``) in which omitted
            cells are NaN. Anything ``np.asarray(..., dtype=float)`` accepts.

    Returns:
        ``int8`` array of the same shape as ``Y``: 1 where ``Y`` is not NaN, 0 where it
        is NaN.

    Raises:
        ValueError: If ``Y`` cannot be converted to floats (pass numeric labels).
    """
    return (~np.isnan(np.asarray(Y, dtype=float))).astype(np.int8)


def load_dataset(config: dict | None = None) -> Dataset:
    """Read the configured CSV into a cleaned ``Dataset`` sorted by ``_id``.

    The file is ``paths.data_dir`` / the file name of ``data.default_set`` (see
    ``data_path``). The header must be ``_id, full_text`` plus 20 category columns.
    Each label cell must be empty (omitted, read as NaN: unknown, not negative),
    ``0.0`` or ``1.0``. Rows are sorted by ``_id`` in plain string order and every
    text goes through ``listen.text.clean_text`` exactly once. The story text is never
    included in an error message.

    Args:
        config: Settings dict, or ``None`` for ``configs/default.yaml``. Uses
            ``paths.data_dir``, ``data.default_set``, ``data.files`` and
            ``data.id_column`` / ``data.text_column``.

    Returns:
        A ``Dataset`` with ``n`` rows (3,459 for the best-guess file): ``ids`` sorted
        and unique, ``texts`` cleaned, ``Y`` float64 ``(n, 20)`` with NaN for empty
        cells, ``mask`` int8 ``(n, 20)`` with 1 where the label is known, and the 20
        ``categories`` from the header.

    Raises:
        FileNotFoundError: If the data file is missing. Run `make data`.
        ValueError: If the header does not start with the configured id and text
            columns or does not have exactly 22 columns; if a row has a different
            number of cells than the header; if a label cell is not "", "0.0" or
            "1.0" (the message names the 0-based data row, header not counted, and
            the column); if two rows share an id; or if the file is not valid UTF-8
            or not readable as CSV. Fix or re-download the file (`make data`).
        KeyError: If the config lacks one of the keys listed under ``config`` above.

    Example:
        Real data (needs `make data` first)::

            ds = load_dataset()
            ds.Y.shape, ds.mask.shape        # ((3459, 20), (3459, 20))
            ds.ids == sorted(ds.ids)         # True
            int(np.isnan(ds.Y).sum()) == int((ds.mask == 0).sum())   # True
    """
    config = _resolve_config(config)
    path = data_path(config)
    id_column = config["data"]["id_column"]
    text_column = config["data"]["text_column"]

    if not path.is_file():
        raise FileNotFoundError(
            f"The data file {path} is missing. Run `make data` to download it."
        )

    csv.field_size_limit(2**31 - 1)  # posts are long; the default field limit is small
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = list(reader)
    except csv.Error as error:
        raise ValueError(
            f"{path.name} is not a readable CSV ({error}). Re-download it with `make data`."
        ) from error

    # Header: fail on the first structural problem before looking at any row.
    problems = []
    if header[:2] != [id_column, text_column]:
        problems.append(f"first columns are {header[:2]}, expected {[id_column, text_column]}")
    if len(header) != N_CATEGORIES + 2:
        problems.append(f"{len(header)} columns, expected {N_CATEGORIES + 2}")
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems) + ". Re-download it with `make data`.")
    categories = header[2:]

    n = len(rows)
    ids: list[str] = []
    raw_texts: list[str] = []
    Y = np.full((n, N_CATEGORIES), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                f"{path.name}: data row {i} (0-based, header not counted) has "
                f"{len(row)} cells, expected {len(header)}."
            )
        ids.append(row[0])
        raw_texts.append(row[1])
        for c, cell in enumerate(row[2:]):
            if cell not in _CELL_VALUES:
                # Only the label cell is quoted, never the story text.
                raise ValueError(
                    f"{path.name}: data row {i} (0-based, header not counted), column "
                    f"{categories[c]!r}: expected '', '0.0' or '1.0', got {cell[:20]!r}."
                )
            Y[i, c] = _CELL_VALUES[cell]

    if len(set(ids)) != n:
        repeated = sorted(story_id for story_id, count in Counter(ids).items() if count > 1)
        raise ValueError(
            f"{path.name}: ids must be unique, but {len(repeated)} id(s) appear more than "
            f"once (for example {repeated[:3]})."
        )

    # Sort by id so a story's position never depends on the CSV's row order. Plain
    # string order, as in listen.splits (group ids are the smallest id).
    order = sorted(range(n), key=ids.__getitem__)
    Y = Y[order]
    return Dataset(
        ids=[ids[i] for i in order],
        texts=[clean_text(raw_texts[i]) for i in order],  # once; texts are never cleaned again
        Y=Y,
        mask=label_mask(Y),
        categories=list(categories),
    )


# --------------------------------------------------------------------------------
# Splitting (computation) and the frozen files
# --------------------------------------------------------------------------------

def make_splits(
    Y: np.ndarray,
    config: dict,
    seed: int,
    groups: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a fresh 80/10/10 split as row indices (not the frozen split).

    This recomputes with the installed ``iterstrat``, so it may differ from the frozen
    split of ``get_split`` if library versions differ. Use ``get_split`` for results;
    use this to experiment or in tests. It is a thin wrapper around
    ``listen.splits.assign_parts``, which treats omitted (NaN) cells as 0 for the
    stratification step only and never changes ``Y``.

    Args:
        Y: Label matrix, ``(n_stories, 20)`` float, NaN where the label is omitted.
        config: Settings dict; ``config["splits"]`` holds the ratios
            (``train``/``val``/``test``).
        seed: Integer seed for the shuffle. Not restricted to ``config["seeds"]``
            here.
        groups: ``n_stories`` group ids (strings); rows with equal ids always land in
            the same part. ``None`` makes every row its own group.

    Returns:
        ``(train_idx, val_idx, test_idx)``: three int arrays of row indices into
        ``Y``, each ascending, disjoint, and together covering every row.

    Raises:
        ValueError: If ``Y`` is not 2-D or ``groups`` does not have one entry per
            row (fix the arguments); or, from ``assign_parts``, if the ratios are
            invalid, ``seed`` is not an integer, or there are fewer distinct groups
            than folds (its message says how to fix each).
        KeyError: If ``config`` has no ``splits`` section.
        ImportError: If ``iterative-stratification`` is not installed
            (`pip install -r requirements.txt`).
    """
    # Imported here, not at the top: get_split must never pull in iterstrat or
    # recompute anything, so teammates get the frozen split whatever they have installed.
    from listen.splits import assign_parts

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n_stories, n_categories), got shape {Y.shape}.")
    if groups is None:
        groups = [str(i) for i in range(Y.shape[0])]  # every row is its own group
    elif len(groups) != Y.shape[0]:
        raise ValueError(
            f"groups must have one entry per row of Y: got {len(groups)} for {Y.shape[0]} rows."
        )
    parts = np.asarray(assign_parts(Y, groups, config["splits"], seed))
    return tuple(np.flatnonzero(parts == name) for name in PARTS)


def _today() -> str:
    """Return today's date as ISO text. A function so tests can pin it."""
    return date.today().isoformat()


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    """Return the installed versions that can change a split, for the meta file."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "iterative-stratification"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@dataclass(frozen=True, eq=False)
class _Frozen:
    """What the frozen files hold, either read from them or recomputed from the data.

    Attributes:
        meta: The meta file as a dict (see ``freeze_splits`` for the keys).
        ids: Story ids in the frozen CSV's row order (sorted by id).
        parts: seed -> list of part names ("train"/"val"/"test"), aligned with ``ids``.
    """

    meta: dict
    ids: list[str]
    parts: dict[int, list[str]]


def _compute_frozen(config: dict) -> _Frozen:
    """Recompute everything the frozen files should contain from the data and the code.

    Shared by ``freeze_splits`` (which writes the result) and ``check_frozen_splits``
    (which compares it with the files), so a check recomputes exactly what a freeze
    would write.
    """
    # Imported here, not at the top, for the same reason as in make_splits.
    from listen.splits import GROUPING_VERSION, assign_parts, near_duplicate_groups

    seeds = _config_seeds(config)
    dataset = load_dataset(config)
    # Near-duplicate stories (reposts) share a group, and a group is never split across
    # parts, so a model cannot be tested on a copy of a story it trained on.
    group_of_row = near_duplicate_groups(
        dataset.texts, dataset.ids, k=GROUPING_K, threshold=GROUPING_THRESHOLD
    )
    parts = {
        seed: assign_parts(dataset.Y, group_of_row, config["splits"], seed).tolist()
        for seed in seeds
    }

    members: dict[str, list[str]] = {}
    for story_id, group_id in zip(dataset.ids, group_of_row):
        members.setdefault(group_id, []).append(story_id)
    # Only real families are listed (a single story is trivially alone); sorted so the
    # JSON is the same on every run.
    families = sorted(sorted(ids) for ids in members.values() if len(ids) > 1)

    path = data_path(config)
    n_splits = round(1 / config["splits"]["test"])  # assign_parts has validated the ratios
    meta = {
        "data_set": config["data"]["default_set"],
        "file": path.name,
        "sha256": _sha256(path),
        "n_stories": len(dataset.ids),
        "categories": dataset.categories,
        "seeds": seeds,
        "ratios": {name: config["splits"][name] for name in PARTS},
        "method": (
            f"Grouped multilabel iterative stratification: near-duplicate stories share a "
            f"group, groups are split into {n_splits} folds stratified on the labels "
            f"(omitted cells counted as 0 for splitting only); fold 0 is test, fold 1 is "
            f"val, the remaining folds are train; rows sorted by _id."
        ),
        "cleaning_version": CLEANING_VERSION,
        "grouping": {
            "version": GROUPING_VERSION,
            "k": GROUPING_K,
            "threshold": GROUPING_THRESHOLD,
            "groups": families,
        },
        "versions": _library_versions(),
        "created": _today(),
    }
    return _Frozen(meta=meta, ids=dataset.ids, parts=parts)


def _write_frozen(config: dict, frozen: _Frozen) -> None:
    """Write the two frozen files deterministically (same input, same bytes)."""
    csv_path, meta_path = frozen_paths(config)
    seeds = list(frozen.meta["seeds"])
    id_column = config["data"]["id_column"]

    # "\n" line endings and a fixed column order: csv.writer would use "\r\n" by
    # default, which would make the file differ between Windows and macOS checkouts.
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([id_column] + [f"seed_{seed}" for seed in seeds])
    for row, story_id in enumerate(frozen.ids):
        writer.writerow([story_id] + [frozen.parts[seed][row] for seed in seeds])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_text = json.dumps(frozen.meta, sort_keys=True, indent=2) + "\n"
    for path, text in ((csv_path, buffer.getvalue()), (meta_path, meta_text)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:  # newline="\n": no "\r\n" on Windows
            f.write(text)


def _read_frozen(config: dict) -> _Frozen:
    """Read and validate the two frozen files.

    Raises:
        FileNotFoundError: If either file is missing (run `make splits`).
        ValueError: If a file is malformed (not JSON, missing keys, wrong header, a
            cell that is not train/val/test, a row of the wrong length).
    """
    csv_path, meta_path = frozen_paths(config)
    missing = [p for p in (csv_path, meta_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Frozen split file(s) missing: " + ", ".join(str(p) for p in missing)
            + ". Run `make splits` to create them (or restore them from git)."
        )

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(
            f"{meta_path} is not valid JSON ({error}). Restore it from git or regenerate "
            "it with `PYTHONPATH=src python scripts/make_splits.py --force`."
        ) from error
    absent = [key for key in _META_KEYS if key not in meta]
    grouping = meta.get("grouping")
    if not isinstance(grouping, dict):
        absent.append("grouping (a mapping)")
    else:
        wanted = ("version", "k", "threshold", "groups")
        absent += [f"grouping.{key}" for key in wanted if key not in grouping]
    if absent:
        raise ValueError(
            f"{meta_path.name} lacks the keys {absent}. Restore it from git or regenerate "
            "it with `PYTHONPATH=src python scripts/make_splits.py --force`."
        )

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0] if rows else []
    seed_columns = [re.fullmatch(r"seed_(\d+)", name) for name in header[1:]]
    if not header or header[0] != config["data"]["id_column"] or not all(seed_columns):
        raise ValueError(
            f"{csv_path.name}: the header must be the id column followed by seed_<n> "
            f"columns, got {header[:4]}. Restore it from git or regenerate it with "
            "`PYTHONPATH=src python scripts/make_splits.py --force`."
        )
    seeds = [int(match.group(1)) for match in seed_columns]
    ids: list[str] = []
    parts: dict[int, list[str]] = {seed: [] for seed in seeds}
    for i, row in enumerate(rows[1:]):
        if len(row) != len(header) or any(cell not in PARTS for cell in row[1:]):
            raise ValueError(
                f"{csv_path.name}: data row {i} (0-based, header not counted) must hold "
                f"an id and {len(seeds)} part names from {list(PARTS)}. Restore the "
                "file from git or regenerate it with "
                "`PYTHONPATH=src python scripts/make_splits.py --force`."
            )
        ids.append(row[0])
        for seed, cell in zip(seeds, row[1:]):
            parts[seed].append(cell)
    return _Frozen(meta=meta, ids=ids, parts=parts)


def _version_warning(frozen_versions: dict) -> None:
    """Warn if the library versions in the meta differ from the installed ones."""
    installed = _library_versions()
    differ = {
        name: (frozen_versions.get(name), installed[name])
        for name in installed
        if frozen_versions.get(name) != installed[name]
    }
    if differ:
        detail = ", ".join(f"{k}: frozen {a}, installed {b}" for k, (a, b) in differ.items())
        warnings.warn(
            f"Library versions differ from those the splits were frozen with ({detail}). "
            "A mismatch may explain differences reported by the check.",
            UserWarning,
            stacklevel=3,
        )


def freeze_splits(config: dict | None = None, force: bool = False) -> None:
    """Compute the near-duplicate groups and the split of every seed, and write them to disk.

    This is the one-time (or rare) step behind `make splits`. It writes two files into
    ``paths.splits_dir`` (see ``frozen_paths``), named after ``data.default_set``:

    * ``<set>.csv``: header ``_id,seed_0,seed_1,...`` (one column per config seed),
      one row per story sorted by ``_id``, each cell exactly ``train``, ``val`` or
      ``test``.
    * ``<set>.meta.json``: ``data_set``, ``file`` (the CSV's name), ``sha256`` (of the
      raw data CSV's bytes), ``n_stories``, ``categories``, ``seeds``, ``ratios``,
      ``method`` (plain-English description), ``cleaning_version``, ``grouping``
      (``version``, ``k``, ``threshold`` and ``groups``: only the near-duplicate
      families of more than one story, each sorted, listed by first id), ``versions``
      (python, numpy, scikit-learn, iterative-stratification) and ``created`` (ISO
      date).

    Both files are written deterministically (sorted keys, ``\\n`` line endings,
    trailing newline), so regenerating from the same data, code and library versions on
    the same day gives byte-identical files; only ``created`` can differ on another day.
    They contain ids and part names only, never story text.

    After changing the data, cleaning, grouping, ratios or seeds, regenerate with
    ``force=True``, bump ``CLEANING_VERSION`` / ``GROUPING_VERSION`` where relevant, and
    add an entry to ``docs/decisions.md`` (see the module docstring).

    Args:
        config: Settings dict, or ``None`` for ``configs/default.yaml``.
        force: ``False`` refuses to overwrite existing frozen files. ``True`` replaces
            them (and creates the folder if needed).

    Returns:
        None. The files are the result.

    Raises:
        FileExistsError: If either frozen file exists and ``force`` is False. Re-run
            with ``--force`` (`PYTHONPATH=src python scripts/make_splits.py --force`)
            if replacing them is intended.
        FileNotFoundError: If the data file is missing. Run `make data`.
        ValueError: If the data file is malformed (see ``load_dataset``), if
            ``config["seeds"]`` is not a non-empty list of distinct integers, if
            ``paths.splits_dir`` is not in the config, or if the split ratios are
            invalid (``assign_parts`` says how to fix them).
        KeyError: If the config lacks another required key (``data``, ``splits``).
        ImportError: If ``iterative-stratification`` is not installed
            (`pip install -r requirements.txt`).
    """
    config = _resolve_config(config)
    existing = [p for p in frozen_paths(config) if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "Frozen split file(s) already exist: " + ", ".join(str(p) for p in existing)
            + ". Replacing them changes every result built on them. If that is intended, "
            "run `PYTHONPATH=src python scripts/make_splits.py --force` and log the "
            "change in docs/decisions.md."
        )
    _write_frozen(config, _compute_frozen(config))


def _diff(expected: _Frozen, actual: _Frozen) -> list[str]:
    """Describe how the files (``actual``) differ from the recomputation (``expected``).

    ``meta["created"]`` and ``meta["versions"]`` are not compared (the dates always
    differ; versions are handled by a warning). The ``method`` text is not compared
    either: it only describes the algorithm, and a change to the algorithm shows up in
    ``grouping.version`` or in the parts.
    """
    problems: list[str] = []
    e, a = expected.meta, actual.meta
    for key in ("data_set", "file", "n_stories", "categories", "seeds", "ratios", "cleaning_version"):
        if e[key] != a[key]:
            problems.append(f"meta {key}: recomputed {e[key]!r}, frozen file has {a[key]!r}")
    if e["sha256"] != a["sha256"]:
        problems.append("meta sha256: the data file's bytes differ from those the splits were frozen from")
    for key in ("version", "k", "threshold"):
        if e["grouping"][key] != a["grouping"][key]:
            problems.append(
                f"meta grouping {key}: recomputed {e['grouping'][key]!r}, "
                f"frozen file has {a['grouping'][key]!r}"
            )
    if e["grouping"]["groups"] != a["grouping"]["groups"]:
        problems.append(
            f"meta grouping groups: recomputed {len(e['grouping']['groups'])} near-duplicate "
            f"groups, frozen file lists {len(a['grouping']['groups'])}, and they are not identical"
        )

    if expected.ids != actual.ids:
        in_data, in_frozen = set(expected.ids), set(actual.ids)
        problems.append(
            f"story ids: the data has {len(expected.ids)}, the frozen CSV has {len(actual.ids)}; "
            f"{len(in_data - in_frozen)} only in the data, {len(in_frozen - in_data)} only in "
            "the frozen CSV (parts were not compared)"
        )
        return problems

    for seed in sorted(set(expected.parts) | set(actual.parts)):
        if seed not in actual.parts:
            problems.append(f"seed {seed}: configured, but the frozen CSV has no seed_{seed} column")
        elif seed not in expected.parts:
            problems.append(f"seed {seed}: in the frozen CSV, but not in config['seeds']")
        else:
            wrong = sum(x != y for x, y in zip(expected.parts[seed], actual.parts[seed]))
            if wrong:
                problems.append(
                    f"seed {seed}: {wrong} of {len(expected.ids)} stories are in a different part "
                    "than recomputed"
                )
    return problems


def check_frozen_splits(config: dict | None = None) -> list[str]:
    """Recompute the split and list every difference from the frozen files.

    Recomputes exactly what ``freeze_splits`` would write (reading the data, cleaning,
    grouping, stratifying with the installed libraries) and compares it with the files
    on disk: the parts of every seed, the near-duplicate groups, the categories,
    ``n_stories``, the story ids, the data set key and file name, the grouping
    settings, the cleaning version, the ratios, the seeds and the data file's SHA-256.
    The ``created`` date, the ``method`` text and the library versions are not
    counted as differences (the versions only trigger the warning below). Writes
    nothing.

    If the library versions recorded in the meta file differ from the installed ones, a
    ``UserWarning`` is issued, because a version change may explain differences in the
    parts. Differences are counted, never shown with story text.

    Args:
        config: Settings dict, or ``None`` for ``configs/default.yaml``.

    Returns:
        A list of human-readable differences, one string each. Empty means the files
        are identical to a fresh computation.

    Raises:
        FileNotFoundError: If the data file is missing (`make data`) or a frozen file
            is missing (`make splits`).
        ValueError: If the data file or a frozen file is malformed (the message says
            how to restore or regenerate it), or if the config is invalid (see
            ``freeze_splits``).
        ImportError: If ``iterative-stratification`` is not installed.
    """
    config = _resolve_config(config)
    actual = _read_frozen(config)
    expected = _compute_frozen(config)
    _version_warning(actual.meta["versions"])
    return _diff(expected, actual)


# --------------------------------------------------------------------------------
# The frozen split that teammates use
# --------------------------------------------------------------------------------

def _make_part(dataset: Dataset, rows: np.ndarray) -> Part:
    """Build one ``Part`` from the given ascending row indices of the dataset."""
    mask = dataset.mask[rows].astype(np.int8)
    # Y is 0 where mask is 0: the dataset's NaN cannot live in an int8 array. That 0
    # means "unknown", and the mask is how callers tell it apart from a real 0.
    Y = np.where(mask == 1, dataset.Y[rows], 0.0).astype(np.int8)
    return Part(
        ids=[dataset.ids[i] for i in rows],
        texts=[dataset.texts[i] for i in rows],
        Y=Y,
        mask=mask,
    )


def get_split(seed: int, config: dict | None = None) -> Split:
    """Return the frozen train/val/test split for one seed.

    Loads the data (``load_dataset``) and the frozen files (``freeze_splits`` wrote
    them), checks they describe the same stories, and cuts the dataset into three
    ``Part`` objects. Nothing is recomputed and ``iterstrat`` is never imported, so
    everyone with the same committed files and data gets the identical split.

    The three parts are disjoint and together cover every story. The frozen files keep
    each near-duplicate group in one part (``check_frozen_splits`` verifies this by
    recomputation; ``get_split`` itself does not re-check it). Rows in each part are
    in ``_id`` order. ``Y`` is int8 and is 0 wherever ``mask`` is 0, where the 0 means
    UNKNOWN (an omitted cell), not "no": only ``mask == 1`` cells are real labels.
    Texts are already cleaned; do not clean them again. Fit on ``train``, tune on
    ``val``, and use ``test`` only for the final evaluation. The ``seed`` must be one
    of ``config["seeds"]``.

    Args:
        seed: An integer in ``config["seeds"]`` (0 to 4 by default). A ``bool``, a
            float or a string is rejected; numpy integers are accepted.
        config: Settings dict, or ``None`` for ``configs/default.yaml``.

    Returns:
        A ``Split`` with ``seed``, the ``train``, ``val`` and ``test`` ``Part`` objects
        (about 80%, 10% and 10% of the stories) and the 20 ``categories``.

    Raises:
        ValueError: If ``seed`` is not an integer in ``config["seeds"]`` (the message
            lists the valid seeds), or not in the frozen files (re-run the freeze); if
            ``config["seeds"]`` is not a non-empty list of distinct integers; if
            the story ids in the data and in the frozen files differ, or the categories
            in the frozen meta file differ from the CSV header (the data or the frozen
            files changed: restore the right files, or regenerate with
            `PYTHONPATH=src python scripts/make_splits.py --force` and log it in
            docs/decisions.md); or if a frozen file is malformed.
        FileNotFoundError: If the data file is missing (run `make data`) or a frozen
            file is missing (run `make splits`, or restore the files from git).

    Warns:
        UserWarning: If the data file's SHA-256 differs from the one recorded at freeze
            time while the ids match (some bytes of the file changed, for example a
            label); if the frozen cleaning version differs from the current
            ``CLEANING_VERSION``; or if the config's split ratios differ from the
            frozen ones. The frozen split is still returned.

    Example:
        Real data and committed frozen files needed::

            split = get_split(0)
            len(split.train.ids), len(split.val.ids), len(split.test.ids)  # about 2767, 346, 346
            model.fit(split.train.texts, split.train.Y, split.train.mask)
            proba = model.predict_proba(split.val.texts)   # shape (len(split.val.ids), 20)
    """
    config = _resolve_config(config)
    valid_seeds = _config_seeds(config)
    if (
        not isinstance(seed, numbers.Integral)
        or isinstance(seed, (bool, np.bool_))
        or int(seed) not in valid_seeds
    ):
        raise ValueError(
            f"seed must be an integer in config['seeds'] = {valid_seeds}, got {seed!r}. "
            "Only these seeds have frozen splits."
        )
    seed = int(seed)

    dataset = load_dataset(config)
    frozen = _read_frozen(config)

    if seed not in frozen.parts or seed not in frozen.meta["seeds"]:
        raise ValueError(
            f"seed {seed} is in the config but not in the frozen files (they hold seeds "
            f"{sorted(frozen.parts)}). The seeds changed after freezing: regenerate with "
            "`PYTHONPATH=src python scripts/make_splits.py --force` and log it in docs/decisions.md."
        )
    if frozen.ids != dataset.ids:
        in_data, in_frozen = set(dataset.ids), set(frozen.ids)
        raise ValueError(
            f"The story ids in the data ({len(dataset.ids)}) and in the frozen files "
            f"({len(frozen.ids)}) differ: {len(in_data - in_frozen)} only in the data, "
            f"{len(in_frozen - in_data)} only in the frozen files. The data or the frozen "
            "files changed. Restore the right ones, or regenerate with "
            "`PYTHONPATH=src python scripts/make_splits.py --force` and log it in docs/decisions.md."
        )
    if frozen.meta["categories"] != dataset.categories:
        raise ValueError(
            "The category columns in the frozen meta file differ from the data file's header "
            f"({len(frozen.meta['categories'])} vs {len(dataset.categories)} names, or a "
            "different order). The data or the frozen files changed. Restore the right ones, "
            "or regenerate with `PYTHONPATH=src python scripts/make_splits.py --force`."
        )

    # Soft problems: the split is still the frozen one, but the caller should know.
    if frozen.meta["sha256"] != _sha256(data_path(config)):
        warnings.warn(
            "The data file's bytes differ from those the splits were frozen from, although "
            "the story ids match (for example a label changed). Returning the frozen split "
            "anyway. If the change is intended, regenerate with "
            "`PYTHONPATH=src python scripts/make_splits.py --force` and log it in docs/decisions.md.",
            UserWarning,
            stacklevel=2,
        )
    if frozen.meta["cleaning_version"] != CLEANING_VERSION:
        warnings.warn(
            f"The splits were frozen with cleaning version {frozen.meta['cleaning_version']}, "
            f"but listen.text is at version {CLEANING_VERSION}, so the near-duplicate groups "
            "may no longer match the cleaned texts. Regenerate with "
            "`PYTHONPATH=src python scripts/make_splits.py --force`.",
            UserWarning,
            stacklevel=2,
        )
    if frozen.meta["ratios"] != {name: config["splits"][name] for name in PARTS}:
        warnings.warn(
            f"config['splits'] is {config['splits']} but the splits were frozen with "
            f"{frozen.meta['ratios']}. Returning the frozen split; regenerate it if the "
            "new ratios are intended.",
            UserWarning,
            stacklevel=2,
        )

    labels = np.asarray(frozen.parts[seed])
    train, val, test = (_make_part(dataset, np.flatnonzero(labels == name)) for name in PARTS)
    return Split(seed=seed, train=train, val=val, test=test, categories=list(dataset.categories))
