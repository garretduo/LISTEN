#!/usr/bin/env python3
"""Freeze the train/val/test split of every seed into results/splits/.

Input:  configs/default.yaml (seeds, splits, paths.data_dir, paths.splits_dir, data) and
        the data CSV in data/ (run `make data` first).
Output: two small files in paths.splits_dir, named after data.default_set:
        <set>.csv (one part name per story and seed) and <set>.meta.json (sha256 of the
        data file, the near-duplicate groups, library versions and so on). They hold
        ids and part names only, never story text, and are committed to git so every
        teammate gets the same split from listen.data.get_split.

Usage:
    make splits                                         # write the files; refuses to overwrite
    PYTHONPATH=src python scripts/make_splits.py        # the same, run directly
    PYTHONPATH=src python scripts/make_splits.py --force   # overwrite existing files
    PYTHONPATH=src python scripts/make_splits.py --check   # recompute and compare, write nothing

`make` cannot pass --force, so replacing the files is always the direct command, and it
needs PYTHONPATH=src because the package is not pip-installed (the Makefile exports it).
--check exits with status 1 and lists the differences if the committed files no longer
match what the current code computes; otherwise it prints "frozen splits match" and exits 0.

Replacing the frozen files changes every result built on them. After a change to the data,
the cleaning, the grouping, the ratios or the seeds, bump CLEANING_VERSION or
GROUPING_VERSION where relevant, re-run with --force, and log it in docs/decisions.md.
"""

import argparse
import sys
from pathlib import Path

from listen import data

ROOT = Path(__file__).resolve().parent.parent


def show(path):
    """Return path relative to the repo root when it lies inside it, else as is."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def main():
    parser = argparse.ArgumentParser(description="Freeze the train/val/test split of every seed.")
    parser.add_argument("--force", action="store_true", help="overwrite existing frozen files")
    parser.add_argument(
        "--check", action="store_true", help="compare the files with a fresh computation; write nothing"
    )
    args = parser.parse_args()
    if args.force and args.check:
        parser.error("--force and --check cannot be combined")

    config = data.load_config()

    try:
        if args.check:
            differences = data.check_frozen_splits(config)
            if differences:
                print("Frozen splits differ from a fresh computation:")
                for line in differences:
                    print(f"  - {line}")
                sys.exit(1)
            print("frozen splits match")
            return

        data.freeze_splits(config, force=args.force)
        csv_path, meta_path = data.frozen_paths(config)
        print(f"wrote {show(csv_path)}")
        print(f"wrote {show(meta_path)}")
        # Read the files back through the public API: this also checks that what was
        # written is what get_split will see.
        for seed in config["seeds"]:
            split = data.get_split(seed, config)
            sizes = {name: len(getattr(split, name).ids) for name in ("train", "val", "test")}
            print(f"seed {seed}: train {sizes['train']}, val {sizes['val']}, test {sizes['test']}")
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        # Each of these carries a message that says what to run next.
        sys.exit(str(error))


if __name__ == "__main__":
    main()
