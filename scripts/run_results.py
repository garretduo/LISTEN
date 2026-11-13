#!/usr/bin/env python3
"""Fit every model and print the results table (the driver behind `make results`).

Input:  configs/default.yaml, the CSVs in data/ (run `make data` first) and the frozen
        splits in results/splits/ (committed; `make splits` recreates them).
Output: one line per seed with the train, val and test sizes, on stdout. The results table
        will follow, and small tables may later be saved to results/.

Not implemented yet: the model and scoring code does not exist, so after loading the
splits this prints a notice and exits with status 1. That keeps `make results` from
passing silently before the models are in.

Run it through make (`make results`) so that src/ is on PYTHONPATH.
"""

import sys
from pathlib import Path

import yaml

from listen import data

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        for seed in config["seeds"]:
            split = data.get_split(seed, config)
            print(
                f"seed {seed}: train {len(split.train.ids)}, "
                f"val {len(split.val.ids)}, test {len(split.test.ids)}"
            )
            # Remaining steps once the model code exists (README, "Splits and metrics"):
            #   fit the five models on split.train (passing split.train.mask), tune
            #   thresholds on split.val, evaluate on split.test once, then print the
            #   table with bootstrap CIs.
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))

    # Always fail until the model code exists, so a green `make results` means results.
    sys.exit("Model and scoring code is not implemented yet.")


if __name__ == "__main__":
    main()
