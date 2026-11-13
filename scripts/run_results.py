#!/usr/bin/env python3
"""Fit every model and print the results table (the driver behind `make results`).

Input:  configs/default.yaml and the CSVs in data/ (run `make data` first).
Output: the results table on stdout. Small tables may later be saved to results/.

Not implemented yet: until the model and scoring code exists, this prints a notice
and exits with status 1.

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
        data.load_dataset(config)
        # Remaining steps once the stand-ins are filled in (README, "Splits and metrics"):
        #   for each seed: split, fit the five models, tune thresholds on validation,
        #   evaluate on the test split once, then print the table with bootstrap CIs.
    except NotImplementedError:
        sys.exit("Model and scoring code is not implemented yet.")


if __name__ == "__main__":
    main()
