#!/usr/bin/env python3
"""Measure how much the raters agree on the shared mistakes.

Usage: python analysis/agreement.py artifacts/label_sheets/garret.csv artifacts/label_sheets/lyndon.csv ...

Only mistakes that every given sheet has a `cause` for are used, which is the shared set.
Prints the share of mistakes all raters agree on, Cohen's kappa for each pair, and Fleiss'
kappa for the whole group with a 95% bootstrap interval over mistakes.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causes import CAUSES  # noqa: E402
from listen.evaluate import fleiss_kappa  # noqa: E402


def main():
    files = sys.argv[1:]
    if len(files) < 2:
        sys.exit("Give at least two labeled sheets.")
    sheets = [pd.read_csv(f, encoding="utf-8-sig").set_index("item")["cause"].rename(Path(f).stem) for f in files]
    causes = pd.concat(sheets, axis=1, join="inner").dropna()
    causes = causes.apply(lambda col: col.astype(str).str.strip().str.lower())
    unknown = set(causes.to_numpy().ravel()) - set(CAUSES)
    if unknown:
        sys.exit(f"Unknown cause(s) {sorted(unknown)}. Use the codes in causes.txt.")

    labels = causes.to_numpy()
    print(f"{len(labels)} shared mistakes labeled by {', '.join(causes.columns)}")
    print(f"All raters agree on {np.mean((labels == labels[:, :1]).all(1)):.0%}")
    for a, b in itertools.combinations(causes.columns, 2):
        print(f"{a} vs {b}: agree {np.mean(causes[a] == causes[b]):.0%}, "
              f"Cohen's kappa {cohen_kappa_score(causes[a], causes[b]):.2f}")
    rng = np.random.default_rng(0)
    boot = [fleiss_kappa(labels[rng.integers(0, len(labels), len(labels))]) for _ in range(1000)]
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    print(f"Fleiss' kappa {fleiss_kappa(labels):.2f} (95% bootstrap interval {low:.2f} to {high:.2f})")


if __name__ == "__main__":
    main()
