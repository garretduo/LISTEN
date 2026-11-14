#!/usr/bin/env python3
"""Pick the TF-IDF mistakes the team will label.

A mistake is one (story, category) cell, among the scored categories, where TF-IDF's
guess is wrong and the label is known. The sample is half missed categories and half false
alarms. The first --shared mistakes are labeled by everyone so we can measure agreement;
the rest are split between the raters. The embedding models' results on the same cells are
recorded here but kept off the sheets, so raters don't see them.

Input:  per-story probabilities and cutoffs from the final `make results` run
        (`--part val` reads the development run instead).
Output: analysis/mistake_sample.csv, which has ids only and can be committed. A `--part val`
        run writes artifacts/mistake_sample_val.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from listen import data  # noqa: E402
from listen.thresholds import apply_thresholds  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Sample the TF-IDF mistakes to label.")
    parser.add_argument("--part", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--shared", type=int, default=30, help="mistakes every rater labels")
    parser.add_argument("--raters", nargs="+", default=["garret", "lyndon", "sohum"])
    args = parser.parse_args()

    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    final = args.part == "test"
    results, artifacts = (ROOT / config["paths"][k] for k in ("results_dir", "artifacts_dir"))
    split = data.get_split(args.seed, config)
    part = getattr(split, args.part)
    categories = np.array(split.categories)
    cutoffs = pd.read_csv(results / ("per_category.csv" if final else "per_category_val.csv"))

    def guess(model):
        """Return the model's probabilities and its yes/no guesses for this part."""
        scores = pd.read_csv(
            (results if final else artifacts) / "scores" / f"seed_{args.seed}" / f"{model}.csv",
            dtype={"_id": str},
        )
        scores = scores[scores["part"] == args.part].set_index("_id").loc[part.ids]
        rows = cutoffs[(cutoffs["model"] == model) & (cutoffs["seed"] == args.seed)]
        t = rows.set_index("category").loc[split.categories, "threshold"].to_numpy()
        prob = scores[split.categories].to_numpy()
        return prob, apply_thresholds(prob, t)

    scored = ~np.isin(categories, config["evaluation"]["excluded_categories"])
    prob, tfidf = guess("tfidf")
    rows, cols = np.nonzero((tfidf != part.Y) & (part.mask == 1) & scored)
    mistakes = pd.DataFrame({
        "story_id": np.array(part.ids)[rows],
        "category": categories[cols],
        "kind": np.where(part.Y[rows, cols] == 1, "missed", "false_alarm"),
        "tfidf_prob": prob[rows, cols].round(4),
        "tfidf_guess": ["; ".join(categories[tfidf[r] == 1]) for r in rows],
    })
    for model in ("emb_lr", "emb_mlp"):
        mistakes[f"{model}_right"] = (guess(model)[1] == part.Y)[rows, cols]

    # Half of each kind; if one kind runs short, fill up from the rest.
    rng = np.random.default_rng(args.seed)
    picked = pd.concat([g.sample(min(len(g), args.n // 2), random_state=rng)
                        for _, g in mistakes.groupby("kind")])
    rest = mistakes.drop(picked.index)
    picked = pd.concat([picked, rest.sample(min(len(rest), args.n - len(picked)), random_state=rng)])
    picked = picked.sample(frac=1, random_state=rng).reset_index(drop=True)
    picked.insert(0, "item", range(1, len(picked) + 1))
    picked["rater"] = ["all" if i < args.shared else args.raters[(i - args.shared) % len(args.raters)]
                       for i in range(len(picked))]

    out = ROOT / "analysis" / "mistake_sample.csv" if final else artifacts / "mistake_sample_val.csv"
    picked.to_csv(out, index=False)
    print(f"{len(mistakes)} mistakes on {args.part}; picked {len(picked)} -> {out.relative_to(ROOT)}")
    print(picked.groupby(["rater", "kind"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
