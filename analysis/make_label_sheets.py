#!/usr/bin/env python3
"""Write one labeling sheet per rater from the mistake sample.

Each sheet has the shared mistakes plus that rater's share. For every row, put one code from
causes.txt in `cause` and use `notes` for anything else. `kind` is "missed" when the story has
the category and TF-IDF didn't flag it, and "false_alarm" when TF-IDF flagged it but the label
says no. The sheets contain story text, so they go under artifacts/ and stay out of git and
off any outside service.

Usage: python analysis/make_label_sheets.py
       python analysis/make_label_sheets.py --part val --sample artifacts/mistake_sample_val.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causes import CAUSES  # noqa: E402
from listen import data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Write the labeling sheets.")
    parser.add_argument("--part", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", default=str(ROOT / "analysis" / "mistake_sample.csv"))
    args = parser.parse_args()

    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    split = data.get_split(args.seed, config)
    part = getattr(split, args.part)
    categories = np.array(split.categories)
    row_of = {story: i for i, story in enumerate(part.ids)}

    sample = pd.read_csv(args.sample, dtype={"story_id": str})
    rows = [row_of[s] for s in sample["story_id"]]
    sample["true_labels"] = ["; ".join(categories[(part.Y[r] == 1) & (part.mask[r] == 1)]) for r in rows]
    sample["text"] = [part.texts[r] for r in rows]
    sheet = sample[["item", "category", "kind", "true_labels", "tfidf_guess", "text"]].assign(cause="", notes="")

    out = ROOT / config["paths"]["artifacts_dir"] / "label_sheets"
    out.mkdir(parents=True, exist_ok=True)
    for rater in sorted(set(sample["rater"]) - {"all"}):
        # utf-8-sig so Excel reads accents correctly
        sheet[sample["rater"].isin(["all", rater])].to_csv(out / f"{rater}.csv", index=False, encoding="utf-8-sig")
    (out / "causes.txt").write_text("\n".join(f"{c}: {d}" for c, d in CAUSES.items()) + "\n", encoding="utf-8")
    print(f"Wrote {len(set(sample['rater']) - {'all'})} sheets and causes.txt to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
