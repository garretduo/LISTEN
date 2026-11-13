"""Run the lay-vs-legal minimal pairs through a model and measure how often its guess flips.

Each row of analysis/minimal_pairs.csv tells the same story twice: once in everyday words
(lay_text) and once in legal terms (legal_text). For every pair we record:

  top1_flip      the model's top-ranked category differs between the two versions
  intended_flip  the yes/no decision for the intended category differs
                 (probability above vs. below that category's threshold)
  intended_delta legal-version probability minus lay-version probability
                 for the intended category

Usage (from the repo root):
  python analysis/run_minimal_pairs.py --stand-in
  python analysis/run_minimal_pairs.py --model artifacts/tfidf_0.joblib --name tfidf

ASSUMED MODEL INTERFACE -- confirm with Sohum and Garret, then adjust load_model() only:
  model.predict_proba(list_of_texts) -> numpy array of shape (n_texts, 20)
  model.categories                   -> list of the 20 category names, in column order
  model.thresholds                   -> array of 20 cutoffs (falls back to 0.5 if missing)
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS_FILE = REPO_ROOT / "analysis" / "minimal_pairs.csv"
RESULTS_DIR = REPO_ROOT / "results"

# Placeholder category names, used only by the stand-in model. Replace with the
# real CATEGORIES list from src/listen/data.py once Garret's loader lands.
PLACEHOLDER_CATEGORIES = [
    "housing", "employment", "family", "consumer", "money_debt", "torts",
    "traffic", "criminal", "immigration", "benefits", "estates", "education",
    "health", "business", "courts", "government", "disaster", "environment",
    "public_rights", "other",
]

# If the real category names differ from the names in minimal_pairs.csv,
# map them here, e.g. {"money_debt": "Money, debt and consumer finance"}.
CATEGORY_ALIASES = {}


class StandInModel:
    """Fake model that returns fixed pseudo-random probabilities for each text.

    The same text always gets the same output, so the script's logic can be
    tested before any real model exists. Its flip rates mean nothing.
    """

    def __init__(self):
        self.categories = PLACEHOLDER_CATEGORIES
        self.thresholds = np.full(len(self.categories), 0.5)

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).random(len(self.categories)))
        return np.vstack(rows)


def load_model(path):
    """Load a saved model. Adjust this function if the agreed interface differs."""
    import joblib

    model = joblib.load(path)
    if not hasattr(model, "thresholds"):
        print("Warning: model has no thresholds; using 0.5 for every category.")
        model.thresholds = np.full(len(model.categories), 0.5)
    return model


def category_index(name, categories):
    """Find the column for a category name, case-insensitively. Returns None if absent."""
    target = CATEGORY_ALIASES.get(name, name).strip().lower()
    for i, cat in enumerate(categories):
        if cat.strip().lower() == target:
            return i
    return None


def run(model, pairs):
    p_lay = model.predict_proba(pairs["lay_text"].tolist())
    p_legal = model.predict_proba(pairs["legal_text"].tolist())
    thresholds = np.asarray(model.thresholds)
    categories = list(model.categories)

    rows = []
    for k, pair in enumerate(pairs.itertuples(index=False)):
        top_lay = categories[int(np.argmax(p_lay[k]))]
        top_legal = categories[int(np.argmax(p_legal[k]))]
        row = {
            "pair_id": pair.pair_id,
            "intended_category": pair.intended_category,
            "top1_lay": top_lay,
            "top1_legal": top_legal,
            "top1_flip": top_lay != top_legal,
            "p_intended_lay": np.nan,
            "p_intended_legal": np.nan,
            "intended_delta": np.nan,
            "intended_flip": np.nan,
        }
        j = category_index(pair.intended_category, categories)
        if j is not None:
            yes_lay = p_lay[k, j] >= thresholds[j]
            yes_legal = p_legal[k, j] >= thresholds[j]
            row.update(
                p_intended_lay=round(float(p_lay[k, j]), 4),
                p_intended_legal=round(float(p_legal[k, j]), 4),
                intended_delta=round(float(p_legal[k, j] - p_lay[k, j]), 4),
                intended_flip=bool(yes_lay != yes_legal),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(results, name):
    matched = results.dropna(subset=["intended_flip"])
    return {
        "model": name,
        "n_pairs": len(results),
        "n_matched_category": len(matched),
        "top1_flip_rate": round(results["top1_flip"].mean(), 3),
        "intended_flip_rate": round(matched["intended_flip"].astype(bool).mean(), 3)
        if len(matched) else np.nan,
        "mean_intended_delta": round(matched["intended_delta"].mean(), 4)
        if len(matched) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stand-in", action="store_true", help="use the fake random model")
    group.add_argument("--model", type=Path, help="path to a saved model (.joblib)")
    parser.add_argument("--name", help="label for the output files (default: model file name)")
    args = parser.parse_args()

    if args.stand_in:
        model, name = StandInModel(), "stand_in"
    else:
        model, name = load_model(args.model), args.name or args.model.stem

    pairs = pd.read_csv(PAIRS_FILE)
    results = run(model, pairs)
    missing = results.loc[results["intended_flip"].isna(), "intended_category"].unique()
    if len(missing):
        print(f"Warning: categories not found in model, add to CATEGORY_ALIASES: {list(missing)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"minimal_pairs_{name}.csv"
    results.to_csv(out_file, index=False)

    summary = summarize(results, name)
    for key, value in summary.items():
        print(f"{key:>22}: {value}")
    print(f"Per-pair results saved to {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()