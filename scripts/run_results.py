#!/usr/bin/env python3
"""Fit the five models, tune thresholds on val, score the test part, print the results table.

Input:  configs/default.yaml, the CSVs in data/ (run `make data`), the frozen splits in
        results/splits/, and data/glove.6B.50d.txt (GloVe vectors for the embedding models).
Output: a table on stdout (per model and seed with 95% bootstrap intervals, then the mean
        over the five seeds) and results/per_category.csv (precision, recall, F1 per
        category). Macro-F1 and micro-F1 use the categories not listed under
        `evaluation.excluded_categories`; recall@3 ranks all 20.

Running this scores the test part, so run it only once model choices are frozen.
Run it through make (`make results`) so that src/ is on PYTHONPATH.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

from listen import data
from listen.baselines import ClassPriorBaseline, RuleBasedBaseline
from listen.evaluate import bootstrap_ci, macro_micro_f1, per_category_prf, recall_at_k
from listen.models.embedding_lr import EmbeddingLogReg
from listen.models.embedding_mlp import EmbeddingMLP
from listen.models.glove import load_glove
from listen.models.tfidf import TfidfLogReg
from listen.thresholds import apply_thresholds, tune_thresholds

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
GLOVE_PATH = ROOT / "data" / "glove.6B.50d.txt"


def build_models(categories, word_vectors, seed):
    return {
        "class prior": ClassPriorBaseline(),
        "rules": RuleBasedBaseline(categories),
        "TF-IDF + LR": TfidfLogReg(),
        "embeddings + LR": EmbeddingLogReg(word_vectors),
        "embeddings + MLP": EmbeddingMLP(word_vectors, seed=seed),
    }


def score(model, split, keep, seed, part="test"):
    """Fit on train, tune thresholds on val, score `part`. Returns {metric: (value, low, high)}
    for macro, micro and recall@3, plus "prf" (per-category precision, recall, F1)."""
    model.fit(split.train.texts, split.train.Y, split.train.mask)
    thresholds = tune_thresholds(split.val.Y, model.predict_proba(split.val.texts), split.val.mask)
    target = getattr(split, part)
    prob = model.predict_proba(target.texts)
    pred = apply_thresholds(prob, thresholds)
    y, m = target.Y, target.mask
    ys, ps, ms = y[:, keep], pred[:, keep], m[:, keep]

    def with_ci(metric, *args):
        return (metric(*args), *bootstrap_ci(metric, *args, seed))

    return {
        "macro": with_ci(lambda t, p, k: macro_micro_f1(t, p, k)[0], ys, ps, ms),
        "micro": with_ci(lambda t, p, k: macro_micro_f1(t, p, k)[1], ys, ps, ms),
        "recall@3": with_ci(recall_at_k, y, prob, m),
        "prf": per_category_prf(y, pred, m),
    }


def fmt(value, low, high):
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not GLOVE_PATH.exists():
        sys.exit(f"Missing {GLOVE_PATH.relative_to(ROOT)} (GloVe vectors for the embedding models).")

    excluded = set(config["evaluation"]["excluded_categories"])
    word_vectors = load_glove(GLOVE_PATH)
    results = {}  # (model, seed) -> score() output
    try:
        for seed in config["seeds"]:
            split = data.get_split(seed, config)
            keep = np.array([c not in excluded for c in split.categories])
            for name, model in build_models(split.categories, word_vectors, seed).items():
                print(f"seed {seed}: {name}", file=sys.stderr, flush=True)
                results[name, seed] = score(model, split, keep, seed)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))

    names = list(dict.fromkeys(name for name, _ in results))
    print(f"Test part. Macro/micro-F1 over {keep.sum()} categories, excluding: {', '.join(sorted(excluded))}.")
    print(f"{'model':18} {'seed':>4}  {'macro-F1':23} {'micro-F1':23} recall@3")
    for name in names:
        for seed in config["seeds"]:
            r = results[name, seed]
            print(f"{name:18} {seed:>4}  {fmt(*r['macro']):23} {fmt(*r['micro']):23} {fmt(*r['recall@3'])}")
        mean = {k: np.mean([results[name, s][k] for s in config["seeds"]], axis=0) for k in ("macro", "micro", "recall@3")}
        print(f"{name:18} {'mean':>4}  {fmt(*mean['macro']):23} {fmt(*mean['micro']):23} {fmt(*mean['recall@3'])}\n")
    print("Intervals: 95% bootstrap over test stories; the mean row averages the five seeds' intervals.")

    out = ROOT / config["paths"]["results_dir"] / "per_category.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "category", "precision", "recall", "f1"])
        for (name, seed), r in results.items():
            for i, category in enumerate(split.categories):
                writer.writerow([name, seed, category, *("" if np.isnan(v[i]) else round(float(v[i]), 4) for v in r["prf"])])
    print(f"Per-category precision/recall/F1 written to {out.relative_to(ROOT)} (blank = undefined).")


if __name__ == "__main__":
    main()
