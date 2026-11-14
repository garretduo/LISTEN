#!/usr/bin/env python3
"""Fit the five models, tune thresholds on val, score the test part, print the results table.

Input:  configs/default.yaml, the CSVs in data/ (run `make data`), the frozen splits in
        results/splits/, and data/glove.6B.50d.txt (GloVe vectors for the embedding models;
        `make results` downloads it if missing).
Output: a table on stdout (per model and seed with 95% bootstrap intervals, then the mean
        over the five seeds) and results/per_category.csv (threshold, precision, recall and
        F1 per category). Macro-F1 and micro-F1 use the categories not listed under
        `evaluation.excluded_categories`; recall@3 ranks all 20. Also scores/seed_N/<model>.csv
        with every scored story's probabilities (ids only), written under results/ for a test
        run, so the final ones can be committed, and under artifacts/ for `--part val`. And,
        under artifacts/ (gitignored), for seed 0 <model>_0.joblib holding
        {"model", "thresholds"}, which the app loads.
        Models are named class_prior, rules, tfidf, emb_lr, emb_mlp.

By default this scores the test part, so run it only once model choices are frozen.
`--part val` scores the validation part instead and never predicts on test. Those scores
are optimistic (the cutoffs were tuned on the same stories), but the saved models and
probabilities are the same, so use it while building. It writes per_category_val.csv.
Run it through make (`make results`) so that src/ is on PYTHONPATH.
"""

import argparse
import csv
import sys
from pathlib import Path

import joblib
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


def build_models(categories, word_vectors, seed):
    return {
        "class_prior": ClassPriorBaseline(),
        "rules": RuleBasedBaseline(categories),
        "tfidf": TfidfLogReg(),
        "emb_lr": EmbeddingLogReg(word_vectors),
        "emb_mlp": EmbeddingMLP(word_vectors, seed=seed),
    }


def score(model, split, keep, seed, part="test"):
    """Fit on train, tune thresholds on val, score `part`. Returns {metric: (value, low, high)}
    for macro, micro and recall@3, plus "prf" (per-category precision, recall, F1), "thresholds"
    and "probs" ({part name: probabilities} for val and `part`)."""
    model.fit(split.train.texts, split.train.Y, split.train.mask)
    val_prob = model.predict_proba(split.val.texts)
    thresholds = tune_thresholds(split.val.Y, val_prob, split.val.mask)
    target = getattr(split, part)
    prob = val_prob if part == "val" else model.predict_proba(target.texts)
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
        "thresholds": thresholds,
        "probs": {"val": val_prob, part: prob},
    }


def save_artifacts(artifacts, scores_dir, name, seed, split, model, r):
    """Write one model's per-story probabilities to scores_dir, and for seed 0 the model the
    app loads to artifacts."""
    out = scores_dir / "scores" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["_id", "part", *split.categories])
        for part, probs in r["probs"].items():
            for story_id, row in zip(getattr(split, part).ids, probs):
                writer.writerow([story_id, part, *np.round(row, 6)])
    if seed == 0:  # the app loads artifacts/<name>_0.joblib
        joblib.dump({"model": model, "thresholds": r["thresholds"]}, artifacts / f"{name}_0.joblib")


def fmt(value, low, high):
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def main():
    parser = argparse.ArgumentParser(description="Fit the five models and print the results table.")
    parser.add_argument("--part", choices=["val", "test"], default="test", help="part to score (default: test)")
    part = parser.parse_args().part
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    glove_path = ROOT / config["paths"]["data_dir"] / "glove.6B.50d.txt"
    if not glove_path.exists():
        sys.exit(f"Missing {glove_path.relative_to(ROOT)}. Run `make glove` to download it.")

    excluded = set(config["evaluation"]["excluded_categories"])
    artifacts = ROOT / config["paths"]["artifacts_dir"]
    scores_dir = ROOT / config["paths"]["results_dir"] if part == "test" else artifacts
    word_vectors = load_glove(glove_path)
    results = {}  # (model, seed) -> score() output
    try:
        for seed in config["seeds"]:
            split = data.get_split(seed, config)
            keep = np.array([c not in excluded for c in split.categories])
            for name, model in build_models(split.categories, word_vectors, seed).items():
                print(f"seed {seed}: {name}", file=sys.stderr, flush=True)
                results[name, seed] = score(model, split, keep, seed, part)
                save_artifacts(artifacts, scores_dir, name, seed, split, model, results[name, seed])
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))

    names = list(dict.fromkeys(name for name, _ in results))
    note = "" if part == "test" else " Cutoffs were tuned on these stories, so scores are optimistic."
    print(f"{part.capitalize()} part.{note} Macro/micro-F1 over {keep.sum()} categories, excluding: {', '.join(sorted(excluded))}.")
    print(f"{'model':12} {'seed':>4}  {'macro-F1':23} {'micro-F1':23} recall@3")
    for name in names:
        for seed in config["seeds"]:
            r = results[name, seed]
            print(f"{name:12} {seed:>4}  {fmt(*r['macro']):23} {fmt(*r['micro']):23} {fmt(*r['recall@3'])}")
        mean = {k: np.mean([results[name, s][k] for s in config["seeds"]], axis=0) for k in ("macro", "micro", "recall@3")}
        print(f"{name:12} {'mean':>4}  {fmt(*mean['macro']):23} {fmt(*mean['micro']):23} {fmt(*mean['recall@3'])}\n")
    print(f"Intervals: 95% bootstrap over {part} stories; the mean row averages the five seeds' intervals.")

    out = ROOT / config["paths"]["results_dir"] / ("per_category.csv" if part == "test" else f"per_category_{part}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "category", "threshold", "precision", "recall", "f1"])
        for (name, seed), r in results.items():
            for i, category in enumerate(split.categories):
                writer.writerow([name, seed, category, round(float(r["thresholds"][i]), 4), *("" if np.isnan(v[i]) else round(float(v[i]), 4) for v in r["prf"])])
    print(f"Per-category precision/recall/F1 written to {out.relative_to(ROOT)} (blank = undefined).")


if __name__ == "__main__":
    main()
