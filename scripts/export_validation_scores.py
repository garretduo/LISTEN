"""Train the five LISTEN models and export their validation scores."""

import argparse
import csv
from pathlib import Path

import numpy as np

from listen.baselines import ClassPriorBaseline, RuleBasedBaseline
from listen.data import get_split
from listen.models.embedding_lr import EmbeddingLogReg
from listen.models.embedding_mlp import EmbeddingMLP
from listen.models.glove import load_glove
from listen.models.tfidf import TfidfLogReg


def export_seed(seed, word_vectors):
    split = get_split(seed)
    train = split.train
    validation = split.val
    output_dir = Path("artifacts/validation_scores") / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("class_prior", ClassPriorBaseline()),
        ("keyword", RuleBasedBaseline(split.categories)),
        ("tfidf", TfidfLogReg()),
        ("embedding_lr", EmbeddingLogReg(word_vectors)),
        ("embedding_mlp", EmbeddingMLP(word_vectors, seed=seed)),
    ]

    for name, model in models:
        print(f"Running seed {seed}: {name}", flush=True)
        model.fit(train.texts, train.Y, train.mask)
        scores = model.predict_proba(validation.texts)

        expected_shape = (len(validation.ids), len(split.categories))
        if scores.shape != expected_shape or not np.isfinite(scores).all():
            raise ValueError(f"Seed {seed}: {name} returned invalid scores.")

        # Keep the story IDs and category columns in the official split order.
        path = output_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["_id", *split.categories])
            for story_id, row in zip(validation.ids, scores):
                writer.writerow([story_id, *row])

        print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=int, nargs="+")
    args = parser.parse_args()

    # Load GloVe once and reuse it for every requested seed.
    print("Loading GloVe vectors...", flush=True)
    word_vectors = load_glove("data/glove.6B.50d.txt")
    for seed in args.seeds:
        export_seed(seed, word_vectors)


if __name__ == "__main__":
    main()