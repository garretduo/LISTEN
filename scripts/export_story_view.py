"""Write the files behind the app's data page, so it can show the stories without the raw dataset.

Input:  the CSVs in data/ (run `make data`), the frozen splits in results/splits/, and the
        per-story scores in results/scores/.
Output: artifacts/story_view/stories.csv.gz   cleaned text, labels (1 yes, 0 no, blank unknown)
                                              and each story's part (train, val, test) per seed
        artifacts/story_view/scores.csv.gz    each model's score per category, for the val and
                                              test stories of each seed
        artifacts/story_view/cutoffs.csv.gz   the cutoff per model and category that turns a
                                              score into a yes (tuned on that seed's val part)
Only cleaned text is written, so usernames and links never leave this machine.

Run from the repo root:
    python scripts/export_story_view.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # so `import listen` works without make

import numpy as np
import pandas as pd

from listen.data import PARTS, get_split, load_config
from listen.thresholds import tune_thresholds

OUT_DIR = ROOT / "artifacts" / "story_view"
MODELS = ["class_prior", "rules", "tfidf", "emb_lr", "emb_mlp"]
MIN_LIKELY = 0.10  # same floor as the classify page: a lower score is never a yes


def main():
    seeds = load_config()["seeds"]
    splits = {seed: get_split(seed) for seed in seeds}
    categories = splits[seeds[0]].categories
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Every seed splits the same stories, so the text and labels come from one seed.
    ids, texts, labels = [], [], []
    for name in PARTS:
        part = getattr(splits[seeds[0]], name)
        ids += part.ids
        texts += part.texts
        labels += list(np.where(part.mask == 1, part.Y, np.nan))
    stories = pd.DataFrame(labels, columns=categories).astype("Int8")
    stories.insert(0, "text", texts)
    stories.insert(0, "id", ids)
    for seed in seeds:
        part_of = {i: name for name in PARTS for i in getattr(splits[seed], name).ids}
        stories[f"seed_{seed}"] = stories["id"].map(part_of)
    stories.sort_values("id").to_csv(OUT_DIR / "stories.csv.gz", index=False)

    # A model says yes where its score reaches the category's cutoff, tuned on that seed's val part.
    score_frames, cutoff_rows = [], []
    for seed in seeds:
        split = splits[seed]
        for model in MODELS:
            scores = pd.read_csv(ROOT / "results" / "scores" / f"seed_{seed}" / f"{model}.csv", dtype={"_id": str})
            val = scores[scores["part"] == "val"]
            cutoffs = np.maximum(tune_thresholds(split.val.Y, val[categories].to_numpy(), split.val.mask), MIN_LIKELY)
            frame = scores.drop(columns="part").rename(columns={"_id": "id"})
            frame.insert(0, "model", model)
            frame.insert(0, "seed", seed)
            score_frames.append(frame)
            cutoff_rows.append([seed, model, *cutoffs])
    scores = pd.concat(score_frames)
    scores.to_csv(OUT_DIR / "scores.csv.gz", index=False)
    pd.DataFrame(cutoff_rows, columns=["seed", "model", *categories]).to_csv(OUT_DIR / "cutoffs.csv.gz", index=False)

    print(f"Wrote {len(stories):,} stories and {len(scores):,} score rows to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
