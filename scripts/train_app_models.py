"""Train the models the app shows and save them to the artifacts/app/

Each model is fit on the seed 0's train part, and its trhesholds are tuned on seed 0's
val part. The model, its thresholds and the categoruy names are saved together
in one file. The test part is never used, so does not touch final eval.

Input:  the CSVs in data/ (run `make data`), the frozen splits in results/splits/,
        and data/glove.6B.50d.txt (GloVe vectors for the neural network).
Output: artifacts/app/<key>.joblib for each model the app offers (gitignored).

Run from the repo root:
    python scripts/train_app_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # so `import listen` works without make

import joblib  

from listen.baselines import RuleBasedBaseline  
from listen.data import get_split  
from listen.models.embedding_mlp import EmbeddingMLP  
from listen.models.glove import load_glove  
from listen.models.tfidf import TfidfLogReg  
from listen.thresholds import tune_thresholds  

GLOVE_PATH = ROOT / "data" / "glove.6B.50d.txt"
OUT_DIR = ROOT / "artifacts" / "app"
SEED = 0


def main():
    if not GLOVE_PATH.exists():
        sys.exit(f"Missing {GLOVE_PATH.relative_to(ROOT)}. Run `make glove` to download it.")

    split = get_split(SEED)
    word_vectors = load_glove(GLOVE_PATH)

    # Keys must match the MODELS dictionary in app/streamlit_app.py.
    models = {
        "rules": RuleBasedBaseline(split.categories),
        "tfidf": TfidfLogReg(),
        "emb_mlp": EmbeddingMLP(word_vectors, seed=SEED),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        print(f"Training {key} on seed {SEED}...", flush=True)
        model.fit(split.train.texts, split.train.Y, split.train.mask)

        # Thresholds come from the validation part only, never the test part.
        val_scores = model.predict_proba(split.val.texts)
        thresholds = tune_thresholds(split.val.Y, val_scores, split.val.mask)

        path = OUT_DIR / f"{key}.joblib"
        joblib.dump(
            {
                "model": model,
                "thresholds": thresholds,
                "categories": list(split.categories),
                "seed": SEED,
            },
            path,
        )
        print(f"  saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()