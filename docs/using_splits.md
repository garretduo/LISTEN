# Using the train/val/test splits

This page is for teammates who train, tune or evaluate a model (Sohum) or who wire the cleaning into the app (Lyndon). It tells you how to get the stories for a seed, what the arrays mean, and which mistakes to avoid. Commands and numbers were run or measured on 2026-11-13 unless a line says otherwise.

The short version:

1. `from listen.data import get_split`, then `split = get_split(0)` (seeds 0 to 4). This only works once `listen` is importable; see [Making `listen` importable](#making-listen-importable) right below.
2. Fit on `split.train`, tune thresholds on `split.val`, look at `split.test` once at the very end.
3. Every label array comes with a `mask`. Where the mask is 0, the label is unknown and the 0 in `Y` does NOT mean "no". Always use the mask.
4. Story texts are already cleaned. Never clean them again.
5. The app never needs `data/` or `get_split`. It only needs `clean_text` (see [For the app](#for-the-app-lyndon)).

## Making `listen` importable

The package lives in `src/` and is not installed with pip. In a plain script or notebook, `import listen` stops with `ModuleNotFoundError: No module named 'listen'` unless `src/` is on the Python path. Checked 2026-11-13:

| Where you run your code | What to do |
|---|---|
| A script in the terminal | From the repo root: `PYTHONPATH=src python your_script.py` |
| A notebook or the Python shell | Before any `listen` import, run `import sys; sys.path.insert(0, "<full path to the repo>/src")`. A relative `"src"` works only if the notebook was started in the repo root. |
| An editor such as VS Code or PyCharm | Mark the `src` folder as a source root, or add it to the editor's extra Python paths. |
| The `make` targets and `pytest` | Nothing to do: the Makefile and `pytest.ini` set the path for you. |

Once `listen` imports, `get_split` finds the data and the frozen split files by itself, from any working directory.

## Words used on this page

| Term | Meaning |
|---|---|
| Story | One Reddit post from the Learned Hands best-guess file (3,459 of them). |
| Category | One of the 20 LIST categories, for example Housing. A story can have several. |
| Positive | A story labeled 1 for a category (the crowd said the category applies). |
| Omitted cell | A label cell left empty in the data file. It is **unknown**, not negative. About 48.6% of all cells are omitted (`docs/data_notes.md`). |
| Mask | An array of the same shape as the labels with 1 where the label is known and 0 where the cell was omitted. |
| Part | One of train, val (validation) or test: a group of stories. |
| Seed | A number that picks one random 80/10/10 division of the stories into parts. We use seeds 0, 1, 2, 3 and 4 and report the average over them. |
| Frozen split | A division that was computed once and saved as lists of story ids in `results/splits/`, so everyone gets exactly the same parts and a library update cannot change them. |
| Near-duplicate group | Stories that are reposts of each other (at least 80% of their 5-word shingles match). A group is always kept in one part, so a model is never tested on a copy of a story it trained on. |

## What you get back

`get_split(seed)` returns a `Split`. A `Split` holds three `Part` objects and the category names. All of them are plain Python lists and numpy arrays; the dataclasses are frozen (you cannot reassign a field), but the lists and arrays inside can be changed, so copy them before editing in place.

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `split.seed` | `int` | | The seed you asked for. |
| `split.categories` | `list[str]` | 20 | Category names, in the column order of every `Y` and `mask`. |
| `split.train`, `split.val`, `split.test` | `Part` | | The three parts. Disjoint, and together they cover every story. |
| `part.ids` | `list[str]` | `(n,)` | Story ids (24-character lowercase hex), sorted in plain string order. |
| `part.texts` | `list[str]` | `(n,)` | Cleaned story texts, aligned with `ids`. Already passed through `clean_text` once. |
| `part.Y` | `numpy int8` | `(n, 20)` | 1 = positive. 0 = negative **or unknown**; tell them apart with `mask`. Always 0 where `mask` is 0. |
| `part.mask` | `numpy int8` | `(n, 20)` | 1 = label known, 0 = label unknown (omitted cell). Pass it to `MultiLabelModel.fit`. |

Row `i` of `texts`, `Y` and `mask` is story `ids[i]` in all three. The part sizes differ a little from seed to seed (see the FAQ).

### Example: get a split and check its shape

<!-- tested -->
```python
import numpy as np

from listen.data import get_split

split = get_split(0)  # seeds 0, 1, 2, 3, 4 are available
assert len(split.categories) == 20

all_ids = []
for name in ("train", "val", "test"):
    part = getattr(split, name)
    n = len(part.ids)
    assert len(part.texts) == n
    assert part.Y.shape == (n, 20) and part.mask.shape == (n, 20)
    assert part.Y.dtype == np.int8 and part.mask.dtype == np.int8
    assert part.ids == sorted(part.ids)  # rows are in _id order
    assert (part.Y[part.mask == 0] == 0).all()  # unknown cells are stored as 0 in Y
    known = int(part.mask.sum())
    print(f"{name}: {n} stories, {known} of {n * 20} label cells known")
    all_ids += part.ids

assert len(set(all_ids)) == len(all_ids)  # no story is in two parts
print(f"{len(all_ids)} stories in total")
```

The printed sizes are the same on every machine because the split is frozen. For seed 0 the output is:

```text
train: 2766 stories, 28476 of 55320 label cells known
val: 347 stories, 3505 of 6940 label cells known
test: 346 stories, 3569 of 6920 label cells known
3459 stories in total
```

## Rules that every user of the splits must follow

1. **Y is 0 where the mask is 0, and that 0 means unknown.** An integer array cannot hold "missing", so unknown cells are stored as 0. Treating `Y == 0` as "no" would mean treating about half of all cells as negatives. Fit, tune and score only on cells where `mask == 1`.
2. **Rows are sorted by `_id`.** A story's position does not depend on the order of the CSV. Do not rely on any other order; shuffle yourself (with your own seed) if your training needs it.
3. **The texts are already cleaned, so do not call `clean_text` on them.** `clean_text` takes RAW text and is called once. It is not guaranteed to give the same result when applied twice.
4. **Fit on train. Tune on val. Touch test once.** The test part is for the final evaluation, after every choice (features, hyperparameters, thresholds) is frozen. If you look at test scores and then change anything, the test result is no longer honest.
5. **Only seeds 0 to 4 exist.** They are listed in `configs/default.yaml` (`seeds`). Any other number raises an error.
6. **Do not split the data yourself and do not read the CSV in `data/` for splitting.** Use `get_split`. Stratifying again would put different stories in test than your teammates use, and the results could not be compared.
7. **The app never needs `data/`.** See the app section.

## Recipes for models (Sohum)

All five models share one interface, `listen.models.MultiLabelModel`: `fit(texts, Y, mask)` and `predict_proba(texts)`. Training looks like this:

```python
split = get_split(seed)
model.fit(split.train.texts, split.train.Y, split.train.mask)
val_proba = model.predict_proba(split.val.texts)   # shape (len(split.val.ids), 20)
```

Run it once for each seed and report the average over the five seeds. Read the seeds from the config (`from listen.data import load_config`; `load_config()["seeds"]`) instead of typing them.

### Logistic regression for one category

Category `c` is fit only on the training rows where `mask[:, c] == 1`. The other rows are not negative examples, they are stories nobody labeled for this category. The same rows (this category's labeled rows in val) are used for its threshold and its metrics.

The example below does this for one high-support category, Housing, with TF-IDF fit on `split.train`. It stops after predicting on val, because thresholds and metrics belong to the evaluation code.

<!-- tested -->
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from listen.data import get_split

split = get_split(0)
c = split.categories.index("Housing")

# Fit the vectorizer on training texts only; val is only transformed.
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform(split.train.texts)

# Keep only the training rows whose Housing label is known.
known = split.train.mask[:, c] == 1
y = split.train.Y[known, c]
assert set(np.unique(y)) == {0, 1}  # both classes present, so a classifier can be fit
clf = LogisticRegression(max_iter=1000).fit(X_train[known], y)

# Predict on val and score only the val rows whose Housing label is known.
val_known = split.val.mask[:, c] == 1
proba = clf.predict_proba(vectorizer.transform(split.val.texts))[:, 1]
print(
    f"Housing: fit on {int(known.sum())} of {len(split.train.ids)} train rows, "
    f"val has {len(split.val.ids)} stories, of which {int(val_known.sum())} are labeled for Housing"
)
assert proba.shape == (len(split.val.ids),)
```

For all 20 categories, repeat the same steps in a loop over `range(20)` and stack the 20 probability columns into one `(n, 20)` array. Before fitting a category, check that its kept labels contain both a 0 and a 1. A category with no positive in train cannot be fit (scikit-learn raises an error for a single class). For this data that is Native American Issues and Tribal Law, which has no positive story in the whole file (see the section on rare categories below).

### Neural network: weight the loss by the mask

The network outputs 20 probabilities per story. The loss must count only the known cells, as in the README:

```python
# bce has shape (n, 20): binary cross-entropy of every cell, unreduced.
# mask is split.train.mask (n, 20); convert it to the framework's float type.
loss = (bce * mask).sum() / max(float(mask.sum()), 1.0)
```

An omitted cell then adds nothing to the loss and sends no gradient. The deep-learning framework is not decided yet, so this is a sketch and not a tested function. Use the `Y` array as the target as it is: where the mask is 0 the target value never matters because the weight is 0.

### Thresholds: tune on val only

A threshold turns a probability into a yes or no. Choose one per category on `split.val`, using only the val rows where that category's mask is 1, and keep the value that maximizes F1 on them. Do not tune on train (the model has seen those stories) and never on test.

### Test: once, at the end

Compute test scores one time, after the models, features and thresholds are final. If you run into a bug afterwards and need to change something, write it up in `docs/decisions.md` and say that the test set was already looked at.

### Categories with no positives in a part

A category can have no positive story in a part: the whole file has only 3 Disaster Relief stories, 4 Veterans & Military, 8 Environmental Justice and none for Native American Issues (counts in `docs/data_notes.md`). In such a part:

- Precision, recall and F1 for that category are undefined (division by zero), and a threshold cannot be tuned on val.
- Check for this before computing a metric. Do not let a library silently turn it into 0 or 1.
- Deciding how to treat such categories (thresholds, macro-F1 with or without them, how to report them) is not part of the splits and is left to the evaluation sprint. It is written down in `docs/decisions.md`, "rare categories keep their column". Until that decision exists, do not report a macro-F1 that averages over undefined values.

## For the app (Lyndon)

The app cleans text a person pastes, then hands it to a model. It calls `clean_text` exactly once on the raw pasted text:

<!-- tested -->
```python
from listen.text import clean_text

pasted = "My landlord emailed me at https://example.org/notice and u/someone said it is fine.  [CA]"
story = clean_text(pasted)  # once, on the raw text
assert "https" not in story and "u/someone" not in story
assert story.endswith("[CA]")  # jurisdiction tags are kept on purpose
```

Points to know:

- **Call it once, on raw text.** Do not call it on text that has already been cleaned. The cleaned text of a dataset story (`part.texts`) must not be cleaned again either.
- `clean_text` takes a `str` and returns a `str`. Anything else (`None`, `bytes`, a number) raises a `TypeError`. A pasted empty box is the string `""`, which is fine.
- It does not lowercase, remove stopwords or stem. If a model wants lowercase, the model does it. This is on purpose: the same cleaned text goes to the TF-IDF model, to the embedding models and to the app.
- It removes links, `u/name`, `r/name` and `@name`, turns curly quotes and dashes into plain ones, fixes line endings and spaces, and keeps accents, emoji and tags like `[CA]`. The exact rules and the reasons are in `docs/decisions.md` ("text cleaning rules, version 1") and in the docstring of `src/listen/text.py`.
- **The app never needs `data/`, `get_split` or `results/splits/`.** Those are for training and evaluation. Importing `listen.text` imports no data code and reads no file. The app only needs the trained models and thresholds from the training step. The app must not log or store what people paste (README).

## Do and don't

**Do**

- Call `get_split(seed)` once per seed and keep the result in a variable. Each call re-reads the data file and re-cleans the texts, which takes about a second.
- Use `mask` whenever you fit, tune or score.
- Run all five seeds and report the average, as the README says.
- Tune thresholds on val and look at test once.
- Log any change to the splits, ratios, seeds or cleaning in `docs/decisions.md`.
- Run `make test` before you push changes to the splits or the cleaning.

**Don't**

- Don't treat `Y == 0` as a negative label.
- Don't call `clean_text` on `part.texts`, and don't clean a text twice.
- Don't re-split the data or call the stratifier yourself (`listen.data.make_splits` recomputes a fresh split that may differ from the frozen one; it is for experiments and tests, never for results).
- Don't edit the files in `results/splits/` by hand.
- Don't choose a model, feature set or threshold by looking at test scores.
- Don't print, save or paste story text into commits, chat or reports. The data is CC BY-NC-SA 4.0 and must not be redistributed. Print ids, counts and shapes instead.
- Don't commit anything from `data/` (it is gitignored; never `git add -f` it).

## Errors, warnings and what to do

Each message below is raised by the code and contains no story text.

| You see | Raised by | What it means and the fix |
|---|---|---|
| `ValueError: seed must be an integer in config['seeds'] = [0, 1, 2, 3, 4], got 7. Only these seeds have frozen splits.` | `get_split` | You asked for a seed that has no frozen split (or passed a `bool`, float or string). Use one of the listed seeds. To add a seed, see the FAQ. |
| `` FileNotFoundError: The data file ... is missing. Run `make data` to download it. `` | `get_split`, `load_dataset` | `data/` has no CSV. Run `make data`. Tests that need the data are skipped in this case. |
| `` FileNotFoundError: Frozen split file(s) missing: ... Run `make splits` to create them (or restore them from git). `` | `get_split` | `results/splits/` is missing or incomplete. Restore the files from git (`git checkout -- results/splits` restores them from the last commit) or run `make splits` only if they were never created. |
| `ValueError: The story ids in the data (...) and in the frozen files (...) differ ...` | `get_split` | The data file and the frozen files describe different stories (a different data download, or edited files). Restore the right ones. Regenerate only if the change is intended: `PYTHONPATH=src python scripts/make_splits.py --force`, then log it in `docs/decisions.md`. |
| `ValueError: The category columns in the frozen meta file differ from the data file's header ...` | `get_split` | Same cause and fix as the previous row, for the category columns. |
| `ValueError: seed 5 is in the config but not in the frozen files ...` | `get_split` | Someone added a seed to `configs/default.yaml` without regenerating. Regenerate with `--force` and log it. |
| `ValueError: ... is not valid JSON` or `... lacks the keys [...]` or `...: the header must be the id column followed by seed_<n> columns` or `... data row ... must hold an id and ... part names` | `get_split` | A frozen file is damaged. Restore it from git, or regenerate with `--force` if damage came from an intended change. |
| `ValueError: ... data row N (0-based, header not counted), column '...': expected '', '0.0' or '1.0'` and other `ValueError`s from `load_dataset` (wrong header, wrong number of columns, repeated ids) | `load_dataset`, `get_split` | The CSV is damaged or is not the right file. Download it again with `PYTHONPATH=src python scripts/download_data.py --force` (plain `make data` skips files that are already present). |
| `UserWarning: The data file's bytes differ from those the splits were frozen from ...` | `get_split` | The ids match, but some bytes of the CSV changed (for example a label). The frozen split is still returned. If the change is intended, regenerate and log it. Otherwise re-download the file. |
| `UserWarning: The splits were frozen with cleaning version ...` | `get_split` | `CLEANING_VERSION` in `src/listen/text.py` was changed after the freeze, so the near-duplicate groups may not match the cleaned texts any more. Regenerate with `--force` and log it. |
| `UserWarning: config['splits'] is ... but the splits were frozen with ...` | `get_split` | The ratios in the config were edited after the freeze. Undo the edit, or regenerate and log it. |
| `TypeError: clean_text expects a str, got ...` | `clean_text` | You passed something that is not a string (`None`, `bytes`, or the NaN pandas gives for an empty cell). Decode bytes, and drop or fill missing values, before calling. |
| `FileExistsError: Frozen split file(s) already exist ...` (printed by `make splits`) | `freeze_splits` | The files exist and `make splits` refuses to overwrite. Overwrite on purpose only with the direct command `PYTHONPATH=src python scripts/make_splits.py --force`, then log it. |
| `` make: unrecognized option `--force' `` | make | `make` takes `--force` as one of its own options. Run the script directly, as in the previous row. |
| `Frozen splits differ from a fresh computation:` followed by a list (exit status 1) | `PYTHONPATH=src python scripts/make_splits.py --check` | The committed files no longer match what the code computes now. Read the list. A library version change or a changed rule is the usual cause. Do not overwrite the files to make the check pass without an entry in `docs/decisions.md`. |
| `ImportError` mentioning `iterstrat` | `freeze_splits`, `--check`, `make_splits` | `iterative-stratification` is not installed. Run `pip install -r requirements.txt`. Only regenerating or checking splits needs it; `get_split` never imports it. |
| `Model and scoring code is not implemented yet.` (exit status 1 of `make results`) | `scripts/run_results.py` | Expected until the models exist. The lines printed before it show the part sizes of every seed. |

## FAQ

**Why can the validation part have no Disaster Relief (or Veterans & Military) stories?**
Because there are almost none in the data. The best-guess file has 3 Disaster Relief positives and 4 Veterans & Military positives among 3,459 stories. A part holds about 10% of the stories, so the chance that a part gets one of 3 stories is small, and nothing in a stratified split can create stories that do not exist. On the frozen splits (measured 2026-11-13), Disaster Relief has no positive in val and none in test for seeds 0 to 3, and Veterans & Military has none in val and none in test for seeds 0 and 4. The exact counts per seed are in the table below. Every category with 10 or more positives has at least 2 positives in every part for every seed. Native American Issues has no positive anywhere. What to do about it in the evaluation is in `docs/decisions.md` ("rare categories keep their column").

Labeled positive stories per part, written train/val/test (measured 2026-11-13 on the frozen files; the number in brackets is the total over all three parts):

| Category | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Disaster Relief (3) | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 | 1/1/1 |
| Veterans & Military (4) | 4/0/0 | 2/1/1 | 3/0/1 | 3/0/1 | 4/0/0 |
| Environmental Justice (8) | 6/1/1 | 7/1/0 | 6/1/1 | 6/1/1 | 6/1/1 |
| Immigration (26) | 20/3/3 | 21/2/3 | 21/3/2 | 21/3/2 | 22/2/2 |
| Native American Issues and Tribal Law (0) | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |

**Why do the part sizes differ by a few stories from 80/10/10 and from seed to seed?**
The split is done on near-duplicate groups, not on single stories. A group with 2 to 4 stories lands in one part as a block, so the story counts can be a few off the ideal. In the frozen files (measured 2026-11-13) the train, val and test sizes are 2,766, 347 and 346 for seed 0; 2,763, 348 and 348 for seed 1; 2,768, 344 and 347 for seed 2; 2,760, 346 and 353 for seed 3; and 2,767, 346 and 346 for seed 4. The tolerance used in the tests is plus or minus 15 stories around 2,767, 346 and 346.

**Why does `Y` have zeros where `mask` is 0?**
Numpy integer arrays cannot hold "missing". An omitted label is stored as 0 so the array has a fixed shape and type. The mask is the other half of the encoding: it says which cells are real. `Y` is meaningful only where `mask` is 1. (The loader keeps the unknown cells as NaN in `load_dataset`'s float array; `get_split` converts to int8 and zero-fills.)

**Can I change the seeds or the split ratios?**
Technically yes, but it changes the stories in every part, so every result made with the old splits becomes incomparable. It costs: editing `configs/default.yaml`, running `PYTHONPATH=src python scripts/make_splits.py --force`, running `PYTHONPATH=src python scripts/make_splits.py --check`, committing the new files in `results/splits/`, re-running every model for every seed, and adding a dated entry to `docs/decisions.md` (README, "Working practice"). The code accepts only ratios where val equals test and train is `1 - 2 / n_splits`, for example 0.8, 0.1, 0.1 (`docs/decisions.md`, grouped iterative stratification). Ask the team before doing it.

**Can I use the 95%-confidence file or the continuous files?**
`get_split` uses the file named by `data.default_set` in `configs/default.yaml` (`best_guess_binary`). The frozen files are made for that one file and carry its SHA-256. Switching `default_set` means freezing splits for the other file (`make splits` writes files named after the set), re-running everything and a decisions entry. Nobody should switch it casually.

**How do I regenerate or verify the frozen splits?**
Verify (writes nothing, takes a few seconds): `PYTHONPATH=src python scripts/make_splits.py --check`. It prints `frozen splits match` when the committed files equal a fresh computation. Regenerate: `make splits` creates missing files and refuses to overwrite; to replace them run `PYTHONPATH=src python scripts/make_splits.py --force` (`make splits --force` does not work because make reads `--force` as its own option). Then run `--check`, commit the files with the code change and a `docs/decisions.md` entry, and re-run results. The two files and their keys are described in `results/splits/README.md`.

**Will `get_split` give me a different split on another computer or after I upgrade a library?**
No. It reads the committed id lists and never recomputes anything or imports the stratification library. Only `--force` and `--check` recompute.

**Do the tests need the data?**
`make test` runs the whole suite. Tests that need `data/` or the frozen files are skipped, not failed, when they are absent. The examples marked `<!-- tested -->` on this page are run by `tests/test_docs.py`, so this page cannot drift away from the code unnoticed.
