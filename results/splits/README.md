# Frozen splits

This folder holds the train/validation/test division of the Learned Hands best-guess stories, computed once and saved so that every teammate and every machine uses exactly the same parts. `listen.data.get_split(seed)` reads these files. To use them, read `docs/using_splits.md`; you do not need to open the files yourself.

Facts below were checked on 2026-11-13.

## Files

| File | What it is |
|---|---|
| `best_guess_binary.csv` | One row per story with the part (`train`, `val` or `test`) it belongs to for each seed. |
| `best_guess_binary.meta.json` | How, when and from which data file the CSV was made, so a mismatch can be detected. |
| `README.md` | This page. |

The file names come from `data.default_set` in `configs/default.yaml`. If that setting ever changes, the files are named after the new set.

### `best_guess_binary.csv`

- Header: `_id,seed_0,seed_1,seed_2,seed_3,seed_4`. There is one `seed_<n>` column for every seed in `configs/default.yaml`.
- One row per story, 3,459 rows, sorted by `_id` in plain string order. The `_id` is the story id from the data file (24 lowercase hex characters).
- Every seed cell is exactly `train`, `val` or `test`. Near-duplicate stories (reposts of each other) always have the same value in a column.
- Line endings are `\n`. The file is written the same way on every operating system, so git diffs stay clean.

A row looks like this, with a made-up id:

```
<24-character hex id>,train,train,val,train,test
```

### `best_guess_binary.meta.json`

Plain JSON with sorted keys. Every key is required: a file that lacks one is rejected with a message that says so.

| Key | Type | Meaning |
|---|---|---|
| `data_set` | string | The key of the data set in `configs/default.yaml` (`best_guess_binary`). |
| `file` | string | Name of the CSV in `data/` the splits were made from (`2019-04-23_best-guess_binary.csv`). |
| `sha256` | string | SHA-256 of that CSV's bytes. If the CSV changes later, `get_split` warns. |
| `n_stories` | integer | Number of stories (3,459). |
| `categories` | list of 20 strings | Category names in the CSV's column order. `get_split` refuses to run if they differ from the CSV header. |
| `seeds` | list of integers | The seeds frozen here (`[0, 1, 2, 3, 4]`). |
| `ratios` | object | The split ratios used: `train`, `val`, `test` (0.8, 0.1, 0.1). |
| `method` | string | Plain-English description of the algorithm. For people only; it is not compared. |
| `cleaning_version` | integer | `CLEANING_VERSION` of `listen.text` when the files were made (1). The near-duplicate groups were found on the cleaned text. |
| `grouping` | object | The near-duplicate settings and result, with the four keys below. |
| `grouping.version` | integer | `GROUPING_VERSION` of `listen.splits` (1). |
| `grouping.k` | integer | Shingle length in words (5). |
| `grouping.threshold` | number | Smallest Jaccard similarity that links two stories (0.8). |
| `grouping.groups` | list of lists of ids | Every near-duplicate group with more than one story, each sorted by id (52 groups, 107 stories). Single stories are not listed. |
| `versions` | object | Versions of `python`, `numpy`, `scikit-learn` and `iterative-stratification` that were installed when the files were made. A difference from the installed versions only produces a warning. |
| `created` | string | Date of creation, ISO format (`YYYY-MM-DD`). |

## What is in the files and what is not

The files hold story ids and part names only, with no story text and no labels. They still derive from the Learned Hands data, which is licensed CC BY-NC-SA 4.0 by the Suffolk LIT Lab and the Stanford Legal Design Lab, so these files carry the same license. The data CSVs themselves are not in git (`data/` is gitignored).

## How they are made

`PYTHONPATH=src python scripts/make_splits.py` (also `make splits`) runs `listen.data.freeze_splits`:

1. Read the data CSV and sort the rows by `_id`.
2. Clean every text once with `listen.text.clean_text` (`CLEANING_VERSION` 1).
3. Group near-duplicate stories: two stories are linked when at least 80% of their 5-word shingles match, and a group is everything connected by links (`listen.splits.near_duplicate_groups`, `GROUPING_VERSION` 1).
4. For each seed, split the groups with multilabel iterative stratification (`MultilabelStratifiedKFold` from `iterative-stratification==0.1.9`, 10 folds, `shuffle=True`, `random_state=seed`). Fold 0 is the test part, fold 1 is the val part and folds 2 to 9 are the train part. Omitted label cells count as 0 for this step only (`listen.splits.assign_parts`).
5. Write the two files.

The writing is deterministic. Running it again on the same data, code and library versions gives identical files, except that `created` carries the new date.

## Rules for this folder

- **Do not edit the files by hand.** A hand edit makes the parts differ from what the code would compute, and `--check` will say so.
- **Verify** that the files still match the code: `PYTHONPATH=src python scripts/make_splits.py --check`. It recomputes everything, writes nothing, prints `frozen splits match` and exits with status 0 when they agree. Otherwise it lists the differences and exits with status 1. It compares the parts of every seed, the near-duplicate groups, the story ids, the categories, the grouping settings, the cleaning version, the ratios, the seeds and the SHA-256. It does not compare `created` or `method`, and it only warns about library `versions`.
- **Regenerate** only on purpose: `PYTHONPATH=src python scripts/make_splits.py --force`. Plain `make splits` writes the files when they are missing and refuses to overwrite them. `make splits --force` does not work, because make treats `--force` as its own option. Regenerating the folder does not delete this README.
- **Log every change** in `docs/decisions.md` in the same commit as the new files, and re-run every result that used the old splits. A change to the data file, the cleaning rules, the grouping, the ratios, the seeds or the `iterative-stratification` version can change which stories are in each part.
- **Commit the files.** They are small (a few hundred kilobytes) and must stay in git so that a fresh clone has them.
