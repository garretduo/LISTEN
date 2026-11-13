# Decisions

Any change to splits or metrics goes here, so results stay comparable. Newest entries last.

## 2026-11-13: initial configuration

Values in `configs/default.yaml` when the repo was scaffolded:

- Seeds `[0, 1, 2, 3, 4]`. Placeholders chosen for the scaffold, not tuned. Changing them after results exist means re-running everything.
- Split ratios 80/10/10 at the story level with iterative stratification, as in the README.
- Default data set: `best_guess_binary` (3,459 stories), chosen by Garret over `confidence95_binary` (1,506 stories). The other three files are downloaded too and can be selected through `data.default_set`.

## 2026-11-13: omitted label cells exist in both versions

Both binary files have many empty label cells (48.6% in best guess, 46.8% in 95% confidence; details in `docs/data_notes.md`). The README had left open whether the best-guess files were fully labeled. They are not, so the mask `m` is never all ones, and omitted cells are treated as unknown in every model, threshold and metric.

## 2026-11-13: text cleaning rules, version 1

**Decision.** Every model and the Streamlit app read text that has been through one function, `listen.text.clean_text` (`CLEANING_VERSION = 1`). It takes RAW story text and is called exactly once per story. The rules run in this order:

1. `\r\n`, a lone `\r`, U+2028 and U+2029 become `\n`.
2. `html.unescape` runs exactly once (the files store `&`, `<` and `>` as entities, once). Rule 2b follows it: any literal `&nbsp;` still left becomes a single space. The corpus double-escapes non-breaking spaces (`&amp;nbsp;`, 75 occurrences in 12 stories, measured 2026-11-13), so one unescape leaves a literal `&nbsp;` behind. There is deliberately no general second unescape.
3. Markdown links `[text](url)` become `text`, then bare `http://`, `https://` and `www.` URLs are removed. The scheme part is case-insensitive, so `HTTP://`, `HTTPS://` and `Www.` are handled too.
4. Reddit names (`u/name`, `r/name`, with or without a leading slash) and `@mentions` are removed. The patterns are anchored so `he/she`, `and/or` and e-mail addresses survive. The `u/` and `r/` pattern is lowercase only, following the Reddit convention.
5. Zero-width characters and the soft hyphen are removed, no-break and other Unicode spaces become a plain space, curly quotes, dashes and the ellipsis become ASCII. Accents, the pound sign, emoji and jurisdiction tags such as `[CA]` are kept.
6. Runs of spaces and tabs become one space, spaces around newlines are trimmed, three or more newlines become two, and the ends are stripped.

Left out on purpose: lowercasing, stopword removal and stemming (the same text feeds TF-IDF, which can lowercase by itself, embedding models that may want the original casing, and the app), and any change to e-mail addresses, phone numbers or Markdown emphasis. Known limitations, accepted: one story keeps a literal `&amp;` (it was escaped as `&amp;amp;`, and a general second unescape would also damage text that really contains `&amp;`), and a zero-width character hidden inside `http` survives link removal, because rule 5 deletes invisible characters after rule 3 has run. Measured effect on the best-guess file is in `docs/data_notes.md`.

**Why.** The README promises that we "remove usernames and links". The five-model comparison is only controlled if the models differ in features and classifier, never in preprocessing, so the cleaning lives in one function. The near-duplicate grouping below also reads the cleaned text.

**Changing it safely.** Cleaning is not guaranteed to give the same result when applied twice (for example `&amp;gt;` unescapes once to `&gt;`, and a second call would turn that into `>`), so never clean already-cleaned text. The unescape step is exactly one `html.unescape` plus the targeted `&nbsp;` rule; do not add a second general unescape to fix a rare leftover entity. Any edit that can change the output of `clean_text` needs all of these in the same commit: add 1 to `CLEANING_VERSION`; regenerate the frozen splits (the groups depend on the cleaned text); re-run every result that used the old splits; add an entry here. Edits that leave the output unchanged (comments, docstrings, variable names) need none of that.

## 2026-11-13: near-duplicate stories are kept in the same part

**Decision.** A near-duplicate group is a set of stories that are reposts of each other. Two stories are linked when at least 80% of their 5-word shingles match (the Jaccard similarity of the two sets of five consecutive lowercase words, taken from the cleaned text), and a group is everything connected by links. Each whole group goes to one part (train, val or test). The grouping parameters are 5 words and 0.8, and `GROUPING_VERSION = 1` in `src/listen/splits.py`.

Measured on the best-guess binary file (2026-11-13): 52 groups holding 107 stories (3.1% of 3,459), the largest group has 4 stories. Of the 52 groups, 12 contain conflicting labels, 14 differ only in cells that are missing in one copy, and 26 have identical labels. Exact duplicate texts: 4 groups (8 stories) as stored, 6 groups (12 stories) after lowercasing and collapsing whitespace.

**Why.** If a story is in train and its repost is in test, a model can score well by remembering the twin, and the reported macro-F1 is too optimistic. Keeping groups together removes that leak. The cost is that part sizes at the story level are off from exactly 80/10/10 by a few stories (see the part sizes in the frozen splits entry below).

**Changing it safely.** Any change to the shingle length, the threshold, the tokenisation or the group-id rule changes who is grouped, so bump `GROUPING_VERSION`, regenerate the frozen splits, re-run results and add an entry here. Changing `CLEANING_VERSION` also counts, because the grouping reads cleaned text.

## 2026-11-13: grouped iterative stratification, and the fold mapping

**Decision.** The split is made over near-duplicate groups, not single stories, with multilabel iterative stratification (`MultilabelStratifiedKFold` from the pinned `iterative-stratification==0.1.9`). Iterative stratification tries to give every part the same share of each category's positive stories, rarest categories first. Details that decide the exact result:

- Rows are put in `_id` order, and the distinct group ids (each group is named by its smallest story `_id`) are sorted as plain strings before stratifying. This makes a story's part depend on the data and the seed, not on the order of rows in the CSV.
- A group counts as positive for a category when any of its stories is positive.
- Omitted label cells are filled with 0 for the stratification step ONLY. That is the one place in the project where an omitted cell is read as 0. Everything after the split uses the original labels and the mask (see the "Y encoding" entry below).
- `n_splits = round(1 / ratios["test"])`, which is 10 for 80/10/10. The stratifier runs with `shuffle=True` and `random_state=seed`. Fold 0 is the test part, fold 1 is the val part, folds 2 to 9 are the train part.
- The config must have `val == test == 1 / n_splits` and `train == 1 - 2 / n_splits`; the code raises a `ValueError` otherwise.

**Why.** The README already chose iterative stratification at the story level (Sechidis et al., 2011). Grouping is added on top because of the near-duplicate decision above. Sorting by id removes a hidden dependence on file order that the stratifier would otherwise have.

**Changing it safely.** The ratios, the fold mapping, the sort rule and the `iterative-stratification` version all change the exact stories in each part. Change them only through `configs/default.yaml` or `src/listen/splits.py`, bump `GROUPING_VERSION`, regenerate and re-check the frozen splits, re-run results and add an entry here. Do not upgrade `iterative-stratification` or scikit-learn without running `PYTHONPATH=src python scripts/make_splits.py --check` (it imports private scikit-learn code and can change behaviour or break).

## 2026-11-13: seeds 0 to 4 are frozen

**Decision.** The seeds `[0, 1, 2, 3, 4]` in `configs/default.yaml` are no longer placeholders. A seed picks one of five different random 80/10/10 splits of the same stories, and results are reported as the average over the five. The splits for these seeds are frozen (next entry). This replaces the "placeholders" wording in the first entry of this file and in the comment in `configs/default.yaml`.

**Why.** All five models must be compared on identical splits, and the final numbers need a fixed set of repeats to be reproducible.

**Changing it safely.** `get_split(seed)` accepts only the seeds listed in `configs/default.yaml` and raises a `ValueError` for any other value, so it cannot be used to try out a new seed. Adding or removing a seed means editing the config, running `PYTHONPATH=src python scripts/make_splits.py --force` (the frozen files hold exactly the old seeds, and `make splits` refuses to overwrite them), re-running every result and adding an entry here.

## 2026-11-13: frozen split files

**Decision.** The splits are computed once and committed as id lists in `results/splits/`:

- `best_guess_binary.csv`: one row per story, columns `_id,seed_0,seed_1,seed_2,seed_3,seed_4`, each seed cell is `train`, `val` or `test`.
- `best_guess_binary.meta.json`: the settings and versions the files were made with (full list of keys in `results/splits/README.md`).

`get_split(seed)` reads these files and the data CSV. It never recomputes a split, so a library upgrade cannot quietly change anyone's train or test set. The files hold ids only, no post text, but they derive from CC BY-NC-SA data and carry that license. Part sizes (train, val, test stories) in the frozen files, measured 2026-11-13: seed 0 has 2,766, 347 and 346; seed 1 has 2,763, 348 and 348; seed 2 has 2,768, 344 and 347; seed 3 has 2,760, 346 and 353; seed 4 has 2,767, 346 and 346.

**Why.** The test set must be identical for every model and every teammate. A committed id list is easy to review and diff, and it works on any machine.

**Commands.**

- `make splits` writes the files if they are missing and refuses to overwrite existing ones.
- `PYTHONPATH=src python scripts/make_splits.py --force` replaces them. (`make splits --force` does not work: make treats `--force` as its own option and stops.)
- `PYTHONPATH=src python scripts/make_splits.py --check` recomputes the splits and compares them to the committed files, without writing anything.

**Changing it safely.** Never edit the files by hand. If a rule changes (cleaning, grouping, stratification, ratios, seeds, dataset), regenerate with `--force`, run `--check`, commit the files together with the code and this entry, and re-run all results. Old results and new results must not be compared.

## 2026-11-13: rare categories keep their column

**Decision.** All 20 category columns are kept, including those with almost no positive stories. In the best-guess binary file (measured 2026-11-13) Native American Issues and Tribal Law has 0 positives, Disaster Relief 3, Veterans & Military 4, Environmental Justice 8 and Immigration 26 (counts of stories labeled 1 for that category, out of about 1,700 to 2,000 stories labeled for it; full table in `docs/data_notes.md`).

**Consequence.** Native American Issues cannot be learned or evaluated from this file. Disaster Relief has no positive stories in the val part and none in the test part for seeds 0 to 3, and Veterans & Military has none in val or test for seeds 0 and 4 (measured on the frozen splits, 2026-11-13). Environmental Justice (8 positives) can also end up with 0 or 1 positive in a part. Exact per-seed counts are in the table below. In a part with no positives, F1, precision and recall for that category are undefined, and a threshold cannot be tuned on val. Every category with 10 or more positives has at least 2 in every part for every seed.

Labeled positive stories per part, written train/val/test (measured 2026-11-13 on the frozen files; the number in brackets is the total over all three parts):

| Category | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Disaster Relief (3) | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 | 1/1/1 |
| Veterans & Military (4) | 4/0/0 | 2/1/1 | 3/0/1 | 3/0/1 | 4/0/0 |
| Environmental Justice (8) | 6/1/1 | 7/1/0 | 6/1/1 | 6/1/1 | 6/1/1 |
| Immigration (26) | 20/3/3 | 21/2/3 | 21/3/2 | 21/3/2 | 22/2/2 |
| Native American Issues and Tribal Law (0) | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |

**Left to the evaluation sprint.** This sprint only provides the splits. How to set thresholds for such categories, and whether macro-F1 skips them, counts them as 0 or uses another rule, is not decided here. Decide it once, write it here, and apply it to all five models. Until then do not report a macro-F1 that silently averages over categories whose F1 is undefined. Decided in "headline metrics use 16 categories", below.

**Changing it safely.** Dropping a column changes the shape of every label array, so it needs the same steps as a split change. Merging or removing categories would also change the research question; ask the team first.

## 2026-11-13: Y encoding, zero-filled labels plus a mandatory mask

**Decision.** Every part returned by `get_split` carries two int8 arrays of shape (n_stories, 20): `Y`, which holds 1 for a positive label and 0 for everything else, and `mask`, which holds 1 where the label is known and 0 where the cell was omitted. Where `mask` is 0, `Y` is 0 and that 0 means UNKNOWN, not "no". `Y` alone cannot tell a known "no" from an unknown. So code that trains or scores must use the mask: logistic regression for category `c` fits only on rows with `mask[:, c] == 1`, the network loss sums only over cells with `mask == 1`, and thresholds and per-category metrics use the same rows.

**Why.** A fixed-shape numeric array is what scikit-learn and PyTorch-style code expect, and NaNs would silently poison them. Pairing it with a mask keeps the README's rule that omitted means unknown. About 48.6% of the label cells are omitted, so ignoring the mask would treat roughly half the cells as negatives.

**Changing it safely.** The array dtype, shape, and the Y/mask convention are used by every model and by the tests in `tests/`. If the convention changes, update `docs/using_splits.md`, the tests and all models in the same change, and add an entry here.

## 2026-11-13: headline metrics use 16 categories

**Decision.** Macro-F1 and micro-F1 are computed over the 16 categories that have 10 or more positive stories in the best-guess file. The other four (Disaster Relief, Environmental Justice, Native American Issues and Tribal Law, Veterans & Military) are listed under `evaluation.excluded_categories` in `configs/default.yaml`. All five models are scored on the same 16 for every seed, so numbers compare across models and seeds. Recall@3 still ranks and counts all 20 categories, because it is defined for any story with a known positive.

**Why.** The rule above guarantees at least 2 positives per scored category in every val and test part (table in "rare categories keep their column"; also checked on the frozen splits when this was written), so F1 and the tuned threshold are always defined. Averaging "whatever is defined" would change the set of categories from seed to seed, and counting undefined F1 as 0 would penalize every model for a category nobody can be scored on.

**What stays the same.** The four columns are still in `Y` and `mask`, still trained, and still listed per category in `per_category.csv`, which `make results` writes to `results/`, where precision, recall or F1 is blank when undefined.

**Other rules.**
- F1 and recall for a category are undefined (n/a) when the part has no known positive story for it; precision is undefined when nothing was predicted.
- Threshold for a category with no known positive in val: 0.5. This only affects the four excluded categories.
- Bootstrap: a resample of stories can lose every positive of a scored category (a category with 2 or 3 test positives loses them all in roughly 1 resample in 7). That category is skipped in that resample's macro-F1, as in the rule above, instead of counting as 0. This slightly widens the low end of the interval.
- A model with constant scores (the class prior) gets a tuned threshold that predicts every category for every story, because that maximizes F1 for it. Its F1 is a reference floor, not a bug.

**Changing it safely.** A different excluded list or minimum count changes every headline number, so log it here and re-run all five models.
