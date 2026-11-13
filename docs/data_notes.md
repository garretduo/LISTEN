# Data notes

Facts about the Learned Hands files, checked on 2026-11-13. Update this file if the source changes.

## Source and license

- Data page: https://learned-hands.github.io/project-hub/data.html
- Files live at `https://learned-hands.github.io/project-hub/texts/`. The page links them relatively.
- License: CC BY-NC-SA 4.0, from Suffolk's LIT Lab and Stanford's Legal Design Lab. Do not redistribute. `data/` is gitignored.
- `scripts/download_data.py` fetches the files; the list lives in `configs/default.yaml`.

## Files

| Key | File | Rows | Size |
|---|---|---|---|
| `best_guess_binary` | `2019-04-23_best-guess_binary.csv` | 3,459 | 4.6 MB |
| `best_guess_continuous` | `2019-04-23_best-guess_continuous.csv` | 3,459 | 4.9 MB |
| `confidence95_binary` | `2019-04-23_95p-confidence_binary.csv` | 1,506 | 1.9 MB |
| `confidence95_continuous` | `2019-04-23_95p-confidence_continuous.csv` | 1,506 | 2.0 MB |

Per the data page, best guess is the center of the Wilson score interval for all texts with at least one label. In the 95% confidence files a label is `1` if the interval's lower bound is above 50%, `0` if its upper bound is below 50%, and omitted if the interval straddles 50%.

## Schema (binary files)

22 columns: `_id`, `full_text`, then 20 category columns. Cells are `0.0`, `1.0` or empty.

The continuous files have the same header and the same empty cells as their binary twins (48.6% and 46.8%). Their labeled cells are Wilson-interval centers, all strictly between 0.125 and 0.867.

The 20 categories, in file order:

1. Accidents, Injuries, and Torts (Problems with Others)
2. Benefits
3. Civil and Human Rights
4. Court and Lawyers
5. Crime & Prisons
6. Disaster Relief
7. Education
8. Environmental Justice
9. Estates & Wills
10. Family
11. Government services
12. Health
13. Housing
14. Immigration
15. Money, Debt, and Consumer Issues
16. Native American Issues and Tribal Law
17. Small Business and IP
18. Traffic and Cars
19. Veterans & Military
20. Work and Employment

## Omitted cells

Both versions have many empty label cells, so the mask `m` is never all ones. An empty cell is unknown, not negative.

| File | Label cells | Empty | Labeled | Labeled = 1 |
|---|---|---|---|---|
| `best_guess_binary` | 69,180 | 33,630 (48.6%) | 35,550 | 3,479 (9.8% of labeled) |
| `confidence95_binary` | 30,120 | 14,106 (46.8%) | 16,014 | 1,362 (8.5% of labeled) |

The labeled totals match the counts on the data page. The page explains the empty cells in the 95% files (interval straddles 50%) but not in the best-guess files; the cause there is unconfirmed.

Posts average about 1,200 characters (`full_text`: 1,260 in best guess, 1,210 in 95% confidence).

## Per-category counts (`best_guess_binary`)

Measured 2026-11-13 on the 3,459 stories. "Labeled" is the number of stories whose cell for that category is not empty. "Positives" is how many of those have the value `1.0`. A positive is a story the crowd labeled as belonging to the category. The last column is positives divided by labeled, so it ignores the empty cells, which are unknown, not negative.

| Category | Labeled | Positives | % of labeled that are positive |
|---|---|---|---|
| Accidents, Injuries, and Torts (Problems with Others) | 1,695 | 482 | 28.4% |
| Benefits | 1,695 | 36 | 2.1% |
| Civil and Human Rights | 1,685 | 134 | 8.0% |
| Court and Lawyers | 1,685 | 399 | 23.7% |
| Crime & Prisons | 1,718 | 344 | 20.0% |
| Disaster Relief | 1,863 | 3 | 0.2% |
| Education | 2,006 | 40 | 2.0% |
| Environmental Justice | 1,685 | 8 | 0.5% |
| Estates & Wills | 1,685 | 71 | 4.2% |
| Family | 1,904 | 263 | 13.8% |
| Government services | 1,685 | 65 | 3.9% |
| Health | 1,976 | 151 | 7.6% |
| Housing | 1,840 | 367 | 19.9% |
| Immigration | 1,686 | 26 | 1.5% |
| Money, Debt, and Consumer Issues | 1,685 | 473 | 28.1% |
| Native American Issues and Tribal Law | 1,686 | 0 | 0.0% |
| Small Business and IP | 1,685 | 100 | 5.9% |
| Traffic and Cars | 1,938 | 200 | 10.3% |
| Veterans & Military | 1,687 | 4 | 0.2% |
| Work and Employment | 2,061 | 313 | 15.2% |
| **Total** | **35,550** | **3,479** | 9.8% |

The totals match the "Omitted cells" table above. Five categories are rare: Native American Issues and Tribal Law has no positive story at all, Disaster Relief has 3, Veterans & Military 4, Environmental Justice 8 and Immigration 26. A model cannot learn Native American Issues from this file, and its precision, recall and F1 are undefined for every part. With so few positives, some parts of a split have none for the others too (`docs/using_splits.md`, FAQ). The consequence for thresholds and macro-F1 is left to the evaluation sprint (`docs/decisions.md`, "rare categories").

## Label structure (`best_guess_binary`)

Measured 2026-11-13. The empty cells are not spread evenly. They come in whole rows: most stories have either all 20 cells filled or only one.

| Stories | Labeled cells per story |
|---|---|
| 1,685 | all 20 |
| 1,752 | exactly 1 |
| 20 | between 2 and 8 (3 stories with 2, 2 with 3, 4 with 4, 3 with 5, 2 with 6, 5 with 7, 1 with 8) |
| 2 | none |

On average a story has 10.3 labeled cells. The 1,752 stories with exactly one labeled cell are almost all negative on it: 1,604 of them have `0.0` in that cell and 148 have `1.0`. So a story with a single labeled cell tells you almost nothing about which categories it belongs to. It mostly tells you about one category it does not belong to, and the other 19 cells are unknown. Fitting the model for category `c` on rows with a mask of 1 uses these rows correctly, but it means each category's training set is the stories labeled for that category, about 1,700 to 2,060 of the 3,459, and not all of them.

Positive labels per story (counting only labeled cells): 1,633 stories have none, 723 have 1, 695 have 2, 292 have 3, 95 have 4, 17 have 5, 3 have 6 and 1 has 7. So 408 stories have three or more positives, and 1,826 have at least one.

## Near-duplicates (`best_guess_binary`)

Measured 2026-11-13 on the cleaned text. Some stories are reposts of each other with an edited opening. We call a set of such stories a near-duplicate group. Two stories are linked when at least 80% of their 5-word shingles match (a shingle is a run of five consecutive lowercase words, and the match is the Jaccard similarity of the two sets of shingles). A group is everything connected by links.

| Measure | Value |
|---|---|
| Groups with 2 or more stories | 52 |
| Stories in those groups | 107 (3.1% of 3,459) |
| Largest group | 4 stories |
| Groups whose members have conflicting labels (a category labeled 1 in one story and 0 in another) | 12 |
| Groups whose members differ only in cells that are empty in one story | 14 |
| Groups whose members have identical labels | 26 |
| Exact duplicate texts as stored | 4 groups (8 stories) |
| Exact duplicate texts after lowercasing and collapsing whitespace | 6 groups (12 stories) |

The grouping itself is in `src/listen/splits.py` (`near_duplicate_groups`, `GROUPING_VERSION = 1`). The splits keep each group inside one part (`docs/decisions.md`). Twelve groups carry conflicting labels, which is a rough measure of how noisy the crowd labels are: the same story told twice did not always get the same labels.

## Text cleaning (`best_guess_binary`)

Every model and the app read the text through `listen.text.clean_text` (`CLEANING_VERSION = 1`). The rules and the reasons are in `docs/decisions.md` ("text cleaning rules, version 1") and in the docstring of `src/listen/text.py`. This section records what the raw text looks like and what cleaning changes. Measured 2026-11-13 on all 3,459 stories, counting stories (not occurrences) unless stated. Patterns differ slightly between counts, so treat numbers as accurate to a few stories. The "After cleaning" column and the totals below were measured with `clean_text` as it stood on 2026-11-13 after rule 2b was added, and they may move slightly if the rules change again. Every number in this section was recomputed from the committed code and the data on 2026-11-13.

| Feature of the raw text | Stories | After cleaning |
|---|---|---|
| Line breaks | 2,464 (71.2%), every one written as CRLF (`\r\n`), none as a lone `\n` | `\n` only; no `\r` left |
| Three or more line breaks in a row | 200 | 0 (at most one blank line) |
| `http://`, `https://` or `www.` URLs | 181 (264 URLs: 261 start with `http://` or `https://`, 3 with a bare `www.`) | 0 |
| Markdown links `[text](url)` with a URL target | 86 (140 links) | link text kept, URL removed |
| `u/name` or `r/name` | 91 (counted with the pattern `clean_text` uses) | 0 |
| `@` characters | 6 | 4 (see below) |
| HTML entities such as `&gt;` | 165, escaped once | decoded once |
| Non-breaking spaces escaped twice (`&amp;nbsp;`) | 12 (75 occurrences) | each becomes a plain space (rule 2b); no literal `&nbsp;` is left |
| Non-ASCII characters | 220 (6.4%), of which 138 have curly quotes | 72 (accents, pound sign, emoji stay; curly quotes: 0) |
| No-break space or zero-width characters | 10 | 0 |
| A short bracketed tag of up to 30 characters, such as a jurisdiction tag `[CA]` | 1,077 (31.1%) | 1,039 (kept; the drop is link text such as `[text](url)` losing its brackets) |

Overall, 2,741 of the 3,459 stories (79.2%) change, mostly through the line-break rule, and the mean length goes from 1,259.6 to 1,244.3 characters. No story becomes empty. After cleaning, the shortest story has 11 characters, the median 920 and the longest 15,136.

What the cleaning leaves on purpose, so nobody is surprised:

- 4 stories still contain an `@`. An `@` is removed only when it starts a mention (`@name`). Of the 4, one is inside an e-mail address, which is kept on purpose, and the other 3 are an `@` that stands alone.
- 3 stories still contain `](`. These are Markdown-style links whose target the link rule (rule 3) does not recognise as a web address, for example a target that is not a URL.
- 1 story still shows a literal `&amp;`. The source escaped it twice (`&amp;amp;`), and decoding only once is the rule, because a general second decode would also change text that was really meant to contain `&amp;`. The one double-escaped entity that is common in this file, the non-breaking space (`&amp;nbsp;`), has its own rule (2b), so no literal `&nbsp;` remains. Before rule 2b was added, 12 stories still showed a literal `&nbsp;` after cleaning.
- A zero-width character hidden inside `http` stops the link rule from matching, because invisible characters are removed (rule 5) after links are (rule 3). No such case occurs in this file (checked 2026-11-13).

## Frozen splits

The 80/10/10 train/val/test parts for seeds 0 to 4 are computed once and stored as id lists in `results/splits/` (`best_guess_binary.csv` and `best_guess_binary.meta.json`). They hold ids and part names only, never story text. `get_split(seed)` reads them. How to use it: `docs/using_splits.md`. How they are made, checked and regenerated: `results/splits/README.md` and `docs/decisions.md`.
