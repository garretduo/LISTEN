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
