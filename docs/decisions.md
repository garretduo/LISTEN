# Decisions

Any change to splits or metrics goes here, so results stay comparable. Newest entries last.

## 2026-11-13: initial configuration

Values in `configs/default.yaml` when the repo was scaffolded:

- Seeds `[0, 1, 2, 3, 4]`. Placeholders chosen for the scaffold, not tuned. Changing them after results exist means re-running everything.
- Split ratios 80/10/10 at the story level with iterative stratification, as in the README.
- Default data set: `best_guess_binary` (3,459 stories), chosen by Garret over `confidence95_binary` (1,506 stories). The other three files are downloaded too and can be selected through `data.default_set`.

## 2026-11-13: omitted label cells exist in both versions

Both binary files have many empty label cells (48.6% in best guess, 46.8% in 95% confidence; details in `docs/data_notes.md`). The README had left open whether the best-guess files were fully labeled. They are not, so the mask `m` is never all ones, and omitted cells are treated as unknown in every model, threshold and metric.
