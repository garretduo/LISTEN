## Garret Duo

Tool for all rows: Claude Sonnet 5.5 (Claude Code)

| **Task** | **How checked** | **Where it appears** |
| --- | --- | --- |
| Consulted AI on cleaning the raw stories, handling inconsistencies, and preparing the dataset for training and evaluation. | The repository includes tests for text cleaning, data loading, and the saved train, validation, and test splits. | `src/listen/data.py`, `src/listen/text.py`, `tests/` |
| Used AI suggestions to organize the project directories and separate the scripts for data preparation, model training, and evaluation. | The final evaluation ran on the saved splits and produced both the overall and per-category results. | `Makefile`, `scripts/`, `results/results_table.txt`, `results/per_category.csv` |
| Used AI to draft docstrings and improve comments explaining what functions do, what they take in, and what they return. | The repository includes tests for documentation examples. | Docstrings and comments in the data-processing and evaluation code |

## Lyndon Huang

Tool for all rows: Claude (Anthropic), Opus 5.5, in the Claude app.

| Task | How checked | Where it appears |
|---|---|---|
| Drafted a team gameplan from the README and rubric | Team adjusted roles and dates to our own split | Planning only, not in the repo |
| Drafted the 50 lay-vs-legal minimal pairs | I read every pair, checked the everyday version has no legal terms and both versions tell the same facts, and edited where needed | `analysis/minimal_pairs.csv` |
| Helped with checking code for the app and for the model training scripts, clarified when I ran into issues. Also helped with the website design.
| Checked the minimal-pair results | Claude noticed the 30-step MLP gave the same top guess for every story; Garret fixed the MLP and I re-ran everything | `results/minimal_pairs_*.csv` |
| Helped with explanations for labeling causes for the error-labeling sheet. I read each story and if I couldn't decide on what the cause was I consulted claude to help me. 
| Help with git, Windows setup and debugging | Commands checked by running them | Not in the repo |

**Effect on results:** my shared labels (items 3–30) were AI-assisted, so the agreement
score partly reflects agreement with Claude's suggestions. The paper should say so next
to the kappa.

**Data handling:** my label sheet, which contains story text from the dataset, was
uploaded to Claude to get these suggestions. The team agreed AI help was allowed for
labeling.

## Sohum Sarwate

Tool for all rows: GPT-6.1 (OpenAI), used in ChatGPT.

| **Task** | **How checked** | **Where it appears** |
| --- | --- | --- |
| Helped draft and troubleshoot parts of the model code, including handling missing labels and sharing averaged GloVe vectors across both embedding models. | I ran small tests with known outcomes, then ran all five models on the first official validation split to verify that each returned scores for all 20 categories. | `src/listen/baselines.py`, `src/listen/models/` |
| Helped implement word-level explanations for the logistic regression models. | I checked that repeated words were highlighted at the correct positions and that the word contributions added up to the model's logit after subtracting the intercept. | `src/listen/explain.py` |
| Helped turn a one-off validation run into a reusable script that exports prediction scores. | I ran the export for all five splits and checked that the saved story IDs and category order matched the official validation data. | `scripts/export_validation_scores.py` |
