# LISTEN

**LISTEN** (Legal Issue Spotting in Textual Everyday Narratives) is a local Streamlit app that reads a person's own story of a legal problem and suggests which of 20 LIST categories apply, with word-level explanations. The app is used as a small study of where TF-IDF and embedding-based classifiers fail on lay legal narratives, built on the [Learned Hands](https://learnedhands.law.stanford.edu/) legal dataset.

LISTEN was built by **Garret Duo, Lyndon Huang, and Sohum Sarwate** as a final project for the COSC 72 course (Accelerated Computational Linguistics) at Dartmouth College for the Fall 2026 quarter.

## What the app does

1. Paste a story, for example "my landlord changed the locks while I was at work", or pick a built-in example.
2. Choose a model (keyword rules, TF-IDF with logistic regression, or averaged embeddings with a feedforward network).
3. Click "Analyze".
 
The result is a ranked list of up to five categories, each with a probability bar. Categories above their tuned threshold are labeled likely, and the rest are greyed out. Below the list, the story appears again with the words that pushed each category up shaded, and clicking a category switches the highlighting to it. If no category clears its threshold, the app says the story is too short or too general and shows two or three follow-up prompts, such as "Who is the other party?"
 
The app logs and stores nothing. Models and data are strictly stored locally.

## Research question

This app was motivated by our central research question:
> "Where does a TF-IDF linear classifier fail on lay legal narratives, and does a dense-embedding model fix those failures or only move them?"

We had three hypotheses up front:

1. Averaged embeddings beat TF-IDF on low-support categories, because they let the model use words it saw only a few times in training.
2. Averaging blurs categories that share vocabulary (consumer, housing, torts), so macro-F1 may rise while errors between neighboring categories increase.
3. With a few thousand stories, a feedforward network adds little over logistic regression, because averaging discards word order and the extra capacity overfits.

## Task and method

The task is, at its core, multi-label classification. A story can belong to several categories at once, so the model makes 20 yes/no judgments per story instead of picking one winner. Each model outputs a probability per category, and a per-category threshold turns those probabilities into a final classification.

We compare five models, ordered below from simplest to most complex. Models 1 and 2 are reference baselines. Models 3 through 5 are controlled comparisons, differing from the previous model in a single component (either the feature representation or the classifier).

1. **Class-prior baseline**: predict each category at its training frequency, ignoring the text.
2. **Rule-based baseline**: hand-written regular expressions built from the LIST category definitions.
3. **TF-IDF**: using unigram and bigram features with one-vs-rest logistic regression; the main baseline.
4. **Averaged pretrained word embeddings**: using mean pooling with logistic regression; the same classifier as Model 3 with a different feature representation.
5. **Averaged pretrained word embeddings with an MLP**: using a feedforward neural network; same features as Model 4 with a different classifier.

Our proposed model is Model 5.

### Splits and metrics

We split 80/10/10 at the story level with iterative stratification, so rare categories appear in every split, and repeat with five seeds. The test set is touched once, after all model choices are frozen. We tune one threshold per category on the validation split to maximize F1. The headline metric is macro-F1. We also report micro-F1, per-category precision, recall and F1, recall at 3, and 95% bootstrap confidence intervals over stories. We do not use per-label accuracy, because most categories are absent from most stories and predicting nothing scores high.

### Partially labeled stories
  
In the 95%-confidence files, a label is omitted when the crowd's agreement interval straddles 50%. We treat an omitted cell as unknown, not negative. Let $m_{ic} = 1$ if story $i$ has a label for category $c$ and $0$ if the cell is omitted. The network's loss counts only labeled cells:

$$
\mathcal{L}_i = -\sum_{c=1}^{20} m_{ic}\left[\,y_{ic}\log p_{ic} + (1-y_{ic})\log(1-p_{ic})\,\right]
$$

The logistic regression for category $c$ is fit only on rows where $m_{ic} = 1$, and thresholds and per-category metrics use the same rows. Iterative stratification needs a complete label matrix, so for splitting only we set omitted cells to 0. Both versions of the dataset have omitted cells, so $m$ is not all ones: 48.6% of the label cells in the best-guess binary file and 46.8% in the 95%-confidence binary file are empty (checked 2026-11-13). The data page explains the omissions in the 95%-confidence files but not in the best-guess files, and we treat both the same way.

## Data

We use [Learned Hands](https://learnedhands.law.stanford.edu/), a corpus of anonymized posts from the r/legaladvice subreddit. Volunteer lawyers and law students, and later the public, labeled the posts in an online game using the Legal Issues Taxonomy (LIST). It was built by Suffolk University's Legal Innovation and Technology (LIT) Lab and Stanford's Legal Design Lab.

There are two versions of the dataset. The best-guess version has 3,459 texts and 20 categories, and the 95%-confidence version is a smaller, higher-agreement subset of 1,506 texts. These counts come from the project's data page.

The data is not in this repository. `scripts/download_data.py` fetches it into `data/`, which is gitignored.

## Quick start

To run a fresh clone end-to-end, run the following commands in the exact order specified:

1. `pip install -r requirements.txt` (installs required libraries)
2. `make data` (downloads the Learned Hands dataset)
3. `make results` (fits every model and prints the results table)
4. `make app` (launches the Streamlit app)

## Repository layout

```
.
├── README.md
├── LICENSE
├── Makefile
├── requirements.txt
├── configs/default.yaml        seeds, split ratios, paths
├── scripts/download_data.py
├── scripts/run_results.py      driver for `make results`
├── src/listen/
│   ├── data.py                 loading, label masks, splits
│   ├── evaluate.py             metrics and bootstrap intervals
│   ├── thresholds.py           per-category threshold tuning
│   ├── baselines.py            class-prior and rule-based baselines
│   ├── models/                 TF-IDF, embedding + LR, embedding + network
│   └── explain.py              word contributions and occlusion
├── app/streamlit_app.py
├── analysis/                   error-labeling sheet, minimal pairs
├── docs/                       decisions.md, data_notes.md
├── results/                    small tables only
└── data/, artifacts/           gitignored
```

## Results

TBD

## Limits and ethics

- LISTEN names categories only. It never states outcomes, rights, deadlines or recommended actions, and it creates no attorney-client relationship.
- A missed category can send someone to the wrong resource. The app shows probabilities and evidence, and says "need more detail" when scores are low instead of guessing.
- The labels are crowd-labeled, and the dataset authors say they have not yet tested them against expert labels.
- The posts are English, U.S.-centered and from Reddit, written in 2019 or earlier. US law varies by state, which the labels do not capture. We do not claim the tool generalizes beyond this.
- The people who wrote the posts did not expect this use. We use only the released, anonymized files, remove usernames and links, make no attempt to re-identify anyone, and paraphrase instead of quoting posts where we can.

## Working practice

- Any change to splits or metrics goes in `docs/decisions.md`, so results stay comparable.
- Seeds live in `configs/default.yaml`.

## License and attribution

The code is MIT licensed (see `LICENSE`).

The Learned Hands data is licensed CC BY-NC-SA 4.0 by the Suffolk LIT Lab and the Stanford Legal Design Lab. We do not redistribute it. Anything we publish that is derived from it, such as trained weights or error-analysis sheets that quote or paraphrase posts, carries the same license. Dependencies and pretrained vectors keep their own licenses.

## References
 
- Learned Hands data page: https://learned-hands.github.io/project-hub/data.html
- LIST, the Legal Issues Taxonomy: https://www.taxonomy.legal/
- Spot, Suffolk's deployed issue spotter built on this data: https://spot.suffolklitlab.org/
- Sechidis, Tsoumakas and Vlahavas (2011). On the Stratification of Multi-Label Data. ECML PKDD.
- Lipton, Elkan and Narayanaswamy (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML PKDD.
- Jurafsky and Martin. Speech and Language Processing (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
