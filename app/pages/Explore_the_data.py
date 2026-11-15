"""LISTEN data explorer: browse the cleaned stories and labels, see which categories go together,
and watch how a model's scores turn into yes/no calls.

Input:  artifacts/story_view/stories.csv.gz, scores.csv.gz and cutoffs.csv.gz, written by
        scripts/export_story_view.py. If they are missing the page shows a notice and nothing else.
Output: an interactive page. Pick a split, click category bars to filter, and a second chart shows
        which other categories go with the selection. Step through the stories, and for a held-out
        story move the cutoff to see how each of the five models decides.
The page logs and stores nothing, and makes no network calls.
"""

import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

VIEW_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "story_view"
PARTS = ("train", "val", "test")
MODELS = {
    "class_prior": "Class prior",
    "rules": "Keyword rules",
    "tfidf": "TF-IDF + logistic regression",
    "emb_lr": "Averaged embeddings + logistic regression",
    "emb_mlp": "Averaged embeddings + neural network",
}
MIN_POSITIVES = 10  # categories with fewer yes stories are left out of the headline results
TOGETHER_MIN = 10  # a category needs this many labeled stories in the selection to be compared
TOGETHER_SHOWN = 6
TUNED, ONE = "Tuned per category", "Same for all"

st.set_page_config(page_title="LISTEN: explore the data", layout="wide")

# Chart blue, validated for color-blind readers on light and dark surfaces.
DARK = getattr(st.context.theme, "type", "light") == "dark"
BLUE = "#3987e5" if DARK else "#2a78d6"
INK = "#e6e6e6" if DARK else "#333333"


@st.cache_data
def load():
    """Return the story table, each model's scores, the tuned cutoffs and the category names."""
    stories = pd.read_csv(VIEW_DIR / "stories.csv.gz", dtype={"id": str})
    scores = pd.read_csv(VIEW_DIR / "scores.csv.gz", dtype={"id": str}).set_index(["seed", "id", "model"]).sort_index()
    cutoffs = pd.read_csv(VIEW_DIR / "cutoffs.csv.gz").set_index(["seed", "model"]).sort_index()
    categories = [c for c in stories.columns if c not in ("id", "text") and not c.startswith("seed_")]
    yes = stories[categories].fillna(0).astype(int)
    stories["labels"] = [" | ".join(c for c in categories if row[c]) or "(none)" for _, row in yes.iterrows()]
    return stories, scores, cutoffs, categories


def bars(values, x_title, select=None):
    """Horizontal bars, one per category, with full category names on the axis."""
    chart = alt.Chart(values.rename_axis("category").reset_index(name="value")).mark_bar(color=BLUE).encode(
        y=alt.Y("category:N", title=None, axis=alt.Axis(labelLimit=400, labelOverlap=False)),  # keep every label
        x=alt.X("value:Q", title=x_title),
        tooltip=["category", "value"],
    )
    if select is not None:  # clicking a bar selects its category and dims the others
        chart = chart.add_params(select).encode(opacity=alt.condition(select, alt.value(1), alt.value(0.35)))
    return chart


def together(base, everyone, chosen):
    """For each other category, the share of the selected stories (`base`) also marked yes, next to the
    share on all stories. Only stories with a label for that category are counted."""
    rows = []
    for c in categories:
        known = base[c].dropna()
        if c not in chosen and len(known) >= TOGETHER_MIN and known.sum() > 0:
            rows.append((c, 100 * known.mean(), int(known.sum()), len(known), 100 * everyone[c].dropna().mean()))
    t = pd.DataFrame(rows, columns=["category", "pct", "also", "labeled", "overall"])
    return t.sort_values("pct", ascending=False, kind="stable").head(TOGETHER_SHOWN)  # nlargest fails on an empty table


def together_chart(t, select):
    """Bars for the share also marked yes, with a tick at the share on all stories."""
    y = alt.Y("category:N", sort=list(t["category"]), title=None, axis=alt.Axis(labelLimit=400, labelOverlap=False))
    tip = [
        alt.Tooltip("category:N", title="Category"),
        alt.Tooltip("pct:Q", title="% also yes", format=".0f"),
        alt.Tooltip("also:Q", title="Stories also yes"),
        alt.Tooltip("labeled:Q", title="Stories labeled"),
        alt.Tooltip("overall:Q", title="% yes on all stories", format=".0f"),
    ]
    bar = alt.Chart(t).mark_bar(color=BLUE).encode(
        y=y, x=alt.X("pct:Q", title="% also marked yes"), tooltip=tip
    ).add_params(select).encode(opacity=alt.condition(select, alt.value(1), alt.value(0.35)))
    tick = alt.Chart(t).mark_tick(color=INK, thickness=2, size=22).encode(
        y=y, x=alt.X("overall:Q", title="% also marked yes"), tooltip=tip
    )
    return alt.layer(bar, tick).properties(height=28 * len(t) + 70)  # the height includes the axis


try:
    stories, scores, cutoffs, categories = load()
except FileNotFoundError:
    st.info("The dataset is not included in this deployment.")
    st.stop()

st.title("Explore the dataset")
st.markdown(
    f"**{len(stories):,} stories** from r/legaladvice, each marked yes, no or unknown for "
    f"**{len(categories)} legal categories**. Click a bar to filter, step through the stories, "
    "and move the cutoff to see how the models decide."
)
st.caption(
    "From the [Learned Hands](https://learnedhands.law.stanford.edu/) dataset by Suffolk University's LIT Lab "
    "and Stanford's Legal Design Lab, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
    "We removed links and usernames and normalised whitespace, so the text differs from the original. "
    "Non-commercial use only; share alike."
)

# The cutoff controls are hidden on train stories; keep their settings while they are.
st.session_state.setdefault("cutoff_mode", TUNED)
st.session_state.setdefault("cutoff", 50)
st.session_state["cutoff_mode"], st.session_state["cutoff"] = st.session_state["cutoff_mode"], st.session_state["cutoff"]

seeds = sorted(int(c.split("_")[1]) for c in stories.columns if c.startswith("seed_"))
seed = st.sidebar.selectbox(
    "Random split (seed)", seeds, help="The stories are split 80/10/10 five different ways. Each seed is one way."
)
part_col = f"seed_{seed}"
total_yes = stories[categories].sum()

sizes = stories[part_col].value_counts()
names = {"all": f"All · {len(stories):,}", **{p: f"{p.capitalize()} · {sizes[p]:,}" for p in PARTS}}
split = st.segmented_control("Split", list(names), default="test", format_func=names.get, label_visibility="collapsed") or "all"
in_split = stories if split == "all" else stories[stories[part_col] == split]

left, right = st.columns([6, 5], gap="large")
with left:
    st.markdown("##### Stories marked yes, per category")
    pick = alt.selection_point(name="pick", fields=["category"])
    counts = in_split[categories].sum().astype(int)
    event = st.altair_chart(
        bars(counts, "stories", pick).properties(height=420), on_select="rerun", key="category_chart", width="stretch"
    )
    chosen = [d["category"] for d in event.selection.get("pick", [])]
    rare = [c for c in categories if total_yes[c] < MIN_POSITIVES]
    st.caption(
        "Click a bar to show only those stories (shift-click to combine, click empty space to clear). "
        f"{', '.join(rare)} have fewer than {MIN_POSITIVES} yes stories, so the headline results leave them out."
    )

    also = []
    if not chosen:
        st.caption("Click a category to see which other categories it is usually marked yes with.")
    else:
        st.markdown("##### Also marked yes with the selection")
        # All stories, not just this split: a split has too few stories for steady percentages.
        t = together(stories[(stories[chosen] == 1).all(axis=1)], stories, chosen)
        if t.empty:
            st.caption("Too few stories with these labels to compare.")
        else:
            # The key changes with the selection, so an old click never filters unseen.
            together_event = st.altair_chart(
                together_chart(t, alt.selection_point(name="also", fields=["category"])),
                on_select="rerun", key=f"also_{'|'.join(chosen)}", width="stretch",
            )
            also = [d["category"] for d in together_event.selection.get("also", []) if d["category"] in set(t["category"])]
            st.caption(
                "Across all stories. Bars: the share of the selected stories also marked yes, among those with a label "
                "for that category. Tick: the share on all stories. A bar well past its tick means the two go together "
                "more than chance. Click a bar to add it to the filter."
            )
    query = st.text_input("Search the stories", placeholder="for example: landlord")

view = in_split
if chosen + also:
    view = view[(view[chosen + also] == 1).all(axis=1)]
if query:
    view = view[view["text"].str.contains(query, case=False, regex=False)]
ids = view["id"].tolist()

with right:
    if not ids:
        st.info("No stories match. Clear the category or search to see more.")
    else:
        story_id = st.session_state.get("story_id")
        if story_id not in ids:
            story_id = random.choice(ids)
        i = ids.index(story_id)
        back, shuffle, forward = st.columns(3)
        if back.button("◀ Previous", disabled=i == 0, width="stretch"):
            story_id = ids[i - 1]
        if shuffle.button("Random story", type="primary", width="stretch"):
            story_id = random.choice(ids)
        if forward.button("Next ▶", disabled=i == len(ids) - 1, width="stretch"):
            story_id = ids[i + 1]
        st.session_state["story_id"] = story_id
        i = ids.index(story_id)

        row = view.iloc[i]
        labels = row[categories].fillna(-1).astype(int)
        yes = [c for c in categories if labels[c] == 1]
        with st.container(border=True):
            st.caption(f"Story {i + 1:,} of {len(ids):,} · {row[part_col]} split")
            st.markdown(" ".join(f":blue-badge[{c}]" for c in yes) or ":gray-badge[no category marked yes]")
            with st.container(height=240 if len(row["text"]) > 600 else "content"):  # long stories scroll
                st.write(row["text"])
            st.caption(f"{int((labels == -1).sum())} of {len(categories)} categories have no label for this story (unknown, not no).")

        if row[part_col] == "train":
            st.info("The models trained on this story, so their guesses would not be a fair test. Pick a test story to compare them.")
        else:
            note = " (val stories tuned the cutoffs, so test is the fair test)" if row[part_col] == "val" else ""
            st.markdown(f"**What each model said**{note}")
            st.caption("Each model gives every category a score. It says yes when the score reaches the cutoff.")
            tuned = (st.segmented_control("Cutoff", [TUNED, ONE], key="cutoff_mode") or TUNED) == TUNED
            cut = None if tuned else st.slider("Say yes at or above", 10, 90, step=5, format="%d%%", key="cutoff") / 100
            for key, label in MODELS.items():
                score = scores.loc[(seed, story_id, key)]
                says = score >= (cutoffs.loc[(seed, key)] if tuned else cut)
                badges = []
                for c in categories:
                    pct = f"{score[c]:.0%}"
                    if says[c] and labels[c] == 1:
                        badges.append(f":green-badge[✓ {c} {pct}]")
                    elif says[c] and labels[c] == 0:
                        badges.append(f":red-badge[✗ {c} {pct}]")
                    elif says[c]:
                        badges.append(f":orange-badge[? {c} {pct}]")
                    elif labels[c] == 1:
                        badges.append(f":gray-badge[missed {c} {pct}]")
                st.markdown(f"{label}: " + (" ".join(badges) or "said none"))
            st.caption("✓ the crowd agreed · ✗ the crowd said no · ? the crowd left this blank · missed: the crowd said yes")

with st.expander(f"All {len(ids):,} matching stories as a table"):
    st.dataframe(view[["labels", "text"]], hide_index=True, width="stretch")
