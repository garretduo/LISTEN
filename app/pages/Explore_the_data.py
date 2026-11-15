"""LISTEN data explorer: browse the cleaned stories and labels, see the splits, compare the models.

Input:  artifacts/story_view/stories.csv.gz and predictions.csv.gz, written by
        scripts/export_story_view.py. If they are missing the page shows a notice and nothing else.
Output: an interactive page. Pick a split, click a category bar to filter, step through the
        stories, and see what each of the five models said about a held-out story.
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

st.set_page_config(page_title="LISTEN: explore the data", layout="wide")

# Chart blue, validated for color-blind readers on light and dark surfaces.
BLUE = "#3987e5" if getattr(st.context.theme, "type", "light") == "dark" else "#2a78d6"


@st.cache_data
def load():
    """Return the story table, the model predictions and the category names."""
    stories = pd.read_csv(VIEW_DIR / "stories.csv.gz", dtype={"id": str})
    predictions = pd.read_csv(VIEW_DIR / "predictions.csv.gz", dtype={"id": str}, keep_default_na=False)
    categories = [c for c in stories.columns if c not in ("id", "text") and not c.startswith("seed_")]
    yes = stories[categories].fillna(0).astype(int)
    stories["labels"] = [" | ".join(c for c in categories if row[c]) or "(none)" for _, row in yes.iterrows()]
    return stories, predictions, categories


def bars(values, x_title, select=None):
    """Horizontal bars, one per category, with full category names on the axis."""
    chart = alt.Chart(values.rename_axis("category").reset_index(name="value")).mark_bar(color=BLUE).encode(
        y=alt.Y("category:N", title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("value:Q", title=x_title),
        tooltip=["category", "value"],
    )
    if select is not None:  # clicking a bar selects its category and dims the others
        chart = chart.add_params(select).encode(opacity=alt.condition(select, alt.value(1), alt.value(0.35)))
    return chart


try:
    stories, predictions, categories = load()
except FileNotFoundError:
    st.info("The dataset is not included in this deployment.")
    st.stop()

st.title("Explore the dataset")
st.markdown(
    f"**{len(stories):,} stories** from r/legaladvice, each marked yes, no or unknown for "
    f"**{len(categories)} legal categories**. Pick a split, click a bar to filter, then step through the stories."
)
st.caption(
    "From the [Learned Hands](https://learnedhands.law.stanford.edu/) dataset by Suffolk University's LIT Lab "
    "and Stanford's Legal Design Lab, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
    "We removed links and usernames and normalised whitespace, so the text differs from the original. "
    "Non-commercial use only; share alike."
)

seeds = sorted(int(c.split("_")[1]) for c in stories.columns if c.startswith("seed_"))
seed = st.sidebar.selectbox(
    "Random split (seed)", seeds, help="The stories are split 80/10/10 five different ways. Each seed is one way."
)
part_col = f"seed_{seed}"
total_yes = stories[categories].sum()

explore, coverage = st.tabs(["Explore stories", "Label coverage"])

with explore:
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
            bars(counts, "stories", pick).properties(height=480), on_select="rerun", key="category_chart", width="stretch"
        )
        chosen = [d["category"] for d in event.selection.get("pick", [])]
        rare = [c for c in categories if total_yes[c] < MIN_POSITIVES]
        st.caption(
            "Click a bar to show only those stories (shift-click to combine, click empty space to clear). "
            f"{', '.join(rare)} have fewer than {MIN_POSITIVES} yes stories, so the headline results leave them out."
        )
        query = st.text_input("Search the stories", placeholder="for example: landlord")

    view = in_split
    if chosen:
        view = view[(view[chosen] == 1).all(axis=1)]
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
                said = predictions[(predictions["seed"] == seed) & (predictions["id"] == story_id)].set_index("model")["predicted"]
                for key, label in MODELS.items():
                    guess = set(said[key].split(" | ")) if said[key] else set()
                    badges = [f":green-badge[✓ {c}]" if c in yes else f":red-badge[✗ {c}]" for c in categories if c in guess]
                    badges += [f":gray-badge[missed {c}]" for c in yes if c not in guess]
                    st.markdown(f"{label}: " + (" ".join(badges) or "said none"))

    with st.expander(f"All {len(ids):,} matching stories as a table"):
        st.dataframe(view[["labels", "text"]], hide_index=True, width="stretch")

with coverage:
    st.markdown("About half of the label cells are empty. An empty cell is unknown, not a no.")
    known = (100 * stories[categories].notna().mean()).round(1)
    st.altair_chart(bars(known, "% of stories with a known label").properties(height=480), width="stretch")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame({"stories marked yes": total_yes, "% with a known label": known}), width="stretch")
