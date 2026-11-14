"""LISTEN Streamlit app.

Input:  a pasted story (or a built-in example) and a model choice.
Output: up to five ranked categories with probability bars; categories above their
        threshold are marked likely. If none clears its threshold, the app asks for
        more detail instead of guessing.

Launch with `make app`, or on Windows:
    python -m streamlit run app/streamlit_app.py --browser.gatherUsageStats false
The app must log and store nothing, so add no logging or network calls here.
"""

import sys
from pathlib import Path
import html
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so `import listen` works without make

from listen.explain import occlusion, word_contributions
from listen.models.stand_in import StandInModel
from listen.models.tfidf import TfidfLogReg

# The 20 categories in file order (docs/data_notes.md). Must match the models' columns.
CATEGORIES = [
    "Accidents, Injuries, and Torts (Problems with Others)",
    "Benefits",
    "Civil and Human Rights",
    "Court and Lawyers",
    "Crime & Prisons",
    "Disaster Relief",
    "Education",
    "Environmental Justice",
    "Estates & Wills",
    "Family",
    "Government services",
    "Health",
    "Housing",
    "Immigration",
    "Money, Debt, and Consumer Issues",
    "Native American Issues and Tribal Law",
    "Small Business and IP",
    "Traffic and Cars",
    "Veterans & Military",
    "Work and Employment",
]

# Label shown in the app -> file-name key for the saved model in artifacts/.
MODELS = {
    "Keyword rules": "rules",
    "TF-IDF + logistic regression": "tfidf",
    "Averaged embeddings + neural network": "emb_mlp",
}

# Written by the team, not taken from Reddit posts.
EXAMPLES = {
    "Locked out by landlord": (
        "My landlord changed the locks on my apartment while I was at work. "
        "All my things are still inside and he says I can't come back until I pay "
        "the rent I owe from last month."
    ),
    "Fired and not paid": (
        "I was fired from my warehouse job last week with no warning. They still "
        "haven't paid me for my last two weeks of work, and my manager won't answer "
        "my calls."
    ),
    "Car accident and insurance": (
        "Another driver ran a red light and hit my car. I hurt my neck and had to go "
        "to the emergency room, and now their insurance company says they won't pay "
        "for my medical bills."
    ),
    "Too vague": "I have a problem with a company.",
}

FOLLOW_UPS = [
    "Who is the other party? For example, a landlord, an employer, a family member, a company, or a government office.",
    "What happened, and when?",
    "What would you like to happen next?",
]

TOP_K = 5
STAND_IN_THRESHOLD = 0.5


@st.cache_resource
def load_model(key):
    """Return (model, thresholds, is_stand_in) for a model key.

    Trained models are saved by scripts/train_app_models.py as
    artifacts/app/<key>.joblib. Until that file exists, the app uses the stand-in.
    """
    path = ROOT / "artifacts" / "app" / f"{key}.joblib"
    if path.exists():
        import joblib

        bundle = joblib.load(path)
        if bundle["categories"] != CATEGORIES:
            raise ValueError(f"{path.name} has different categories from the app's CATEGORIES list.")
        return bundle["model"], bundle["thresholds"], False
    return StandInModel(len(CATEGORIES)), np.full(len(CATEGORIES), STAND_IN_THRESHOLD), True

def explain(model, text, category):
    """Per word contributions to one category's score.

    The two logistic regression models explain from their own weights. Every other model 
    (neural network, keyword rules and stand-in) uses occulsion to remove each word and see 
    how much to score drops. 
    """
    if isinstance(model, TfidfLogReg):
        return word_contributions(model, text, category)
    return occlusion(model, text, category)

def highlight_html(text, spans):
    """Return the story as HTML with words shaded by how much they affected the score"""
    strongest = max((s["contribution"] for s in spans), default = 0.0)
    pieces = []
    position = 0
    for span in spans:
        pieces.append(html.escape(text[position:span["start"]]))
        word = html.escape(text[span["start"]:span["end"]])
        if strongest > 0 and span["contribution"] > 0:
            #Darker shading for more impactful words
            alpha =  0.15 + 0.6 * span["contribution"] / strongest
            pieces.append(
                f'<span style="background-color: rgba(255, 196, 0, {alpha:.2f}); '
                f'border-radius: 3px; padding: 0 2px;">{word}</span>'
            )
        else:
            pieces.append(word)
        position = span["end"]
    pieces.append(html.escape(text[position:]))
    return '<div style="line-height: 1.9;">' + "".join(pieces) + "</div>"

st.set_page_config(page_title="LISTEN")
st.title("LISTEN")
st.caption("Legal Issue Spotting in Textual Everyday Narratives")
st.warning(
    "LISTEN only suggests which legal categories a story may involve. It is not legal "
    "advice, does not predict outcomes or deadlines, and creates no attorney-client relationship."
)

example = st.selectbox("Start from an example (optional)", ["(none)"] + list(EXAMPLES))
story = st.text_area(
    "Describe your situation in your own words",
    value=EXAMPLES.get(example, ""),
    height=180,
    key=f"story_{example}",
)

model_label = st.radio("Model", list(MODELS))

if st.button("Analyze", type="primary"):
    if not story.strip():
        st.session_state.pop("result", None)
        st.error("Type or paste a story first.")
    else:
        model, thresholds, is_stand_in = load_model(MODELS[model_label])
        # Saved so the results stay on screen when the page reruns
        # (for example, when a different category is picked for highlighting).
        st.session_state["result"] = {
            "story": story,
            "model_label": model_label,
            "probs": model.predict_proba([story])[0],
        }

result = st.session_state.get("result")
if result:
    model, thresholds, is_stand_in = load_model(MODELS[result["model_label"]])
    probs = result["probs"]
    top = [int(i) for i in np.argsort(probs)[::-1][:TOP_K]]
    likely = [i for i in top if probs[i] >= thresholds[i]]

    if result["story"] != story or result["model_label"] != model_label:
        st.caption("The story or model changed. Press Analyze to update these results.")
    if is_stand_in:
        st.info("Using a stand-in model with random scores. These results mean nothing yet.")

    if not likely:
        st.subheader("Need more detail")
        st.write(
            "Your story may be too short or too general for LISTEN to suggest a "
            "category. It may help to add:"
        )
        for question in FOLLOW_UPS:
            st.markdown(f"- {question}")

    st.subheader("Possible categories")
    st.caption("Each category has its own cutoff, so a lower bar can still be marked likely.")
    for i in top:
        if probs[i] >= thresholds[i]:
            st.markdown(f"**{CATEGORIES[i]}** · likely")
        else:
            st.markdown(f":gray[{CATEGORIES[i]}]")
        st.progress(float(probs[i]), text=f"{probs[i]:.0%}")

    st.subheader("Words behind the guess")
    chosen = st.radio(
        "Show the words that pushed this category up",
        top,
        format_func=lambda i: CATEGORIES[i],
    )
    spans = explain(model, result["story"], chosen)
    st.markdown(highlight_html(result["story"], spans), unsafe_allow_html=True)
    st.caption("Darker shading means the word raised this category's score more.")