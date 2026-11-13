"""Explain category scores by removing one word at a time."""

import re

from .models import MultiLabelModel


def word_contributions(model: MultiLabelModel, text: str, category: int):
    """Return each word's position and its effect on the category score."""
    return occlusion(model, text, category)


def occlusion(model: MultiLabelModel, text: str, category: int):
    """Compare the original score with the score after removing each word."""
    words = list(re.finditer(r"\b\w+(?:['’]\w+)*\b", text))
    if not words:
        return []

    # Score the original and every edited version in one call.
    stories = [text]
    for word in words:
        stories.append(text[:word.start()] + " " + text[word.end():])

    scores = model.predict_proba(stories)[:, category]
    original_score = scores[0]

    contributions = []
    for index, word in enumerate(words, start=1):
        contributions.append({
            "word": word.group(),
            "start": word.start(),
            "end": word.end(),
            "contribution": float(original_score - scores[index]),
        })

    return contributions