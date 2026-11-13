"""Word-level explanations for the app's highlighting."""

import re

from .models import MultiLabelModel
from .models.embedding_lr import EmbeddingLogReg
from .models.tfidf import TfidfLogReg


def word_contributions(model: MultiLabelModel, text: str, category: int):
    """Return each word's contribution to the selected category's logit."""
    words = list(re.finditer(r"\b\w+\b", text))
    contributions = [0.0] * len(words)
    classifier = model.classifiers[category]

    if not isinstance(classifier, float):
        weights = classifier.coef_[0]

        if isinstance(model, EmbeddingLogReg):
            vectors = model.vectorizer.word_vectors
            known_words = [
                (index, vectors[word.group().lower()])
                for index, word in enumerate(words)
                if word.group().lower() in vectors
            ]

            # Each known word contributes its share of the averaged vector.
            for index, vector in known_words:
                contributions[index] = float(weights @ vector) / len(known_words)

        elif isinstance(model, TfidfLogReg):
            vectorizer = model.vectorizer
            features = vectorizer.transform([text])[0]
            token_indexes = [
                index for index, word in enumerate(words) if len(word.group()) >= 2
            ]

            # Track every occurrence of each unigram and bigram in the story.
            occurrences = {}
            for position, index in enumerate(token_indexes):
                token = words[index].group().lower()
                occurrences.setdefault(token, []).append((index,))

                if position + 1 < len(token_indexes):
                    next_index = token_indexes[position + 1]
                    next_token = words[next_index].group().lower()
                    bigram = f"{token} {next_token}"
                    occurrences.setdefault(bigram, []).append((index, next_index))

            for term, groups in occurrences.items():
                column = vectorizer.vocabulary_.get(term)
                if column is None:
                    continue

                # Split repeated features across occurrences and bigrams across both words.
                share = float(features[0, column] * weights[column])
                share /= len(groups) * len(groups[0])
                for group in groups:
                    for index in group:
                        contributions[index] += share

    return [
        {
            "word": word.group(),
            "start": word.start(),
            "end": word.end(),
            "contribution": score,
        }
        for word, score in zip(words, contributions)
    ]


def occlusion(model: MultiLabelModel, text: str, category: int):
    """Compare the original probability with the probability after removing each word."""
    words = list(re.finditer(r"\b\w+(?:['’]\w+)*\b", text))
    if not words:
        return []

    stories = [text]
    for word in words:
        stories.append(text[:word.start()] + " " + text[word.end():])

    scores = model.predict_proba(stories)[:, category]
    original_score = scores[0]

    return [
        {
            "word": word.group(),
            "start": word.start(),
            "end": word.end(),
            "contribution": float(original_score - scores[index]),
        }
        for index, word in enumerate(words, start=1)
    ]