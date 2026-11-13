"""Word-level explanations for the app's highlighting.

Not implemented yet. For a story and a category, return how much each word pushed
that category's probability up or down. Everything runs locally and nothing is logged.
"""

from .models import MultiLabelModel


def word_contributions(model: MultiLabelModel, text: str, category: int):
    """Per-word contribution to the probability of category, from the model's own weights."""
    raise NotImplementedError


def occlusion(model: MultiLabelModel, text: str, category: int):
    """Per-word contribution measured by removing each word and re-scoring the story."""
    raise NotImplementedError
