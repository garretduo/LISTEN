"""Models 1 and 2: the reference baselines.

Neither is part of the controlled comparison in Models 3 to 5. Both follow the
MultiLabelModel interface so they run through the same evaluation.
"""

from .models import MultiLabelModel


class ClassPriorBaseline(MultiLabelModel):
    """Model 1: predict each category at its training frequency, ignoring the text.

    Not implemented yet. The frequency for a category must come from the rows where
    that category's label is known (mask == 1).
    """


class RuleBasedBaseline(MultiLabelModel):
    """Model 2: hand-written regular expressions built from the LIST category definitions.

    Not implemented yet.
    """
