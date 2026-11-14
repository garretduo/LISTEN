"""Causes the team can assign to a mistake. One main cause per mistake. Settle this list
before labeling starts, because the sheets and the agreement script both read it."""

CAUSES = {
    "label": "The label looks wrong, or the story supports more than one reading.",
    "lay_wording": "Everyday wording that the model can't connect to the legal category.",
    "neighbor": "A closely related category took over, such as housing vs. money and debt.",
    "vague": "The story is too short or too vague to tell.",
    "multi_issue": "Several issues compete in one story.",
    "other": "None of the above. Say why in notes.",
}
