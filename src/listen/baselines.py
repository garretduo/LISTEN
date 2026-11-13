"""Models 1 and 2: the reference baselines."""

import re

import numpy as np

from .models import MultiLabelModel


class ClassPriorBaseline(MultiLabelModel):
    """Score each category by how often it appears in the training set."""

    def fit(self, texts, Y, mask):
        self.category_scores = np.empty(Y.shape[1])

        for category in range(Y.shape[1]):
            # Skip stories whose label for this category is unknown.
            known = mask[:, category] == 1
            labels = Y[known, category]

            if labels.size == 0:
                raise ValueError(f"Category column {category} has no known training labels.")

            # The average of the 0s and 1s is the fraction marked yes.
            self.category_scores[category] = labels.mean()

        return self

    def predict_proba(self, texts):
        # Give every new story the same category scores.
        return np.tile(self.category_scores, (len(texts), 1))


class RuleBasedBaseline(MultiLabelModel):
    """Score each category using hand-written keyword rules."""

    keywords = {
        "Accidents, Injuries, and Torts (Problems with Others)": (
            r"personal injury|slip and fall|dog bite|medical malpractice|defamation"
        ),
        "Benefits": r"food stamps|unemployment benefits|social security|disability benefits",
        "Civil and Human Rights": r"civil rights|racial discrimination|hate crime",
        "Court and Lawyers": r"court date|summons|subpoena|attorney fees|small claims",
        "Crime & Prisons": r"arrested|criminal charges|probation|jail|prison",
        "Disaster Relief": r"fema|disaster assistance|disaster relief",
        "Education": r"school district|expelled|special education|iep",
        "Environmental Justice": r"toxic waste|water contamination|air pollution",
        "Estates & Wills": r"probate|inheritance|last will|executor",
        "Family": r"divorce|child custody|child support|adoption|restraining order",
        "Government services": r"government agency|public records|foia|building permit",
        "Health": r"health insurance|medical bills?|patient records|hospital",
        "Housing": r"landlord|eviction|security deposit|rent increase|lease agreement",
        "Immigration": r"green card|deportation|asylum|work visa|immigration",
        "Money, Debt, and Consumer Issues": (
            r"debt collection|credit report|credit card fraud|refund|bankruptcy"
        ),
        "Native American Issues and Tribal Law": r"tribal court|tribal law|tribal land",
        "Small Business and IP": r"copyright|trademark|patent|business license",
        "Traffic and Cars": r"traffic ticket|speeding ticket|car accident|dui|vehicle registration",
        "Veterans & Military": r"veterans benefits|va disability|military discharge",
        "Work and Employment": r"overtime|paychecks?|wrongful termination|workplace harassment",
    }

    def __init__(self, category_names):
        self.patterns = []

        # Keep the score columns in the same order as the category names.
        for category in category_names:
            pattern = re.compile(rf"\b(?:{self.keywords[category]})\b", re.IGNORECASE)
            self.patterns.append(pattern)

    def fit(self, texts, Y, mask):
        # The written rules are fixed, so there is nothing to learn from training.
        return self

    def predict_proba(self, texts):
        scores = np.zeros((len(texts), len(self.patterns)))

        for row, story in enumerate(texts):
            for column, pattern in enumerate(self.patterns):
                if pattern.search(story):
                    scores[row, column] = 1.0

        return scores