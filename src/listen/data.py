"""Loading the Learned Hands CSVs, the label mask, and the splits.

Not implemented yet. Rules the real code must keep (see README, "Partially labeled
stories"):

- An empty cell is unknown, not negative. Read it as missing, never as 0.
- The mask is 1 where a label exists and 0 where the cell is omitted. Both dataset
  versions have omitted cells (docs/data_notes.md), so the mask is never all ones.
- Iterative stratification needs a complete label matrix, so omitted cells may be
  filled with 0 for splitting only. Nowhere else.
"""

import numpy as np


def load_dataset(config: dict):
    """Load the CSV chosen by config["data"]["default_set"].

    Should return the story texts, the (n_stories, 20) label matrix with missing
    cells kept as missing, the mask, and the 20 category names from the CSV header.
    """
    raise NotImplementedError


def label_mask(Y: np.ndarray) -> np.ndarray:
    """Return the (n_stories, 20) 0/1 mask: 1 where Y has a label, 0 where it is missing."""
    raise NotImplementedError


def make_splits(Y: np.ndarray, config: dict, seed: int):
    """Story-level 80/10/10 iterative stratification (ratios from config["splits"]).

    Y here may have omitted cells filled with 0, for splitting only.
    """
    raise NotImplementedError
