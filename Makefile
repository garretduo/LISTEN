# LISTEN entry points. Run the targets in this order: data, results, app, test.
# `splits` only regenerates the frozen train/val/test split files, which are
# already committed; run it only when the split rules change (docs/decisions.md).
# Written to work with GNU Make 3.81 (the version macOS ships).

PYTHON ?= python3

# The package lives in src/, so make it importable without `pip install -e .`.
export PYTHONPATH := src

.PHONY: data splits results app test

# Download the Learned Hands CSVs into data/ (gitignored).
data:
	$(PYTHON) scripts/download_data.py

# Write the frozen train/val/test id lists into results/splits/. Refuses to overwrite;
# to replace them, run `PYTHONPATH=src python scripts/make_splits.py --force` (make cannot pass --force).
splits:
	$(PYTHON) scripts/make_splits.py

# Fit every model and print the results table.
results:
	$(PYTHON) scripts/run_results.py

# Launch the Streamlit app. Usage stats are off because the app must log nothing.
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --browser.gatherUsageStats false

# Run the test suite (cleaning, splits, data loading, documentation examples).
test:
	$(PYTHON) -m pytest
