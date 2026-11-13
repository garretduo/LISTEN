# LISTEN entry points. Run the targets in this order: data, results, app.
# Written to work with GNU Make 3.81 (the version macOS ships).

PYTHON ?= python3

# The package lives in src/, so make it importable without `pip install -e .`.
export PYTHONPATH := src

.PHONY: data results app

# Download the Learned Hands CSVs into data/ (gitignored).
data:
	$(PYTHON) scripts/download_data.py

# Fit every model and print the results table.
results:
	$(PYTHON) scripts/run_results.py

# Launch the Streamlit app. Usage stats are off because the app must log nothing.
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --browser.gatherUsageStats false
