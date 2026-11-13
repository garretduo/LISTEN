"""LISTEN Streamlit app (stand-in).

Input:  none yet. The finished app takes a pasted story and a model choice.
Output: a notice that the app is not built yet.

Launch with `make app`, which turns Streamlit's usage statistics off. The app must
log and store nothing, so add no logging or network calls here.
"""

import streamlit as st

st.set_page_config(page_title="LISTEN")
st.title("LISTEN")
st.caption("Legal Issue Spotting in Textual Everyday Narratives")
st.info("The app is not implemented yet.")
