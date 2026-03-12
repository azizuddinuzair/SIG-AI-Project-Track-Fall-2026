"""Starter Streamlit app for GA/GMM project workflows.

Run:
    py -m streamlit run app/streamlit_app.py
"""

import streamlit as st


st.set_page_config(page_title="Pokemon Team Optimizer", layout="wide")

st.title("Pokemon Team Optimizer")
st.caption("Streamlit-first interface for GA/GMM experimentation")

st.markdown("""
### Status
- Legacy CLI/scripts have been moved to `Proj1/legacy/scripts/`
- GA core remains active in `Proj1/src/ga/`

### Next Steps
- Add controls for GA config and seed
- Add run button + result table
- Add clustering diagnostics tab
""")
