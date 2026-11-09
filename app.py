"""Entry point: one site, two pages, nav bar across the top.

    streamlit run app.py

Each page is its own script and only reads files in artifacts/:
  epm_app.py        -- NBA EPM projections (predictions.parquet etc., built by build_artifacts.py)
  draft_rankings.py -- draft model board (draft_rankings.csv, written by Darko_Draft.ipynb)
"""
import streamlit as st

st.set_page_config(page_title="Danny Harlan Basketball Models", layout="wide")

pg = st.navigation(
    [
        st.Page("epm_app.py", title="NBA EPM Projections", default=True),
        st.Page("draft_rankings.py", title="Draft Rankings"),
    ],
    position="top",
)
pg.run()
