"""
app.py — Navigation entry-point for AI Reality Check.

Defines sidebar page order and labels using st.navigation / st.Page.
Run with:  streamlit run app.py
"""
import streamlit as st

pg = st.navigation([
    st.Page("pages/1_Dashboard.py",       title="Dashboard",        icon="📊"),
    st.Page("pages/2_Text_Evaluator.py",  title="AI Reality Check", icon="🔍"),
    st.Page("pages/3_Image_Evaluator.py", title="Image Evaluator",  icon="🖼️"),
])
pg.run()
