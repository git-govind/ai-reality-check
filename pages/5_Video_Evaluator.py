"""
pages/5_Video_Evaluator.py — Video Evaluator (coming soon)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Video Evaluator — VeritasIQ",
    page_icon="🎥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .viq-brand  { display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.15rem; }
    .viq-name   { font-size:0.95rem; font-weight:700; color:#94a3b8; letter-spacing:-0.01em; }
    .viq-tag    { font-size:0.7rem; color:#4a5568; font-style:italic; }
    [data-testid="stAppDeployButton"] { display:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="display:none" class="viq-brand">'
    '<span class="viq-name">VeritasIQ</span>'
    '<span class="viq-tag">Where Intelligence Meets Integrity</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center; padding: 5rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🎥</div>
        <h1 style="font-size:2rem; font-weight:800; margin-bottom:0.5rem;">Video Evaluator</h1>
        <p style="color:#6b7280; font-size:1.1rem; font-style:italic;">Coming soon…</p>
        <p style="color:#4b5563; font-size:0.9rem; margin-top:1rem; max-width:480px; margin-left:auto; margin-right:auto;">
            Detect deepfakes, AI-generated video, and frame-level manipulation
            using temporal forensics, facial consistency analysis, and
            per-frame authenticity scoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
