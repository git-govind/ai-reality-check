"""
pages/1_Dashboard.py — Reliability & Trend Dashboard
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard — AI Reality Check", page_icon="📊", layout="wide")
st.title("📊 Reliability Dashboard")
st.caption("Aggregated statistics across all evaluations this session.")

history = st.session_state.get("history", [])

if not history:
    st.info("No evaluations yet. Go to the main page and run some prompts first.")
    st.stop()

# ── Build dataframe ──
rows = []
for entry in history:
    r = entry["report"]
    rows.append({
        "timestamp": entry["ts"][:19].replace("T", " "),
        "model": entry["model"],
        "prompt": entry["prompt"][:50],
        "confidence": r.confidence_score,
        "accuracy": r.accuracy_score,
        "consistency": r.consistency_score,
        "safety": r.safety_score,
        "bias": r.bias_score,
        "clarity": r.clarity_score,
        "completeness": r.completeness_score,
        "grade": r.grade,
        "issues": len(r.all_issues),
    })

df = pd.DataFrame(rows)

# ── Summary metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Evaluations", len(df))
col2.metric("Avg Confidence", f"{df['confidence'].mean():.1f}/100")
col3.metric("Avg Accuracy", f"{df['accuracy'].mean():.1f}/100")
col4.metric("Avg Safety", f"{df['safety'].mean():.1f}/100")

st.divider()

# ── Confidence trend ──
st.markdown("### Confidence Score Over Time")
st.line_chart(df.set_index("timestamp")[["confidence"]])

st.divider()

# ── Score breakdown ──
st.markdown("### Score Breakdown Per Evaluation")
score_cols = ["accuracy", "consistency", "safety", "bias", "clarity", "completeness"]
st.bar_chart(df[score_cols])

st.divider()

# ── Model comparison ──
if df["model"].nunique() > 1:
    st.markdown("### Model Comparison (Average Scores)")
    model_avg = df.groupby("model")[score_cols + ["confidence"]].mean().round(1)
    st.dataframe(model_avg, use_container_width=True)
    st.divider()

# ── Failure heatmap ──
st.markdown("### Issue Frequency Heatmap")
issue_counts: dict[str, int] = {}
for entry in history:
    for issue in entry["report"].all_issues:
        category = issue.split("]")[0].strip("[") if "]" in issue else "Other"
        issue_counts[category] = issue_counts.get(category, 0) + 1

if issue_counts:
    issue_df = pd.DataFrame(
        list(issue_counts.items()), columns=["Category", "Count"]
    ).sort_values("Count", ascending=False)
    st.bar_chart(issue_df.set_index("Category"))
else:
    st.success("No issues found in any evaluation. ")

st.divider()

# ── Raw data table ──
with st.expander("📋 Raw Evaluation Data"):
    st.dataframe(df, use_container_width=True)
