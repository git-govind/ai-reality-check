"""
pages/1_Dashboard.py — Reliability & Trend Dashboard
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard — AI Reality Check", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"] { display:none; }
    .grade-chip {
        display:inline-block; border-radius:8px; padding:2px 10px;
        font-weight:700; font-size:0.82rem; margin-right:4px;
    }
    .g-A { background:#14532d; color:#4ade80; }
    .g-B { background:#1e3a5f; color:#60a5fa; }
    .g-C { background:#2d2a1a; color:#fbbf24; }
    .g-D { background:#3a1a1a; color:#fb923c; }
    .g-F { background:#3a1010; color:#f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📊 Reliability Dashboard")
st.caption("Aggregated statistics across all evaluations this session.")

tab_text, tab_image = st.tabs(["📝 Text Evaluations", "🖼️ Image Evaluations"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Text Evaluations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_text:
    history = st.session_state.get("history", [])

    if not history:
        st.info("No text evaluations yet. Go to the main page and run some prompts first.")
    else:
        # ── Build dataframe ──────────────────────────────────────────────────
        rows = []
        for entry in history:
            r = entry["report"]
            rows.append({
                "timestamp":   entry["ts"][:19].replace("T", " "),
                "model":       entry["model"],
                "prompt":      entry["prompt"][:50],
                "confidence":  r.confidence_score,
                "accuracy":    r.accuracy_score,
                "consistency": r.consistency_score,
                "safety":      r.safety_score,
                "bias":        r.bias_score,
                "clarity":     r.clarity_score,
                "completeness":r.completeness_score,
                "grade":       r.grade,
                "issues":      len(r.all_issues),
            })

        df = pd.DataFrame(rows)

        # ── Summary metrics ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Evaluations",    len(df))
        col2.metric("Avg Confidence", f"{df['confidence'].mean():.1f}/100")
        col3.metric("Avg Accuracy",   f"{df['accuracy'].mean():.1f}/100")
        col4.metric("Avg Safety",     f"{df['safety'].mean():.1f}/100")

        st.divider()

        # ── Confidence trend ─────────────────────────────────────────────────
        st.markdown("### Confidence Score Over Time")
        st.line_chart(df.set_index("timestamp")[["confidence"]])

        st.divider()

        # ── Score breakdown ──────────────────────────────────────────────────
        st.markdown("### Score Breakdown Per Evaluation")
        score_cols = ["accuracy", "consistency", "safety", "bias", "clarity", "completeness"]
        st.bar_chart(df[score_cols])

        st.divider()

        # ── Model comparison ─────────────────────────────────────────────────
        if df["model"].nunique() > 1:
            st.markdown("### Model Comparison (Average Scores)")
            model_avg = df.groupby("model")[score_cols + ["confidence"]].mean().round(1)
            st.dataframe(model_avg, use_container_width=True)
            st.divider()

        # ── Issue frequency ──────────────────────────────────────────────────
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
            st.success("No issues found in any evaluation.")

        st.divider()

        # ── Raw data ─────────────────────────────────────────────────────────
        with st.expander("📋 Raw Evaluation Data"):
            st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Image Evaluations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_image:
    img_history = st.session_state.get("img_history", [])

    if not img_history:
        st.info("No image evaluations yet. Go to **Image Evaluator** and analyse some images first.")
    else:
        # ── Build dataframe ──────────────────────────────────────────────────
        irows = []
        for entry in img_history:
            r = entry["report"]
            irows.append({
                "timestamp":          entry["ts"][:19].replace("T", " "),
                "filename":           entry["filename"][:30],
                "caption":            entry.get("caption", "")[:40] or "—",
                "authenticity":       r.authenticity_score,
                "ai_likelihood":      r.ai_likelihood,
                "editing_likelihood": r.editing_likelihood,
                "grade":              r.grade,
                "metadata_flags":     len(r.evidence.get("metadata_flags",   [])),
                "pixel_artifacts":    len(r.evidence.get("pixel_artifacts",  [])),
                "consistency_issues": len(r.evidence.get("consistency_issues", [])),
                "total_flags":        (
                    len(r.evidence.get("metadata_flags",   [])) +
                    len(r.evidence.get("pixel_artifacts",  [])) +
                    len(r.evidence.get("consistency_issues", []))
                ),
            })

        idf = pd.DataFrame(irows)

        # ── Summary metrics ──────────────────────────────────────────────────
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        ic1.metric("Images Evaluated",    len(idf))
        ic2.metric("Avg Authenticity",    f"{idf['authenticity'].mean():.1f}/100")
        ic3.metric("Avg AI Likelihood",   f"{idf['ai_likelihood'].mean():.1f}%")
        ic4.metric("Avg Edit Likelihood", f"{idf['editing_likelihood'].mean():.1f}%")
        grade_counts = idf["grade"].value_counts()
        ic5.metric("Most Common Grade",   grade_counts.idxmax() if not grade_counts.empty else "—")

        st.divider()

        # ── Score trends ─────────────────────────────────────────────────────
        st.markdown("### Score Trends Over Time")
        st.line_chart(
            idf.set_index("timestamp")[["authenticity", "ai_likelihood", "editing_likelihood"]]
        )

        st.divider()

        # ── Per-image breakdown ──────────────────────────────────────────────
        st.markdown("### Per-Image Score Breakdown")
        st.bar_chart(
            idf.set_index("filename")[["authenticity", "ai_likelihood", "editing_likelihood"]]
        )

        st.divider()

        col_grade, col_flags = st.columns(2)

        # ── Grade distribution ───────────────────────────────────────────────
        with col_grade:
            st.markdown("### Grade Distribution")
            grade_df = (
                idf["grade"]
                .value_counts()
                .reindex(["A", "B", "C", "D", "F"])
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            grade_df.columns = ["Grade", "Count"]
            st.bar_chart(grade_df.set_index("Grade"))

        # ── Flag frequency ───────────────────────────────────────────────────
        with col_flags:
            st.markdown("### Flag Frequency")
            flag_sums = idf[["metadata_flags", "pixel_artifacts", "consistency_issues"]].sum()
            flag_sums = flag_sums[flag_sums > 0].rename(index={
                "metadata_flags":     "Metadata",
                "pixel_artifacts":    "Pixel",
                "consistency_issues": "Consistency",
            })
            if not flag_sums.empty:
                st.bar_chart(flag_sums)
            else:
                st.success("No flags raised across all evaluations.")

        st.divider()

        # ── Most suspicious ──────────────────────────────────────────────────
        st.markdown("### Most Suspicious Images")
        for _, row in idf.sort_values("authenticity").head(5).iterrows():
            grade    = row["grade"]
            chip_cls = f"g-{grade}"
            st.markdown(
                f'<span class="grade-chip {chip_cls}">{grade}</span> '
                f'**{row["filename"]}** — '
                f'Auth: **{row["authenticity"]:.0f}** · '
                f'AI: {row["ai_likelihood"]:.0f}% · '
                f'Edit: {row["editing_likelihood"]:.0f}%',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Raw data ─────────────────────────────────────────────────────────
        with st.expander("📋 Raw Image Evaluation Data"):
            st.dataframe(
                idf[[
                    "timestamp", "filename", "caption", "grade",
                    "authenticity", "ai_likelihood", "editing_likelihood",
                    "metadata_flags", "pixel_artifacts", "total_flags",
                ]],
                use_container_width=True,
            )
