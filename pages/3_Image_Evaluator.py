"""
pages/2_Image_Evaluator.py — Image Authenticity Evaluator UI

Standalone page; does not touch the text evaluation pipeline.
"""
from __future__ import annotations

import io
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

import streamlit as st
from PIL import Image

from image_evaluator import evaluate_image
from image_evaluator.datatypes import ImageEvaluationReport
from utils.scoring_utils import score_to_color
from utils.cache_utils import make_cache_key

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Evaluator — VeritasIQ",
    page_icon="🖼️",
    layout="wide",
)

# ── Shared CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .img-title  { font-size:2rem; font-weight:800; }
    .img-sub    { color:#888; font-size:1rem; margin-top:-0.5rem; }
    .viq-brand  { display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.15rem; }
    .viq-name   { font-size:0.95rem; font-weight:700; color:#94a3b8; letter-spacing:-0.01em; }
    .viq-tag    { font-size:0.7rem; color:#4a5568; font-style:italic; }
    .grade-box  {
        border-radius:16px; padding:18px 28px; text-align:center;
        font-size:3rem; font-weight:900; line-height:1;
    }
    .grade-A { background:#14532d; color:#4ade80; border:2px solid #4ade80; }
    .grade-B { background:#1e3a5f; color:#60a5fa; border:2px solid #60a5fa; }
    .grade-C { background:#2d2a1a; color:#fbbf24; border:2px solid #fbbf24; }
    .grade-D { background:#3a1a1a; color:#fb923c; border:2px solid #fb923c; }
    .grade-F { background:#3a1010; color:#f87171; border:2px solid #f87171; }
    .ev-section { background:#111827; border:1px solid #1f2937;
                  border-radius:10px; padding:12px 16px; margin-bottom:8px; }
    .ev-label   { font-size:0.75rem; color:#9ca3af; text-transform:uppercase;
                  letter-spacing:.06em; margin-bottom:4px; }
    .flag-item  { color:#fbbf24; font-size:0.88rem; padding:3px 0; }
    .ok-item    { color:#4ade80; font-size:0.88rem; }
    .feat-item  { color:#93c5fd; font-size:0.82rem; padding:2px 0; }
    [data-testid="stAppDeployButton"] { display:none; }
    div[data-testid="stButton"] button {
        padding-top: 0.4rem !important;
        padding-bottom: 0.4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────────
if "img_history" not in st.session_state:
    st.session_state["img_history"] = []

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display: none" class="viq-brand">'
    '<span class="viq-name">🔍 VeritasIQ</span>'
    '<span class="viq-tag">Where Intelligence Meets Integrity</span>'
    '</div>'
    '<div class="img-title">🖼️ Image Authenticity Evaluator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="img-sub">Detect AI-generated images, manipulation, and editing artefacts</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Input area ───────────────────────────────────────────────────────────────
col_upload, col_caption = st.columns([1, 1], gap="large")

with col_upload:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "tiff", "bmp"],
        help="JPEG, PNG, WEBP, TIFF or BMP",
    )

with col_caption:
    caption = st.text_area(
        "Caption / prompt (optional)",
        placeholder="Describe what this image is supposed to show…\n"
                    "e.g. 'A golden retriever running on a beach at sunset'",
        height=120,
        help="Providing a caption enables Step 4: image–text consistency check.",
    )

# ── Evaluate button row ───────────────────────────────────────────────────────
is_evaluating = st.session_state.get("img_evaluating", False)

btn_col, cancel_col, _ = st.columns([2, 1, 5])
with btn_col:
    run_btn = st.button(
        "🔍 Evaluate Image",
        disabled=(uploaded is None or is_evaluating),
        use_container_width=True,
        type="primary",
    )
with cancel_col:
    cancel_btn = st.button(
        "✖ Cancel",
        disabled=not is_evaluating,
        use_container_width=True,
    )

if cancel_btn:
    st.session_state["img_evaluating"] = False
    st.session_state["img_cancel"] = True
    st.rerun()

if run_btn and uploaded:
    st.session_state["img_evaluating"] = True
    st.session_state["img_cancel"]     = False
    st.session_state["img_report"]     = None
    st.session_state["img_filename"]   = uploaded.name
    st.session_state["img_caption"]    = caption.strip() if caption else ""
    st.rerun()

# ── Pipeline execution ────────────────────────────────────────────────────────
if st.session_state.get("img_evaluating") and uploaded:
    image_bytes = uploaded.read()

    # ── Cache lookup ─────────────────────────────────────────────────────────
    # Key = sha256(image_bytes ‖ caption) so the same image with a different
    # caption gets a separate cache entry (consistency result would differ).
    _caption_for_key = st.session_state.get("img_caption", "")
    _cache_key = make_cache_key(image_bytes, _caption_for_key)

    if "img_cache" not in st.session_state:
        st.session_state["img_cache"] = {}

    if _cache_key in st.session_state["img_cache"]:
        # Serve from cache — skip the entire pipeline
        st.info("Result served from cache (same image + caption evaluated before).")
        st.session_state["img_report"]     = st.session_state["img_cache"][_cache_key]
        st.session_state["img_bytes"]      = image_bytes
        st.session_state["img_evaluating"] = False
        st.rerun()

    # Cache miss — run the full pipeline
    with st.status("Analysing image…", expanded=True, state="running") as status:

        def _cancelled() -> bool:
            return st.session_state.get("img_cancel", False)

        # ── Phase 1: metadata (fast, ~0.1 s) — always runs first ─────────────
        # Running metadata before the heavier steps lets us detect AI-generator
        # software tags and skip the AI classifier entirely when found.
        st.write("⬜ Step 1 — Reading metadata …")
        from image_evaluator import metadata_checker
        meta_result = metadata_checker.run(image_bytes)
        if _cancelled():
            status.update(label="Cancelled.", state="error", expanded=False)
            st.session_state["img_evaluating"] = False
            st.stop()
        st.write("✅ Step 1 — Metadata")

        # ── Phase 2: parallel pixel + AI (or early exit) ──────────────────────
        from concurrent.futures import ThreadPoolExecutor
        from image_evaluator import pixel_forensics, ai_artifact_classifier
        from image_evaluator.datatypes import AIArtifactResult, ConsistencyResult

        if meta_result.detected_ai_generator:
            # ── Early exit: metadata already confirms AI origin ───────────────
            # Pixel forensics still runs — needed for editing_likelihood.
            # AI classifier is skipped; ai_prob is set to 1.0 (certainty).
            # Consistency (CLIP) is skipped — irrelevant for confirmed AI.
            gen_name = meta_result.detected_ai_generator
            st.write(
                f"⚡ Early exit — AI generator confirmed in metadata: **{gen_name}**"
            )

            st.write("⬜ Step 2 — Pixel forensics …")
            pixel_result = pixel_forensics.run(image_bytes)
            if _cancelled():
                status.update(label="Cancelled.", state="error", expanded=False)
                st.session_state["img_evaluating"] = False
                st.stop()
            st.write("✅ Step 2 — Pixel forensics")

            ai_result = AIArtifactResult(
                ai_prob         = 1.0,
                confidence_band = 0.0,   # zero uncertainty — metadata is definitive
                method          = f"metadata:ai_generator",
                features        = [
                    f"AI generation software confirmed in metadata: '{gen_name}'",
                    "AI classifier skipped — metadata evidence is definitive",
                ],
                features_dict   = {
                    "metadata_ai_generator": gen_name,
                    "ai_prob_final":         1.0,
                    "confidence_band":       0.0,
                },
            )
            st.write(f"⏭️ Step 3 — Skipped (AI origin confirmed: {gen_name})")

            consistency_result = ConsistencyResult(ran=False)
            st.write("⏭️ Step 4 — Skipped (AI origin confirmed)")

        else:
            # ── Normal path: pixel forensics + AI classifier in parallel ──────
            st.write("⬜ Steps 2–3 — Parallel analysis (pixel forensics · AI classifier) …")
            with ThreadPoolExecutor(max_workers=2) as _pool:
                _fut_pixel = _pool.submit(pixel_forensics.run,        image_bytes)
                _fut_ai    = _pool.submit(ai_artifact_classifier.run, image_bytes)
                pixel_result = _fut_pixel.result()
                ai_result    = _fut_ai.result()

            if _cancelled():
                status.update(label="Cancelled.", state="error", expanded=False)
                st.session_state["img_evaluating"] = False
                st.stop()
            st.write("✅ Step 2 — Pixel forensics")
            st.write("✅ Step 3 — AI artefact classifier")

            if caption and caption.strip():
                st.write("⬜ Step 4 — Image–text consistency …")
                from image_evaluator import image_text_consistency
                consistency_result = image_text_consistency.run(
                    image_bytes, caption=caption.strip()
                )
                st.write("✅ Step 4 — Consistency check")
            else:
                consistency_result = ConsistencyResult(ran=False)
                st.write("⏭️ Step 4 — Skipped (no caption)")

        if _cancelled():
            status.update(label="Cancelled.", state="error", expanded=False)
            st.session_state["img_evaluating"] = False
            st.stop()

        st.write("⬜ Step 5 — Reverse image search …")
        from image_evaluator import reverse_image_search
        reverse_result = reverse_image_search.run(image_bytes)
        if reverse_result.ran:
            st.write("✅ Step 5 — Reverse search")
        else:
            st.write("⏭️ Step 5 — Skipped (no API key configured)")

        st.write("⬜ Aggregating scores …")
        from image_evaluator import image_scoring
        detected_type = image_scoring.detect_image_type(image_bytes)
        report: ImageEvaluationReport = image_scoring.aggregate(
            metadata       = meta_result,
            pixel          = pixel_result,
            ai_artifact    = ai_result,
            consistency    = consistency_result,
            reverse_search = reverse_result,
            image_type     = detected_type,
        )
        st.write("✅ Done")
        status.update(label="Evaluation complete", state="complete", expanded=False)

    st.session_state["img_report"]     = report
    st.session_state["img_bytes"]      = image_bytes
    st.session_state["img_evaluating"] = False

    # ── Write to cache ───────────────────────────────────────────────────────
    st.session_state["img_cache"][_cache_key] = report

    # ── Append to history ────────────────────────────────────────────────────
    st.session_state["img_history"].append({
        "ts":               datetime.now().isoformat(),
        "filename":         st.session_state.get("img_filename", "unknown"),
        "caption":          st.session_state.get("img_caption", ""),
        "report":           report,
        "authenticity":     report.authenticity_score,
        "ai_likelihood":    report.ai_likelihood,
        "editing_likelihood": report.editing_likelihood,
        "grade":            report.grade,
    })

# ── Results ───────────────────────────────────────────────────────────────────
report: ImageEvaluationReport | None = st.session_state.get("img_report")
img_bytes: bytes | None = st.session_state.get("img_bytes")

if report and img_bytes:
    st.markdown("## 📊 Results")

    res_left, res_right = st.columns([1, 2], gap="large")

    # ── Left: image preview + grade ──────────────────────────────────────────
    with res_left:
        st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)

        grade = report.grade
        st.markdown(
            f'<div class="grade-box grade-{grade}">{grade}</div>',
            unsafe_allow_html=True,
        )
        st.caption(report.summary)

    # ── Right: score cards ────────────────────────────────────────────────────
    with res_right:
        m1, m2, m3 = st.columns(3)

        def _colour(val: float, invert: bool = False) -> str:
            return score_to_color(val, invert=invert)

        with m1:
            c         = _colour(report.authenticity_score)
            auth_band = round(report.authenticity_band)
            st.markdown(
                f"<div style='font-size:.8rem;color:#aaa'>Authenticity</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{c}'>"
                f"{report.authenticity_score:.0f}"
                f"<span style='font-size:1rem'>/100</span></div>"
                f"<div style='font-size:.72rem;color:#6b7280'>± {auth_band} pts</div>",
                unsafe_allow_html=True,
            )
        with m2:
            c           = _colour(report.ai_likelihood, invert=True)
            ai_band_pct = round(report.ai_likelihood_band)
            st.markdown(
                f"<div style='font-size:.8rem;color:#aaa'>AI Likelihood</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{c}'>"
                f"{report.ai_likelihood:.0f}<span style='font-size:1rem'>%</span></div>"
                f"<div style='font-size:.72rem;color:#6b7280'>± {ai_band_pct}%</div>",
                unsafe_allow_html=True,
            )
        with m3:
            c         = _colour(report.editing_likelihood, invert=True)
            edit_band = round(report.editing_likelihood_band)
            st.markdown(
                f"<div style='font-size:.8rem;color:#aaa'>Editing Likelihood</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{c}'>"
                f"{report.editing_likelihood:.0f}<span style='font-size:1rem'>%</span></div>"
                f"<div style='font-size:.72rem;color:#6b7280'>± {edit_band}%</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # OOD warning — shown when CLIP detects an image style outside the
        # AI detector's training distribution (anime, 3D render, illustration, screenshot)
        ood_warn = report.evidence.get("ood_warning", "")
        if ood_warn:
            st.warning(ood_warn)

        # Detected image-type badge
        img_type = report.evidence.get("image_type", "photo")
        type_icon = {"photo": "📷", "illustration": "🎨", "screenshot": "🖥️"}.get(img_type, "📷")
        st.markdown(
            f"<div style='font-size:.78rem;color:#9ca3af;margin-bottom:6px'>"
            f"Detected type: <span style='color:#e5e7eb;font-weight:600'>"
            f"{type_icon} {img_type.capitalize()}</span>"
            f" &nbsp;·&nbsp; weights adjusted accordingly</div>",
            unsafe_allow_html=True,
        )

        # Top signals
        top_sigs = report.top_signals
        if top_sigs:
            st.markdown(
                "<div style='font-size:.78rem;color:#9ca3af;text-transform:uppercase;"
                "letter-spacing:.06em;margin:10px 0 4px'>Key signals</div>",
                unsafe_allow_html=True,
            )
            for sig in top_sigs:
                st.markdown(
                    f"<div class='flag-item' style='padding:2px 0'>⚠ {sig}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

        # Component score breakdown
        comp = report.evidence.get("component_scores", {})
        wts  = report.evidence.get("component_weights", {})
        if comp:
            st.markdown("**Component scores**")
            for key, val in comp.items():
                w   = wts.get(key, 0)
                c   = _colour(float(val))
                bar = int(val / 100 * 20)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
                    f"<span style='width:120px;font-size:.82rem;color:#aaa;text-transform:capitalize'>{key}</span>"
                    f"<span style='flex:1;background:#1f2937;border-radius:4px;height:8px'>"
                    f"<span style='display:block;width:{val}%;height:8px;border-radius:4px;background:{c}'></span>"
                    f"</span>"
                    f"<span style='width:50px;text-align:right;font-size:.82rem;color:{c}'>{val:.0f}</span>"
                    f"<span style='width:50px;text-align:right;font-size:.72rem;color:#555'>w={w:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Explanation ──────────────────────────────────────────────────────────
    if getattr(report, "explanation", ""):
        st.info(report.explanation)

    # ── Evidence tabs ─────────────────────────────────────────────────────────
    st.markdown("### 🔬 Evidence Breakdown")
    ev = report.evidence
    tab_meta, tab_pixel, tab_ai, tab_consist, tab_rev = st.tabs([
        "📋 Metadata",
        "🔎 Pixel Forensics",
        "🤖 AI Artefacts",
        "📝 Consistency",
        "🌐 Reverse Search",
    ])

    with tab_meta:
        img_type = ev.get("image_type", "photo")
        type_icon = {"photo": "📷", "illustration": "🎨", "screenshot": "🖥️"}.get(img_type, "📷")
        st.caption(f"Detected image type: {type_icon} **{img_type.capitalize()}**")
        flags = ev.get("metadata_flags", [])
        if flags:
            for f in flags:
                st.markdown(f'<div class="flag-item">⚠ {f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-item">✓ No metadata anomalies detected</div>', unsafe_allow_html=True)

    with tab_pixel:
        arts = ev.get("pixel_artifacts", [])
        if arts:
            for a in arts:
                st.markdown(f'<div class="flag-item">⚠ {a}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-item">✓ No pixel-level artefacts detected</div>', unsafe_allow_html=True)

    with tab_ai:
        ood = ev.get("ood_warning", "")
        if ood:
            st.warning(ood)
        method = ev.get("ai_method", "heuristic")
        band   = ev.get("ai_confidence_band", 0.0)
        st.caption(
            f"Method: **{method}** &nbsp;|&nbsp; "
            f"Confidence band: **± {band * 100:.0f}%**"
        )
        feats = ev.get("ai_artifact_features", [])
        for feat in feats:
            st.markdown(f'<div class="feat-item">• {feat}</div>', unsafe_allow_html=True)

    with tab_consist:
        if ev.get("consistency_ran"):
            issues = ev.get("consistency_issues", [])
            if issues:
                for iss in issues:
                    st.markdown(f'<div class="flag-item">⚠ {iss}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-item">✓ Image is consistent with caption</div>', unsafe_allow_html=True)
        else:
            st.info("Consistency check was skipped — provide a caption to enable it.")

    with tab_rev:
        if ev.get("reverse_search_ran"):
            if ev.get("reverse_search_found"):
                st.success(f"Image found online ({len(ev.get('reverse_search_hits', []))} sources)")
                for url in ev.get("reverse_search_hits", []):
                    st.markdown(f"- {url}")
            elif ev.get("reverse_search_error"):
                st.error(f"API error: {ev['reverse_search_error']}")
                st.caption("Common causes: Cloud Vision API not enabled in your Google Cloud project, "
                           "or the key has no Vision API permissions.")
            else:
                st.info("No matching images found online.")
        else:
            st.info(
                "Reverse search was skipped — set `GOOGLE_API_KEY`, "
                "`BING_SEARCH_KEY`, or `SERPAPI_KEY` to enable it."
            )

# ── Evaluation History ────────────────────────────────────────────────────────
img_history = st.session_state.get("img_history", [])
if img_history:
    st.markdown("---")
    st.markdown("## 📜 Evaluation History")
    for entry in reversed(img_history):
        r    = entry["report"]
        grad = entry["grade"]
        grade_colours = {
            "A": "#4ade80", "B": "#60a5fa",
            "C": "#fbbf24", "D": "#fb923c", "F": "#f87171",
        }
        c = grade_colours.get(grad, "#aaa")
        ts = entry["ts"][:19].replace("T", " ")
        with st.expander(
            f"{ts}  ·  {entry['filename']}  ·  Grade {grad}  ·  "
            f"Auth {entry['authenticity']:.0f}  ·  AI {entry['ai_likelihood']:.0f}%",
            expanded=False,
        ):
            h1, h2, h3 = st.columns(3)
            h1.metric("Authenticity", f"{entry['authenticity']:.0f}/100")
            h2.metric("AI Likelihood", f"{entry['ai_likelihood']:.0f}%")
            h3.metric("Editing Likelihood", f"{entry['editing_likelihood']:.0f}%")
            st.caption(r.summary)
            if entry.get("caption"):
                st.caption(f"Caption: {entry['caption']}")
            meta_flags = r.evidence.get("metadata_flags", [])
            pixel_arts = r.evidence.get("pixel_artifacts", [])
            all_flags  = meta_flags + pixel_arts
            if all_flags:
                st.markdown("**Flags:**")
                for fl in all_flags:
                    st.markdown(f'<div class="flag-item">⚠ {fl}</div>', unsafe_allow_html=True)
