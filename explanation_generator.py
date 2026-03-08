"""
explanation_generator.py
------------------------
Generates a concise, human-readable explanation for any EvaluationReportBase
subclass produced by the AI Reality Check pipeline.

Public API
----------
  generate_explanation(report: EvaluationReportBase) -> str
      Returns a plain-text paragraph (3–5 sentences) covering:
        · the overall verdict / grade,
        · key strengths,
        · notable issues,
        · a brief pointer to supporting evidence.

Usage
-----
    from explanation_generator import generate_explanation
    report.explanation = generate_explanation(report)

Dispatch strategy
-----------------
Uses ``hasattr`` instead of ``isinstance`` to avoid importing the concrete
subclasses (which would create circular-import paths).  Any object that has
an ``authenticity_score`` attribute is treated as an image report; any object
with a ``confidence_score`` attribute is treated as a text report; everything
else falls back to the generic base-field renderer.
"""

from __future__ import annotations

from evaluation_report_base import EvaluationReportBase


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(report: EvaluationReportBase) -> str:
    """Return a short, human-readable explanation of *report*.

    Covers: overall verdict, key strengths, notable issues, and a pointer to
    the primary supporting evidence.  Always returns a non-empty string.
    """
    if hasattr(report, "authenticity_score"):
        return _explain_image(report)
    if hasattr(report, "confidence_score"):
        return _explain_text(report)
    return _explain_generic(report)


# ---------------------------------------------------------------------------
# Grade quality labels (shared)
# ---------------------------------------------------------------------------

_GRADE_QUALITY: dict[str, str] = {
    "A": "excellent",
    "B": "good",
    "C": "uncertain",
    "D": "concerning",
    "F": "very poor",
}


# ---------------------------------------------------------------------------
# Image pipeline helper
# ---------------------------------------------------------------------------

def _explain_image(report) -> str:
    """Build an explanation for an ImageEvaluationReport."""
    grade   = (report.grade or "?").strip()
    auth    = report.authenticity_score
    ai      = report.ai_likelihood
    edit    = report.editing_likelihood
    ev: dict = report.evidence or {}
    parts: list[str] = []

    # ── Verdict ──────────────────────────────────────────────────────────────
    quality = _GRADE_QUALITY.get(grade, "unknown")
    parts.append(
        f"Authenticity grade {grade} ({quality}): "
        f"authenticity {auth:.0f}%, AI-generation likelihood {ai:.0f}%, "
        f"editing likelihood {edit:.0f}%."
    )

    # ── Strengths — inferred from absence of bad signals ─────────────────────
    strengths: list[str] = []
    n_pixel_arts = len(ev.get("pixel_artifacts", []))
    if auth >= 65 and ai < 35 and n_pixel_arts == 0:
        strengths.append("no strong AI-generation signals detected")
    if edit < 30:
        strengths.append("minimal editing artefacts present")
    if not ev.get("metadata_flags"):
        strengths.append("metadata is consistent with an authentic camera image")
    if not ev.get("pixel_artifacts"):
        strengths.append("pixel forensics found no anomalies")
    if strengths:
        parts.append("Strengths: " + "; ".join(strengths) + ".")

    # ── Key issues — top_signals first, fall back to raw evidence lists ───────
    issues = list(report.top_signals or [])
    if not issues:
        issues = (
            list(ev.get("metadata_flags", []))[:2]
            + list(ev.get("pixel_artifacts", []))[:1]
        )
    if issues:
        parts.append("Key concerns: " + "; ".join(issues[:3]) + ".")

    # ── Evidence pointers ─────────────────────────────────────────────────────
    ev_refs: list[str] = []
    n_meta = len(ev.get("metadata_flags", []))
    n_px   = len(ev.get("pixel_artifacts", []))
    if n_meta:
        ev_refs.append(f"{n_meta} metadata flag{'s' if n_meta != 1 else ''}")
    if n_px:
        ev_refs.append(f"{n_px} pixel artefact{'s' if n_px != 1 else ''}")
    if ev.get("consistency_ran") and ev.get("consistency_issues"):
        n = len(ev["consistency_issues"])
        ev_refs.append(f"{n} caption-consistency issue{'s' if n != 1 else ''}")
    if ev.get("reverse_search_found"):
        ev_refs.append("a reverse-image-search match was found")
    if ev_refs:
        parts.append("Supporting evidence: " + ", ".join(ev_refs) + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Text pipeline helper
# ---------------------------------------------------------------------------

def _explain_text(report) -> str:
    """Build an explanation for an EvaluationReport (text pipeline)."""
    conf  = report.confidence_score
    grade = report.grade or "?"
    # grade may be a full label such as "A — Excellent"; extract the letter
    grade_letter = grade[0] if grade else "?"
    quality = _GRADE_QUALITY.get(grade_letter, "unknown")
    parts: list[str] = []

    # ── Verdict ──────────────────────────────────────────────────────────────
    parts.append(
        f"Confidence {conf:.0f}/100 (Grade {grade_letter}, {quality}): "
        f"the AI response scores {quality} for reliability and quality."
    )

    # ── Strengths / weaknesses by sub-score ───────────────────────────────────
    score_map = {
        "accuracy":     getattr(report, "accuracy_score",     None),
        "safety":       getattr(report, "safety_score",       None),
        "consistency":  getattr(report, "consistency_score",  None),
        "clarity":      getattr(report, "clarity_score",      None),
        "completeness": getattr(report, "completeness_score", None),
        "bias":         getattr(report, "bias_score",         None),
    }
    strong = [k for k, v in score_map.items() if v is not None and v >= 80]
    weak   = [k for k, v in score_map.items() if v is not None and v < 60]

    if strong:
        s = "score" if len(strong) == 1 else "scores"
        parts.append(f"Strengths: {', '.join(strong)} {s} are high.")
    if weak:
        s = "score" if len(weak) == 1 else "scores"
        parts.append(f"Weaknesses: {', '.join(weak)} {s} are low.")

    # ── Top issues ───────────────────────────────────────────────────────────
    all_issues: list[str] = list(
        getattr(report, "all_issues", None) or report.issues or []
    )
    if all_issues:
        top = all_issues[:3]
        tail = "…" if len(all_issues) > 3 else "."
        parts.append("Top issues: " + "; ".join(top) + tail)

    # ── Evidence pointers ─────────────────────────────────────────────────────
    ev_refs: list[str] = []
    fd = getattr(report, "factual_details", None) or []
    if fd:
        n = len(fd)
        ev_refs.append(f"{n} factual claim{'s' if n != 1 else ''} checked")
    if getattr(report, "critique_text", ""):
        ev_refs.append("LLM self-critique available")
    if ev_refs:
        parts.append("Evidence: " + "; ".join(ev_refs) + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

def _explain_generic(report: EvaluationReportBase) -> str:
    """Fallback explanation using only EvaluationReportBase fields."""
    parts: list[str] = []

    if report.grade:
        parts.append(f"Grade: {report.grade}.")

    named = {
        k: v for k, v in (report.scores or {}).items()
        if isinstance(v, (int, float))
    }
    if named:
        parts.append("Scores — " + ", ".join(f"{k} {v:.0f}" for k, v in named.items()) + ".")

    issues = report.issues or []
    if issues:
        top  = issues[:3]
        tail = "…" if len(issues) > 3 else "."
        parts.append("Issues: " + "; ".join(top) + tail)

    return " ".join(parts) or "No explanation available."
