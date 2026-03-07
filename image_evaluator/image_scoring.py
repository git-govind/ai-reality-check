"""
image_scoring.py
-----------------
Scoring engine for the Image Authenticity Evaluator pipeline.

Aggregates results from all pipeline steps into a single
:class:`ImageEvaluationReport` using the exact weighted formula below.

Exact formula (spec §1)
-----------------------
  authenticity_score =
      0.20 * metadata_score          +
      0.25 * pixel_score             +
      0.35 * (1 - ai_generated_prob) +   (component already × 100 at the end)
      0.10 * consistency_score       +   (skipped → weight redistributed)
      0.10 * reverse_search_score        (skipped → weight redistributed)

All per-step scores are carried as floats in [0, 1] internally;
the final authenticity_score is scaled to [0, 100].

Derived metrics (spec §1)
-------------------------
  ai_likelihood       = ai_generated_prob × 100
  editing_likelihood  = derived from metadata flags + pixel artifacts

Reverse-search score mapping (spec §2)
---------------------------------------
  ReverseSearchResult.similarity ≥ 0.90  →  score = 1.00  (exact match)
  ReverseSearchResult.similarity ≥ 0.70  →  score = 0.80  (high similarity)
  ReverseSearchResult.similarity ≥ 0.30  →  score = 0.50  (partial match)
  otherwise                              →  score = 0.00  (no match)

Weight redistribution
---------------------
  When optional steps are skipped (ran=False), their weight is distributed
  proportionally among the remaining active components so the final score
  is always scaled to a 0–100 range.

Grade thresholds
----------------
  A  ≥ 80   almost certainly authentic
  B  ≥ 65   probably authentic
  C  ≥ 50   uncertain
  D  ≥ 35   probably manipulated / AI-generated
  F  <  35  almost certainly manipulated / AI-generated
"""

from __future__ import annotations

from typing import Optional

from .datatypes import (
    AIArtifactResult,
    ConsistencyResult,
    ImageEvaluationReport,
    MetadataResult,
    PixelForensicsResult,
    ReverseSearchResult,
)

# ---------------------------------------------------------------------------
# Exact weights (spec §1)
# ---------------------------------------------------------------------------

_W_METADATA = 0.20
_W_PIXEL    = 0.25
_W_AI       = 0.35
_W_CONSIST  = 0.10
_W_REVERSE  = 0.10

# ---------------------------------------------------------------------------
# Reverse-search tiered score (spec §2)
# ---------------------------------------------------------------------------

def _reverse_score(result: ReverseSearchResult) -> float:
    """
    Map ReverseSearchResult to an authenticity score in [0, 1].

    Tier mapping (spec §2):
      similarity ≥ 0.90  →  1.00  exact match       (→ 100 pts)
      similarity ≥ 0.70  →  0.80  high similarity   (→  80 pts)
      similarity ≥ 0.30  →  0.50  partial match     (→  50 pts)
      otherwise          →  0.00  no match           (→   0 pts)

    When the step was skipped (ran=False) returns None so the caller
    knows to exclude this component from the weighted average.
    """
    if not result.ran:
        return None  # will be redistributed

    if not result.found:
        return 0.0   # no match

    sim = result.similarity
    if sim >= 0.90:
        return 1.00  # exact match
    if sim >= 0.70:
        return 0.80  # high similarity
    if sim >= 0.30:
        return 0.50  # partial match
    return 0.00


# ---------------------------------------------------------------------------
# Editing likelihood estimation
# ---------------------------------------------------------------------------

_EDIT_KEYWORDS = {
    "ELA":          0.30,
    "quantisation": 0.20,
    "editing":      0.25,
    "photoshop":    0.30,
    "lightroom":    0.20,
    "gimp":         0.20,
    "affinity":     0.20,
    "manipulation": 0.25,
    "timestamp":    0.15,
    "noise":        0.10,
    "FFT":          0.10,
    "software":     0.15,
}


def _estimate_editing_likelihood(
    meta:  MetadataResult,
    pixel: PixelForensicsResult,
) -> float:
    """
    Estimate editing likelihood (0–100) from metadata flags and pixel artifacts.

    Sums weights for edit-related keywords found in the flag / artifact strings,
    then caps at 100.
    """
    raw_signals = meta.flags + pixel.artifacts
    total_weight = 0.0
    for signal in raw_signals:
        sl = signal.lower()
        for keyword, weight in _EDIT_KEYWORDS.items():
            if keyword.lower() in sl:
                total_weight += weight
                break  # count each signal once

    return min(100.0, total_weight * 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    metadata:       MetadataResult,
    pixel:          PixelForensicsResult,
    ai_artifact:    AIArtifactResult,
    consistency:    Optional[ConsistencyResult]    = None,
    reverse_search: Optional[ReverseSearchResult] = None,
) -> ImageEvaluationReport:
    """
    Aggregate pipeline step results into an :class:`ImageEvaluationReport`.

    Implements the exact scoring formula from spec §1.

    Parameters
    ----------
    metadata       : MetadataResult
    pixel          : PixelForensicsResult
    ai_artifact    : AIArtifactResult
    consistency    : ConsistencyResult or None
    reverse_search : ReverseSearchResult or None

    Returns
    -------
    ImageEvaluationReport
    """
    # ── Defaults for optional steps ──────────────────────────────────────────
    if consistency is None:
        consistency = ConsistencyResult(score=0.5, ran=False)
    if reverse_search is None:
        reverse_search = ReverseSearchResult(ran=False)

    # ── Fixed components (always active) ─────────────────────────────────────
    weights: dict[str, float] = {
        "metadata": _W_METADATA,
        "pixel":    _W_PIXEL,
        "ai":       _W_AI,
    }
    scores: dict[str, float] = {
        "metadata": metadata.score,                 # [0, 1]
        "pixel":    pixel.score,                    # [0, 1]
        "ai":       1.0 - ai_artifact.ai_prob,      # [0, 1]
    }

    # ── Optional: consistency ────────────────────────────────────────────────
    if consistency.ran:
        weights["consistency"] = _W_CONSIST
        scores["consistency"]  = consistency.score  # [0, 1]

    # ── Optional: reverse search ─────────────────────────────────────────────
    rev = _reverse_score(reverse_search)
    if rev is not None:
        weights["reverse"] = _W_REVERSE
        scores["reverse"]  = rev                    # [0, 1]

    # ── Normalise weights to sum to 1.0 ──────────────────────────────────────
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}

    # ── Exact spec formula: weighted sum × 100 ───────────────────────────────
    raw_score          = sum(scores[k] * norm_weights[k] for k in norm_weights)
    authenticity_score = float(raw_score * 100.0)

    # ── Derived metrics ──────────────────────────────────────────────────────
    ai_likelihood      = float(ai_artifact.ai_prob * 100.0)
    editing_likelihood = _estimate_editing_likelihood(metadata, pixel)

    # ── Grade ─────────────────────────────────────────────────────────────────
    s = authenticity_score
    if s >= 80:
        grade, summary = "A", "Image appears authentic — no significant anomalies detected."
    elif s >= 65:
        grade, summary = "B", "Image is probably authentic with minor anomalies."
    elif s >= 50:
        grade, summary = "C", "Image authenticity is uncertain — further review recommended."
    elif s >= 35:
        grade, summary = "D", "Image shows significant manipulation or AI-generation signals."
    else:
        grade, summary = "F", "Image is very likely AI-generated or heavily manipulated."

    # ── Evidence bundle (spec §5) ─────────────────────────────────────────────
    evidence = {
        "metadata_flags":        metadata.flags,
        "pixel_artifacts":       pixel.artifacts,
        "ai_artifact_features":  ai_artifact.features,
        "ai_artifact_features_dict": ai_artifact.features_dict,
        "ai_method":             ai_artifact.method,
        "consistency_issues":    consistency.issues if consistency.ran else [],
        "consistency_ran":       consistency.ran,
        "reverse_search_hits":   reverse_search.source_urls,
        "reverse_search_ran":    reverse_search.ran,
        "reverse_search_found":  reverse_search.found,
        "reverse_search_error":  reverse_search.error,
        "component_scores": {
            k: round(scores[k] * 100, 1) for k in scores
        },
        "component_weights": {
            k: round(norm_weights[k], 3) for k in norm_weights
        },
    }

    return ImageEvaluationReport(
        authenticity_score = round(authenticity_score, 1),
        ai_likelihood      = round(ai_likelihood,      1),
        editing_likelihood = round(editing_likelihood,  1),
        grade              = grade,
        summary            = summary,
        evidence           = evidence,
    )
