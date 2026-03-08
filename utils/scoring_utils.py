"""
utils/scoring_utils.py
-----------------------
Shared scoring and math helpers used across both evaluation pipelines.

Public API
----------
  clamp(x, lo, hi)               – clamp x to [lo, hi]
  clamp100(x)                    – clamp x to [0.0, 100.0]
  letter_grade(score, thresholds) – map 0-100 score → "A"|"B"|"C"|"D"|"F"
  normalize_weights(weights)     – scale dict values to sum to 1.0
  weighted_average(scores, weights) – normalised dot-product
  score_to_color(score, ...)     – map score to CSS hex colour string
"""
from __future__ import annotations

from typing import Dict, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    """Return *x* clamped to the closed interval [*lo*, *hi*]."""
    return float(max(lo, min(hi, x)))


def clamp100(x: float) -> float:
    """Return *x* clamped to [0.0, 100.0]."""
    return clamp(x, 0.0, 100.0)


def letter_grade(
    score: float,
    thresholds: Tuple[float, float, float, float] = (80.0, 65.0, 50.0, 35.0),
) -> str:
    """
    Map a 0–100 *score* to a letter grade.

    Parameters
    ----------
    score      : float   0–100
    thresholds : tuple   (A_min, B_min, C_min, D_min) — defaults match the
                         image-scorer config values and are close to the
                         text-scorer values.

    Returns
    -------
    ``"A"`` | ``"B"`` | ``"C"`` | ``"D"`` | ``"F"``
    """
    a, b, c, d = thresholds
    if score >= a:
        return "A"
    if score >= b:
        return "B"
    if score >= c:
        return "C"
    if score >= d:
        return "D"
    return "F"


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Return a copy of *weights* scaled so that all values sum to 1.0.

    If the total is zero every value is set to 0.0 (safe no-op).
    """
    total = sum(weights.values())
    if total == 0.0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def weighted_average(
    scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Compute a weighted average of *scores* using *weights*.

    Weights are normalised internally so they do not need to sum to 1.0.
    Both dicts must share the same keys.

    Returns
    -------
    float  in the same value range as the score values.
    """
    norm = normalize_weights(weights)
    return sum(scores[k] * norm[k] for k in norm)


def score_to_color(
    score: float,
    invert: bool = False,
    thresholds: Tuple[float, float] = (70.0, 45.0),
    colors: Tuple[str, str, str] = ("#4ade80", "#fbbf24", "#f87171"),
) -> str:
    """
    Map a 0–100 *score* to a CSS hex colour string.

    Parameters
    ----------
    score      : float  0–100
    invert     : bool   When ``True``, high scores are treated as bad
                        (e.g. AI-likelihood scores where 100 = certain AI).
    thresholds : tuple  (high_min, mid_min) — boundary values for the two
                        upper colour bands.  Default: (70, 45).
    colors     : tuple  (high_color, mid_color, low_color) — CSS hex strings.
                        Default: green / amber / red.

    Returns
    -------
    str  One of the three *colors* entries.

    Examples
    --------
    # Image evaluator (default thresholds, amber mid-tone):
    score_to_color(82)                              # → "#4ade80"

    # Text evaluator (tighter thresholds, orange mid-tone):
    score_to_color(60, thresholds=(78, 55),
                   colors=("#4ade80", "#fb923c", "#f87171"))  # → "#fb923c"
    """
    hi, mid = thresholds
    v = (100.0 - score) if invert else score
    if v >= hi:
        return colors[0]
    if v >= mid:
        return colors[1]
    return colors[2]
