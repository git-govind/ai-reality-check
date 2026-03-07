"""
Scoring Engine
Aggregates scores from all evaluation modules into a unified report.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.factual_checker import FactualCheckResult
from src.evaluation.consistency_checker import ConsistencyResult
from src.evaluation.bias_safety_checker import BiasResult
from src.evaluation.clarity_scorer import ClarityResult


@dataclass
class EvaluationReport:
    # Sub-scores (0–100)
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    safety_score: float = 0.0
    bias_score: float = 0.0
    clarity_score: float = 0.0
    completeness_score: float = 0.0

    # Weighted final confidence score
    confidence_score: float = 0.0

    # Grade label
    grade: str = "N/A"

    # Human-readable summaries from each checker
    factual_summary: str = ""
    consistency_summary: str = ""
    bias_safety_summary: str = ""
    clarity_summary: str = ""

    # All issues collected across checkers
    all_issues: list[str] = None  # type: ignore[assignment]

    # Raw detail data (claim-level facts)
    factual_details: list[dict] = None  # type: ignore[assignment]

    # LLM critique text (if available)
    critique_text: str = ""

    def __post_init__(self):
        if self.all_issues is None:
            self.all_issues = []
        if self.factual_details is None:
            self.factual_details = []

    def grade_label(self) -> str:
        s = self.confidence_score
        if s >= 90:
            return "A — Excellent"
        if s >= 78:
            return "B — Good"
        if s >= 65:
            return "C — Acceptable"
        if s >= 50:
            return "D — Needs Improvement"
        return "F — Poor / Unsafe"

    def color(self) -> str:
        """Streamlit-friendly color string for the confidence score."""
        s = self.confidence_score
        if s >= 78:
            return "green"
        if s >= 55:
            return "orange"
        return "red"

    def to_json(self, indent: int = 2) -> str:
        d: dict[str, Any] = {
            "scores": {
                "accuracy": self.accuracy_score,
                "consistency": self.consistency_score,
                "safety": self.safety_score,
                "bias": self.bias_score,
                "clarity": self.clarity_score,
                "completeness": self.completeness_score,
                "confidence": self.confidence_score,
            },
            "grade": self.grade,
            "summaries": {
                "factual": self.factual_summary,
                "consistency": self.consistency_summary,
                "bias_safety": self.bias_safety_summary,
                "clarity": self.clarity_summary,
            },
            "issues": self.all_issues,
            "factual_details": self.factual_details,
            "critique": self.critique_text,
        }
        return json.dumps(d, indent=indent, ensure_ascii=False)


# Weights must sum to 1.0
_WEIGHTS = {
    "accuracy": 0.30,
    "consistency": 0.20,
    "safety": 0.20,
    "bias": 0.10,
    "clarity": 0.10,
    "completeness": 0.10,
}


def aggregate(
    factual: FactualCheckResult,
    consistency: ConsistencyResult,
    bias: BiasResult,
    clarity: ClarityResult,
) -> EvaluationReport:
    """
    Combine all sub-evaluations into a single EvaluationReport.
    """
    report = EvaluationReport(
        accuracy_score=factual.score,
        consistency_score=consistency.score,
        safety_score=bias.safety_score,
        bias_score=bias.bias_score,
        clarity_score=clarity.clarity_score,
        completeness_score=clarity.completeness_score,
        factual_summary=factual.summary(),
        consistency_summary=consistency.summary(),
        bias_safety_summary=bias.summary(),
        clarity_summary=clarity.summary(),
        factual_details=factual.details,
        critique_text=consistency.critique_text,
    )

    # Weighted confidence score
    report.confidence_score = round(
        report.accuracy_score * _WEIGHTS["accuracy"]
        + report.consistency_score * _WEIGHTS["consistency"]
        + report.safety_score * _WEIGHTS["safety"]
        + report.bias_score * _WEIGHTS["bias"]
        + report.clarity_score * _WEIGHTS["clarity"]
        + report.completeness_score * _WEIGHTS["completeness"],
        1,
    )

    report.grade = report.grade_label()

    # Collect all issues
    issues: list[str] = []
    issues.extend(f"[Consistency] {i}" for i in consistency.issues)
    issues.extend(f"[Bias] {i}" for i in bias.bias_flags)
    issues.extend(f"[Safety] {i}" for i in bias.safety_flags)
    issues.extend(f"[Clarity] {i}" for i in clarity.issues)
    report.all_issues = issues

    return report
