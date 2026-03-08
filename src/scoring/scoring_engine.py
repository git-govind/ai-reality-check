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

from config_loader import get_feature, get_threshold, get_weight
from evaluation_report_base import EvaluationReportBase
from explanation_generator import generate_explanation

# ---------------------------------------------------------------------------
# Config-driven constants (loaded once at import time)
# ---------------------------------------------------------------------------
_GRADE_A      = get_threshold("text.grade.a")
_GRADE_B      = get_threshold("text.grade.b")
_GRADE_C      = get_threshold("text.grade.c")
_GRADE_D      = get_threshold("text.grade.d")
_COLOR_GREEN  = get_threshold("text.score_color.green")
_COLOR_ORANGE = get_threshold("text.score_color.orange")


@dataclass
class EvaluationReport(EvaluationReportBase):
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

    # Raw detail data (claim-level facts)
    factual_details: list[dict] = None  # type: ignore[assignment]

    # LLM critique text (if available)
    critique_text: str = ""

    def __post_init__(self):
        if self.factual_details is None:
            self.factual_details = []

    # Backward-compat alias: code that reads/writes report.all_issues
    # transparently maps to the base-class `issues` list.
    @property
    def all_issues(self) -> list[str]:
        return self.issues

    @all_issues.setter
    def all_issues(self, value: list[str]) -> None:
        self.issues = value

    def grade_label(self) -> str:
        s = self.confidence_score
        if s >= _GRADE_A:
            return "A — Excellent"
        if s >= _GRADE_B:
            return "B — Good"
        if s >= _GRADE_C:
            return "C — Acceptable"
        if s >= _GRADE_D:
            return "D — Needs Improvement"
        return "F — Poor / Unsafe"

    def color(self) -> str:
        """Streamlit-friendly color string for the confidence score."""
        s = self.confidence_score
        if s >= _COLOR_GREEN:
            return "green"
        if s >= _COLOR_ORANGE:
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
    "accuracy":     get_weight("text.scoring.accuracy"),
    "consistency":  get_weight("text.scoring.consistency"),
    "safety":       get_weight("text.scoring.safety"),
    "bias":         get_weight("text.scoring.bias"),
    "clarity":      get_weight("text.scoring.clarity"),
    "completeness": get_weight("text.scoring.completeness"),
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

    report.explanation = generate_explanation(report)

    if get_feature("debug"):
        report.metadata["debug"] = {
            "intermediate_scores": {
                "accuracy":              factual.score,
                "consistency":           consistency.score,
                "heuristic_consistency": consistency.heuristic_score,
                "llm_consistency":       consistency.llm_score,    # None if LLM skipped
                "safety":                bias.safety_score,
                "bias":                  bias.bias_score,
                "clarity":               clarity.clarity_score,
                "completeness":          clarity.completeness_score,
            },
            "weighted_contributions": {
                k: round(v * _WEIGHTS[k_map], 1)
                for k, v, k_map in [
                    ("accuracy",     factual.score,              "accuracy"),
                    ("consistency",  consistency.score,          "consistency"),
                    ("safety",       bias.safety_score,          "safety"),
                    ("bias",         bias.bias_score,            "bias"),
                    ("clarity",      clarity.clarity_score,      "clarity"),
                    ("completeness", clarity.completeness_score, "completeness"),
                ]
            },
            "raw_evidence": {
                "claims_checked":      factual.claims_checked,
                "claims_supported":    factual.supported,
                "claims_contradicted": factual.contradicted,
                "claims_unverified":   factual.unverified,
                "factual_details":     factual.details,
                "consistency_issues":  consistency.issues,
                "critique_text":       consistency.critique_text,
                "bias_flags":          bias.bias_flags,
                "safety_flags":        bias.safety_flags,
                "word_count":          clarity.word_count,
                "sentence_count":      clarity.sentence_count,
                "avg_sentence_len":    clarity.avg_sentence_len,
                "has_structure":       clarity.has_structure,
            },
            "thresholds_used": {
                "grades":       {"A": _GRADE_A, "B": _GRADE_B, "C": _GRADE_C, "D": _GRADE_D},
                "weights":      dict(_WEIGHTS),
                "score_colors": {"green": _COLOR_GREEN, "orange": _COLOR_ORANGE},
            },
            "module_decisions": {
                "llm_consistency_ran": consistency.llm_score is not None,
                "llm_bias_enabled":    get_feature("text.llm_bias_check"),
                "consistency_blend":   (
                    "heuristic+llm"
                    if consistency.llm_score is not None
                    else "heuristic-only"
                ),
            },
        }

    return report
