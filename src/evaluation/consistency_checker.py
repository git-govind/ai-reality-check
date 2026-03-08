"""
Logical Consistency Checker
Uses a second LLM pass to critique the original response for contradictions
and logical gaps. Falls back to heuristic checks when no LLM is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from config_loader import get_feature, get_threshold, get_weight
from models.llm_registry import get_model as _get_llm
from utils.text_utils import extract_issue_bullets, parse_llm_score

# ---------------------------------------------------------------------------
# Config-driven constants (loaded once at import time)
# ---------------------------------------------------------------------------
_HEURISTIC_DEFAULT        = get_threshold("text.consistency.heuristic_default")
_CONTRADICTION_PENALTY    = get_threshold("text.consistency.contradiction_penalty")
_SCORE_FLOOR              = get_threshold("text.consistency.score_floor")
_W_HEURISTIC              = get_weight("text.consistency.heuristic_weight")
_W_LLM                    = get_weight("text.consistency.llm_weight")


@dataclass
class ConsistencyResult:
    issues: List[str] = field(default_factory=list)
    critique_text: str = ""
    llm_score: float | None = None   # extracted from LLM if present
    heuristic_score: float = _HEURISTIC_DEFAULT
    score: float = 0.0               # final 0–100

    def summary(self) -> str:
        if not self.issues:
            return "No obvious logical inconsistencies detected."
        return f"{len(self.issues)} potential consistency issue(s) found."


# Contradiction signal phrases
_CONTRADICTION_PATTERNS = [
    r"\bbut\s+(also|then|simultaneously)\b",
    r"\bon\s+the\s+one\s+hand.+on\s+the\s+other\b",
    r"\bhowever.{0,60}(said|stated|mentioned) (earlier|above|before)\b",
    r"\bcontradicts\b",
    r"\bunless.+always\b",
    r"\bnever.+always\b",
]


def _heuristic_check(text: str) -> tuple[float, list[str]]:
    """Light-weight rule-based consistency check."""
    issues = []
    for pat in _CONTRADICTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE | re.DOTALL):
            issues.append(f"Possible contradiction pattern: «{pat}»")

    # Penalty: configured pts per detected pattern, floor at configured minimum
    score = max(_SCORE_FLOOR, 100.0 - len(issues) * _CONTRADICTION_PENALTY)
    return score, issues


def run(
    original_prompt: str,
    ai_response: str,
    model_display_name: str | None = None,
) -> ConsistencyResult:
    """
    Check the AI response for logical consistency.

    If a model is provided, uses LLM-based critique; always runs heuristics.
    """
    result = ConsistencyResult()

    # --- Heuristic pass (always runs) ---
    h_score, h_issues = _heuristic_check(ai_response)
    result.heuristic_score = h_score
    result.issues.extend(h_issues)

    # --- LLM critique pass (optional) ---
    if model_display_name and get_feature("text.llm_consistency_check"):
        try:
            critique = _get_llm(model_display_name).generate_critique(original_prompt, ai_response)
            result.critique_text = critique
            result.llm_score = parse_llm_score(critique)

            # Extract bullet-point issues, guarding against negation and
            # "Category: None" style false-positives.
            result.issues.extend(
                extract_issue_bullets(
                    critique,
                    problem_kw_pattern=(
                        r"\b(error|inconsisten|contradict|missing|wrong|"
                        r"incorrect|inaccurate|mislead|unsupport)\w*\b"
                    ),
                    min_len=20,
                )
            )
        except Exception as exc:
            result.critique_text = f"LLM critique unavailable: {exc}"

    # --- Final score ---
    if result.llm_score is not None:
        result.score = round(_W_HEURISTIC * result.heuristic_score + _W_LLM * result.llm_score, 1)
    else:
        result.score = round(result.heuristic_score, 1)

    # Deduplicate issues
    result.issues = list(dict.fromkeys(result.issues))
    return result
