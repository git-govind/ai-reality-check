"""
Logical Consistency Checker
Uses a second LLM pass to critique the original response for contradictions
and logical gaps. Falls back to heuristic checks when no LLM is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConsistencyResult:
    issues: List[str] = field(default_factory=list)
    critique_text: str = ""
    llm_score: float | None = None   # extracted from LLM if present
    heuristic_score: float = 80.0
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

    # Penalty: 10 pts per detected pattern, max 50 pts off
    score = max(50.0, 100.0 - len(issues) * 10)
    return score, issues


def _parse_llm_score(critique: str) -> float | None:
    """Try to extract a numeric score from LLM critique text."""
    # Look for "N/10" or "score: N" or "rating: N"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"(?:score|rating)[:\s]+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+out\s+of\s+10",
    ]
    for pat in patterns:
        m = re.search(pat, critique, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return min(val * 10, 100.0)  # normalize to 0–100
    return None


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
    if model_display_name:
        try:
            from src.llm.response_generator import generate_critique

            critique = generate_critique(original_prompt, ai_response, model_display_name)
            result.critique_text = critique
            result.llm_score = _parse_llm_score(critique)

            # Only extract lines that look like bullet points AND flag a real problem.
            # Negation guard: skip lines that say "no X" / "does not X" etc.
            _neg = re.compile(
                r"\b(no|not|none|never|without|doesn'?t|don'?t|didn'?t|cannot|"
                r"absent|absence|free\s+from|does\s+not)\b",
                re.I,
            )
            _problem_kw = re.compile(
                r"\b(error|inconsisten|contradict|missing|wrong|incorrect|inaccurate|mislead|unsupport)\w*\b",
                re.I,
            )
            _bullet = re.compile(r"^\s*[-•*]\s+|^\s*\d+[.)]\s+")
            # Matches lines where the answer after ':' is None/N/A/nothing
            _none_answer = re.compile(
                r":\s*(None|N/A|No\s+\w[\w\s]{0,30}|nothing|not\s+(any|applicable))[.;,]?\s*$",
                re.I,
            )

            for line in critique.splitlines():
                stripped = line.strip()
                if not _bullet.match(line):
                    continue
                clean = stripped.lstrip("-•*0123456789.) ")
                if len(clean) < 20:
                    continue
                if not _problem_kw.search(clean):
                    continue
                # Skip "Category: None." style answers
                if _none_answer.search(clean):
                    continue
                # Skip if negation appears before the keyword
                kw_match = _problem_kw.search(clean)
                preceding = clean[: kw_match.start()]
                if _neg.search(preceding):
                    continue
                result.issues.append(clean)
        except Exception as exc:
            result.critique_text = f"LLM critique unavailable: {exc}"

    # --- Final score ---
    if result.llm_score is not None:
        # Blend heuristic (30%) and LLM (70%)
        result.score = round(0.3 * result.heuristic_score + 0.7 * result.llm_score, 1)
    else:
        result.score = round(result.heuristic_score, 1)

    # Deduplicate issues
    result.issues = list(dict.fromkeys(result.issues))
    return result
