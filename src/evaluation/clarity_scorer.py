"""
Completeness & Clarity Scorer
Evaluates the structure, depth, and readability of an AI response.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ClarityResult:
    issues: List[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_len: float = 0.0
    has_structure: bool = False     # uses lists/headers
    completeness_score: float = 0.0  # 0–100: does it answer the question?
    clarity_score: float = 0.0       # 0–100: readability & structure
    score: float = 0.0               # combined

    def summary(self) -> str:
        parts = [
            f"{self.word_count} words",
            f"{self.sentence_count} sentences",
            f"avg {self.avg_sentence_len:.0f} words/sentence",
        ]
        if self.has_structure:
            parts.append("structured response")
        return " | ".join(parts)


_VAGUE_PHRASES = [
    r"\bit\s+depends\b",
    r"\bvaries?\b",
    r"\bsomewhat\b",
    r"\bkind\s+of\b",
    r"\bsort\s+of\b",
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
]

_FILLER_PATTERNS = [
    r"\bAs an AI(?: language model)?,?\b",
    r"\bI (don't|do not) have (the ability|access|opinions)\b",
    r"\bCertainly[,!]\b",
    r"\bOf course[,!]\b",
    r"\bAbsolutely[,!]\b",
    r"\bGreat question[,!]\b",
    r"\bI'm happy to help\b",
]


def _count_structure(text: str) -> bool:
    """True if the response uses markdown headers, lists, or numbered items."""
    structure_pat = re.compile(r"(^#{1,3}\s|^\s*[-*•]\s|^\s*\d+[.)]\s)", re.MULTILINE)
    return bool(structure_pat.search(text))


def _completeness(prompt: str, response: str) -> tuple[float, list[str]]:
    """
    Heuristically estimate how well the response addresses the prompt.
    Checks keyword overlap and length appropriateness.
    """
    issues = []

    prompt_keywords = set(re.findall(r"\b[a-z]{4,}\b", prompt.lower())) - {
        "what", "when", "where", "which", "how", "why", "does", "have", "this", "that", "with", "from", "about"
    }
    response_words = set(re.findall(r"\b[a-z]{4,}\b", response.lower()))

    if not prompt_keywords:
        return 70.0, issues

    overlap = len(prompt_keywords & response_words) / len(prompt_keywords)
    score = min(100.0, overlap * 130)  # generous scaling

    if overlap < 0.3:
        issues.append("Response may not address the key topics in the prompt.")
    if len(response.split()) < 30:
        issues.append("Response is very short — may be incomplete.")
        score = min(score, 55.0)
    if len(response.split()) > 600:
        issues.append("Response is very long — may contain padding or off-topic content.")

    return round(score, 1), issues


def _clarity(text: str) -> tuple[float, list[str]]:
    """Score readability and flag vague / filler language."""
    issues = []
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.split()
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    avg_len = word_count / sent_count

    score = 100.0

    # Penalize very long sentences (hard to read)
    if avg_len > 35:
        issues.append(f"Average sentence length is {avg_len:.0f} words — consider shorter sentences.")
        score -= 10

    # Vague phrases
    vague = [p for p in _VAGUE_PHRASES if re.search(p, text, re.I)]
    if len(vague) > 2:
        issues.append(f"{len(vague)} vague qualifiers found (e.g. 'it depends', 'maybe').")
        score -= min(20, len(vague) * 4)

    # Filler patterns (AI speak)
    fillers = [p for p in _FILLER_PATTERNS if re.search(p, text, re.I)]
    if fillers:
        issues.append(f"{len(fillers)} filler phrase(s) detected (e.g. 'As an AI', 'Great question').")
        score -= min(15, len(fillers) * 5)

    # Structural bonus
    if _count_structure(text):
        score = min(100.0, score + 5)

    return max(0.0, round(score, 1)), issues


def run(prompt: str, ai_response: str) -> ClarityResult:
    """Evaluate completeness and clarity of the AI's response."""
    result = ClarityResult()

    sentences = [s.strip() for s in re.split(r"[.!?]+", ai_response) if s.strip()]
    result.word_count = len(ai_response.split())
    result.sentence_count = len(sentences)
    result.avg_sentence_len = result.word_count / max(result.sentence_count, 1)
    result.has_structure = _count_structure(ai_response)

    c_score, c_issues = _completeness(prompt, ai_response)
    cl_score, cl_issues = _clarity(ai_response)

    result.completeness_score = c_score
    result.clarity_score = cl_score
    result.issues = c_issues + cl_issues

    # Final: 60% completeness, 40% clarity
    result.score = round(0.6 * c_score + 0.4 * cl_score, 1)
    return result
