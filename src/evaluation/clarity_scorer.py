"""
Completeness & Clarity Scorer
Evaluates the structure, depth, and readability of an AI response.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from config_loader import get_threshold, get_weight
from utils.text_utils import split_sentences

# ---------------------------------------------------------------------------
# Config-driven constants (loaded once at import time)
# ---------------------------------------------------------------------------
_DEFAULT_COMPLETENESS  = get_threshold("text.clarity.default_completeness")
_COMPLETENESS_SCALE    = get_threshold("text.clarity.completeness_scale")
_LOW_OVERLAP_THRESHOLD = get_threshold("text.clarity.low_overlap_threshold")
_SHORT_RESPONSE_WORDS  = int(get_threshold("text.clarity.short_response_words"))
_SHORT_RESPONSE_CAP    = get_threshold("text.clarity.short_response_cap")
_LONG_RESPONSE_WORDS   = int(get_threshold("text.clarity.long_response_words"))
_LONG_SENTENCE_WORDS   = get_threshold("text.clarity.long_sentence_words")
_LONG_SENTENCE_PENALTY = get_threshold("text.clarity.long_sentence_penalty")
_VAGUE_PHRASE_COUNT    = int(get_threshold("text.clarity.vague_phrase_count"))
_VAGUE_PENALTY         = get_threshold("text.clarity.vague_phrase_penalty")
_VAGUE_MAX_PENALTY     = get_threshold("text.clarity.vague_phrase_max_penalty")
_FILLER_PENALTY        = get_threshold("text.clarity.filler_penalty")
_FILLER_MAX_PENALTY    = get_threshold("text.clarity.filler_max_penalty")
_STRUCTURAL_BONUS      = get_threshold("text.clarity.structural_bonus")
_W_COMPLETENESS        = get_weight("text.clarity.completeness_weight")
_W_CLARITY             = get_weight("text.clarity.clarity_weight")


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
        return _DEFAULT_COMPLETENESS, issues

    overlap = len(prompt_keywords & response_words) / len(prompt_keywords)
    score = min(100.0, overlap * _COMPLETENESS_SCALE)  # generous scaling

    if overlap < _LOW_OVERLAP_THRESHOLD:
        issues.append("Response may not address the key topics in the prompt.")
    if len(response.split()) < _SHORT_RESPONSE_WORDS:
        issues.append("Response is very short — may be incomplete.")
        score = min(score, _SHORT_RESPONSE_CAP)
    if len(response.split()) > _LONG_RESPONSE_WORDS:
        issues.append("Response is very long — may contain padding or off-topic content.")

    return round(score, 1), issues


def _clarity(text: str) -> tuple[float, list[str]]:
    """Score readability and flag vague / filler language."""
    issues = []
    sentences = split_sentences(text)
    words = text.split()
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    avg_len = word_count / sent_count

    score = 100.0

    # Penalize very long sentences (hard to read)
    if avg_len > _LONG_SENTENCE_WORDS:
        issues.append(f"Average sentence length is {avg_len:.0f} words — consider shorter sentences.")
        score -= _LONG_SENTENCE_PENALTY

    # Vague phrases
    vague = [p for p in _VAGUE_PHRASES if re.search(p, text, re.I)]
    if len(vague) > _VAGUE_PHRASE_COUNT:
        issues.append(f"{len(vague)} vague qualifiers found (e.g. 'it depends', 'maybe').")
        score -= min(_VAGUE_MAX_PENALTY, len(vague) * _VAGUE_PENALTY)

    # Filler patterns (AI speak)
    fillers = [p for p in _FILLER_PATTERNS if re.search(p, text, re.I)]
    if fillers:
        issues.append(f"{len(fillers)} filler phrase(s) detected (e.g. 'As an AI', 'Great question').")
        score -= min(_FILLER_MAX_PENALTY, len(fillers) * _FILLER_PENALTY)

    # Structural bonus
    if _count_structure(text):
        score = min(100.0, score + _STRUCTURAL_BONUS)

    return max(0.0, round(score, 1)), issues


def run(prompt: str, ai_response: str) -> ClarityResult:
    """Evaluate completeness and clarity of the AI's response."""
    result = ClarityResult()

    sentences = split_sentences(ai_response)
    result.word_count = len(ai_response.split())
    result.sentence_count = len(sentences)
    result.avg_sentence_len = result.word_count / max(result.sentence_count, 1)
    result.has_structure = _count_structure(ai_response)

    c_score, c_issues = _completeness(prompt, ai_response)
    cl_score, cl_issues = _clarity(ai_response)

    result.completeness_score = c_score
    result.clarity_score = cl_score
    result.issues = c_issues + cl_issues

    result.score = round(_W_COMPLETENESS * c_score + _W_CLARITY * cl_score, 1)
    return result
