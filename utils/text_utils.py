"""
utils/text_utils.py
--------------------
Shared text-processing helpers used across the text evaluation pipeline.

Public API
----------
  STOPWORDS                  – combined frozenset of common stopwords
  word_overlap(a, b)         – Jaccard-style |A∩B| / max(|A|, 1)
  split_sentences(text)      – split on .!? boundaries, strip empties
  parse_llm_score(text)      – extract a 0–100 score from "N/10" LLM output
  extract_issue_bullets(...) – pull non-trivial problem bullets from critique
"""
from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Shared stopword set
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    # Articles / conjunctions / prepositions
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "by", "as", "is", "it", "its", "be", "been", "from", "with",
    # Common verbs / pronouns
    "are", "was", "were", "has", "had", "have", "does", "not", "who",
    "which", "that", "this", "they", "their", "there", "just", "very",
    # Common filler words
    "about", "also", "also", "into", "more", "quite", "how", "when",
    "what", "where", "why",
})


# ---------------------------------------------------------------------------
# Word tokenisation / overlap
# ---------------------------------------------------------------------------

def word_overlap(text_a: str, text_b: str) -> float:
    """
    Jaccard-style word overlap: |A ∩ B| / max(|A|, 1).

    Both strings are lower-cased and split on word boundaries.
    Returns 0.0 when *text_a* is empty.
    """
    a = set(re.findall(r"\w+", text_a.lower()))
    b = set(re.findall(r"\w+", text_b.lower()))
    return len(a & b) / max(len(a), 1)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Split *text* into non-empty sentences on ``.`` ``!`` ``?`` boundaries.

    Returns a list of stripped sentence strings (empties are dropped).
    """
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


# ---------------------------------------------------------------------------
# LLM score extraction
# ---------------------------------------------------------------------------

def parse_llm_score(text: str) -> Optional[float]:
    """
    Extract a 0–100 score from LLM output containing an N/10 rating.

    Patterns recognised (case-insensitive):
      ``N/10``           e.g. "8/10"
      ``score: N``       e.g. "Score: 7.5"
      ``N out of 10``    e.g. "7 out of 10"

    Returns the value scaled to 0–100, or ``None`` if no pattern is found.
    """
    patterns = [
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"(?:score|rating)[:\s]+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+out\s+of\s+10",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return min(float(m.group(1)) * 10.0, 100.0)
    return None


# ---------------------------------------------------------------------------
# Bullet-issue extraction from LLM critique text
# ---------------------------------------------------------------------------

# Shared compiled patterns (module-level for performance)
_BULLET_RE: re.Pattern = re.compile(
    r"^\s*[-•*]\s+|^\s*\d+[.)]\s+"
)
_NONE_ANSWER_RE: re.Pattern = re.compile(
    r":\s*(None|N/A|No\s+\w[\w\s]{0,30}|nothing|not\s+(any|applicable))"
    r"[.;,]?\s*$",
    re.I,
)
_BASE_NEG_RE: re.Pattern = re.compile(
    r"\b(no|not|none|never|without|doesn'?t|don'?t|didn'?t|cannot|"
    r"absent|absence|free\s+from|does\s+not)\b",
    re.I,
)


def extract_issue_bullets(
    critique: str,
    problem_kw_pattern: str,
    min_len: int = 15,
    neg_re: Optional[re.Pattern] = None,
    skip_line_re: Optional[re.Pattern] = None,
) -> List[str]:
    """
    Extract non-trivial problem bullet points from LLM critique text.

    Parameters
    ----------
    critique : str
        The full critique string from the LLM.
    problem_kw_pattern : str
        Regex pattern (``re.I`` is applied) that must match for a line to
        be considered a real problem.
    min_len : int
        Minimum character length of the cleaned line to accept.
    neg_re : re.Pattern or None
        Negation-guard pattern.  Lines where the problem keyword is
        preceded by a negation word are discarded.
        Defaults to :data:`_BASE_NEG_RE`.
    skip_line_re : re.Pattern or None
        If provided, lines matching this pattern are dropped entirely
        before any other check (useful for skipping "Bias Score: …" lines).

    Returns
    -------
    list[str]
        Cleaned bullet texts in order, duplicates included (let callers
        deduplicate with ``dict.fromkeys``).
    """
    neg        = neg_re or _BASE_NEG_RE
    problem_kw = re.compile(problem_kw_pattern, re.I)
    issues: List[str] = []

    for line in str(critique).splitlines():
        stripped = line.strip()
        if skip_line_re and skip_line_re.search(stripped):
            continue
        if not _BULLET_RE.match(line):
            continue
        clean = stripped.lstrip("-•*0123456789.) ")
        if len(clean) < min_len:
            continue
        if not problem_kw.search(clean):
            continue
        if _NONE_ANSWER_RE.search(clean):
            continue
        kw_match  = problem_kw.search(clean)
        preceding = clean[: kw_match.start()]
        if neg.search(preceding):
            continue
        issues.append(clean)

    return issues
