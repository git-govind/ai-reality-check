"""
Factual Checker
Cross-references claims in an AI response with Wikipedia.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.retrieval.wikipedia_retriever import extract_key_claims, verify_claim_against_wiki


@dataclass
class FactualCheckResult:
    claims_checked: int = 0
    supported: int = 0
    unverified: int = 0
    no_source: int = 0
    details: List[dict] = field(default_factory=list)
    score: float = 0.0  # 0–100

    def summary(self) -> str:
        if self.claims_checked == 0:
            return "No verifiable factual claims detected."
        return (
            f"{self.supported}/{self.claims_checked} claims supported by Wikipedia. "
            f"{self.unverified} unverified, {self.no_source} with no source."
        )


def run(ai_response: str) -> FactualCheckResult:
    """
    Extract claims from the AI response and verify each against Wikipedia.

    Returns a FactualCheckResult with per-claim verdicts and an overall score.
    """
    claims = extract_key_claims(ai_response)
    result = FactualCheckResult(claims_checked=len(claims))

    for claim in claims:
        verdict_data = verify_claim_against_wiki(claim)
        result.details.append(verdict_data)
        v = verdict_data["verdict"]
        if v == "supported":
            result.supported += 1
        elif v == "unverified":
            result.unverified += 1
        else:
            result.no_source += 1

    # Score: supported gets full credit, unverified half, no_source zero
    if claims:
        raw = (result.supported * 1.0 + result.unverified * 0.5) / len(claims)
        result.score = round(raw * 100, 1)
    else:
        # No claims → neutral score
        result.score = 70.0

    return result
