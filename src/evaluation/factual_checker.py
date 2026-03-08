"""
Factual Checker
Cross-references claims in an AI response against:
  1. Local DuckDB structured facts (fast, offline)  ← checked first
  2. Wikipedia API (fallback when DB has no match)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.retrieval.duckdb_retriever import DuckDBCheckResult, verify_claim_against_duckdb
from src.retrieval.wikipedia_retriever import extract_key_claims, verify_claim_against_wiki

from config_loader import get_threshold, get_weight

# ---------------------------------------------------------------------------
# Config-driven constants (loaded once at import time)
# ---------------------------------------------------------------------------
_DEFAULT_SCORE           = get_threshold("text.factual.default_score")
_CONTRADICTION_PENALTY   = get_threshold("text.factual.contradiction_penalty")
_W_DUCKDB_SUPPORTED      = get_weight("text.factual.duckdb_supported")
_W_WIKIPEDIA_SUPPORTED   = get_weight("text.factual.wikipedia_supported")
_W_UNVERIFIED            = get_weight("text.factual.unverified")


@dataclass
class FactualCheckResult:
    claims_checked: int = 0
    supported:      int = 0
    contradicted:   int = 0   # new: DB found a conflicting fact
    unverified:     int = 0
    no_source:      int = 0

    # Per-claim detail dicts — schema:
    # {
    #   "claim":           str,
    #   "verdict":         "supported" | "contradicted" | "unverified" | "no_source",
    #   "source":          "duckdb" | "wikipedia" | "none",
    #   "match_quality":   float,          (DuckDB only, else None)
    #   "duckdb_evidence": list[dict],     (DuckDB rows that matched)
    #   "wiki_title":      str | None,
    #   "wiki_summary":    str | None,
    #   "wiki_url":        str | None,
    # }
    details: List[dict] = field(default_factory=list)
    score:   float       = 0.0   # 0–100

    def summary(self) -> str:
        if self.claims_checked == 0:
            return "No verifiable factual claims detected."

        db_hits = sum(
            1 for d in self.details if d.get("source") == "duckdb"
        )
        wiki_hits = sum(
            1 for d in self.details if d.get("source") == "wikipedia"
        )
        parts = [
            f"{self.supported}/{self.claims_checked} claims supported",
        ]
        if db_hits:
            parts.append(f"{db_hits} via local DB")
        if wiki_hits:
            parts.append(f"{wiki_hits} via Wikipedia")
        if self.contradicted:
            parts.append(f"{self.contradicted} contradicted by local DB")
        if self.unverified:
            parts.append(f"{self.unverified} unverified")
        if self.no_source:
            parts.append(f"{self.no_source} no source found")
        return ". ".join(parts) + "."


# ── Main run function ──────────────────────────────────────────────────────────

def run(ai_response: str) -> FactualCheckResult:
    """
    Extract claims from the AI response and verify each one.

    Verification order per claim:
      1. DuckDB local facts DB  → if match_found, use that verdict.
      2. Wikipedia API fallback → only reached when DuckDB has no match.

    Returns a FactualCheckResult with per-claim verdicts and an overall score.
    """
    claims = extract_key_claims(ai_response)
    result = FactualCheckResult(claims_checked=len(claims))

    for claim in claims:
        detail = _check_claim(claim)
        result.details.append(detail)

        v = detail["verdict"]
        if v == "supported":
            result.supported += 1
        elif v == "contradicted":
            result.contradicted += 1
        elif v == "unverified":
            result.unverified += 1
        else:
            result.no_source += 1

    result.score = _compute_score(result)
    return result


def _check_claim(claim: str) -> dict:
    """
    Verify a single claim.  Returns a detail dict regardless of which
    source produced the verdict.
    """
    # ── Step 1: DuckDB ──────────────────────────────────────────
    db: DuckDBCheckResult = verify_claim_against_duckdb(claim)

    if db.match_found:
        return {
            "claim":           claim,
            "verdict":         db.verdict,          # "supported" | "contradicted"
            "source":          "duckdb",
            "match_quality":   db.match_quality,
            "duckdb_evidence": db.evidence,
            "wiki_title":      None,
            "wiki_summary":    None,
            "wiki_url":        None,
        }

    # ── Step 2: Wikipedia fallback ──────────────────────────────
    wiki = verify_claim_against_wiki(claim)
    return {
        "claim":           claim,
        "verdict":         wiki.get("verdict", "no_source"),
        "source":          "wikipedia" if wiki.get("wiki_title") else "none",
        "match_quality":   None,
        "duckdb_evidence": [],
        "wiki_title":      wiki.get("wiki_title"),
        "wiki_summary":    wiki.get("wiki_summary"),
        "wiki_url":        wiki.get("wiki_url"),
    }


def _compute_score(result: FactualCheckResult) -> float:
    """
    Weighted scoring across all claims:
      - DuckDB supported      → 1.00 (high-confidence structured match)
      - Wikipedia supported   → 0.90 (good but unstructured)
      - Unverified            → 0.50
      - No source             → 0.00
      - Contradicted          → 0.00 (penalty on top of zero credit)

    A contradiction penalty of 15 pts per contradicted claim is added
    on top of zero credit, capped so the score never goes negative.
    """
    if result.claims_checked == 0:
        return _DEFAULT_SCORE

    weighted_sum = 0.0
    for d in result.details:
        verdict = d["verdict"]
        source  = d.get("source", "none")

        if verdict == "supported":
            weighted_sum += _W_DUCKDB_SUPPORTED if source == "duckdb" else _W_WIKIPEDIA_SUPPORTED
        elif verdict == "unverified":
            weighted_sum += _W_UNVERIFIED
        # "contradicted" and "no_source" → 0.0

    raw = weighted_sum / result.claims_checked
    base_score = round(raw * 100, 1)

    # Contradiction penalty
    penalty = result.contradicted * _CONTRADICTION_PENALTY
    return max(0.0, base_score - penalty)
