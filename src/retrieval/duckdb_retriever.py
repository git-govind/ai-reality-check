"""
DuckDB Fact Retriever
Checks claims against a local structured facts database before falling
back to Wikipedia.  The DB is created and seeded automatically on first
use at data/facts.db.

------------------------------------------------------------
DUCKDB FACTUAL MATCHING — TOKEN OVERLAP THRESHOLD
------------------------------------------------------------
A similarity threshold of 0.40 (40%) determines whether a DuckDB
fact row is a valid match for a claim.

How overlap is computed:
  1. SQL phase  — _extract_keywords() pulls stopword-filtered,
                  length-filtered tokens from the claim and runs
                  LIKE queries to retrieve candidate fact rows.
  2. Score phase — _word_overlap() computes:

       overlap_ratio = |tokens_claim ∩ tokens_fact| / |tokens_claim|

     where tokens are ALL words (re.findall(r"\\w+", text.lower())),
     including stopwords like "is", "the", "of".
     The fact side concatenates: entity + attribute + value.

Threshold behaviour:
  - overlap_ratio >= 0.40  →  valid match ("supported" or "contradicted")
  - overlap_ratio <  0.40  →  rejected; fallback to Wikipedia

Why 0.40 and why ALL tokens (not stopword-filtered)?
  Keeping stopwords in the denominator dilutes overlap for weak
  single-keyword matches. For example:

      Claim : "The moon is made of cheese"   (6 tokens)
      Best  : "neil armstrong ... walk the Moon in 1969"
      ∩     : {the, moon}  →  2/6 = 0.33  →  rejected  ✓

      Claim : "Paris is the capital of France"  (6 tokens)
      Fact  : "france capital paris"
      ∩     : {paris, capital, france}  →  3/6 = 0.50  →  accepted  ✓

  Using only meaningful tokens would shrink the denominator and let
  weak matches pass; ALL tokens keep the bar honest.

If NO DuckDB rows pass the threshold:
  → Fallback to existing Wikipedia-based verification.

This ensures:
  - High precision for supported / contradicted verdicts.
  - Graceful fallback when the DB lacks relevant facts.
  - Transparent, explainable factual scoring.
------------------------------------------------------------
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from config_loader import get_threshold
from utils.text_utils import word_overlap

# ── DB path ────────────────────────────────────────────────────────────────────
_DB_PATH = Path(os.getenv("DUCKDB_PATH", "data/facts.db"))

# ── Thresholds ─────────────────────────────────────────────────────────────────
_SUPPORT_THRESHOLD     = get_threshold("text.duckdb.support_threshold")    # word-overlap ratio to call a claim "supported"
_CONTRADICT_THRESHOLD  = get_threshold("text.duckdb.contradict_threshold") # minimum overlap to attempt contradiction check


# ── Seed data ──────────────────────────────────────────────────────────────────
# Format: (entity, attribute, value)
# Add more rows here to grow the knowledge base over time.
_SEED_FACTS: list[tuple[str, str, str]] = [
    # Science & physics
    ("light",              "speed in vacuum",     "299,792,458 metres per second"),
    ("speed of light",     "value",               "approximately 300,000 km/s"),
    ("water",              "chemical formula",    "H2O"),
    ("water",              "boiling point",       "100 degrees Celsius at sea level"),
    ("water",              "freezing point",      "0 degrees Celsius at standard pressure"),
    ("earth",              "age",                 "approximately 4.5 billion years"),
    ("earth",              "shape",               "oblate spheroid"),
    ("earth",              "distance from sun",   "approximately 150 million kilometres"),
    ("human body",         "normal temperature",  "approximately 37 degrees Celsius or 98.6 Fahrenheit"),
    ("dna",                "structure",           "double helix discovered by Watson and Crick in 1953"),
    ("dna",                "full form",           "deoxyribonucleic acid"),
    ("photosynthesis",     "process",             "converts sunlight carbon dioxide and water into glucose and oxygen"),
    ("photosynthesis",     "location",            "occurs in chloroplasts of plant cells"),
    ("gravity",            "discoverer",          "Isaac Newton formulated laws of gravity in 1687"),
    ("atom",               "components",          "protons neutrons and electrons"),
    ("oxygen",             "symbol",              "O"),
    ("carbon dioxide",     "formula",             "CO2"),

    # History
    ("world war 1",        "start year",          "1914"),
    ("world war 1",        "end year",            "1918"),
    ("world war 2",        "start year",          "1939"),
    ("world war 2",        "end year",            "1945"),
    ("world war ii",       "start year",          "1939"),
    ("world war ii",       "end year",            "1945"),
    ("world war ii",       "causes",              "German invasion of Poland rise of fascism and Nazi ideology"),
    ("moon landing",       "first date",          "July 20 1969"),
    ("apollo 11",          "mission",             "first crewed lunar landing on July 20 1969"),
    ("neil armstrong",     "achievement",         "first human to walk on the Moon in 1969"),
    ("french revolution",  "period",              "1789 to 1799"),
    ("berlin wall",        "fall year",           "1989"),

    # Geography
    ("france",             "capital",             "Paris"),
    ("germany",            "capital",             "Berlin"),
    ("japan",              "capital",             "Tokyo"),
    ("united states",      "capital",             "Washington D.C."),
    ("united kingdom",     "capital",             "London"),
    ("china",              "capital",             "Beijing"),
    ("india",              "capital",             "New Delhi"),
    ("australia",          "capital",             "Canberra"),
    ("brazil",             "capital",             "Brasília"),
    ("russia",             "capital",             "Moscow"),
    ("mount everest",      "height",              "8,848.86 metres above sea level"),
    ("mount everest",      "location",            "Nepal and Tibet border in the Himalayas"),
    ("amazon river",       "location",            "South America primarily Brazil"),
    ("nile river",         "location",            "Africa flows through Egypt"),

    # Technology & computing
    ("python",             "creator",             "Guido van Rossum"),
    ("python",             "first release",       "1991"),
    ("internet",           "origin",              "developed from ARPANET in the 1960s and 1970s"),
    ("world wide web",     "inventor",            "Tim Berners-Lee in 1989"),
    ("artificial intelligence", "coined by",      "John McCarthy in 1956"),
    ("deep learning",      "based on",            "artificial neural networks inspired by the human brain"),

    # Biology & medicine
    ("einstein",           "birth year",          "1879"),
    ("einstein",           "nationality",         "German-born theoretical physicist"),
    ("charles darwin",     "theory",              "theory of evolution by natural selection published in 1859"),
    ("penicillin",         "discoverer",          "Alexander Fleming in 1928"),
    ("vaccination",        "pioneer",             "Edward Jenner developed first vaccine against smallpox in 1796"),

    # Literature & culture
    ("shakespeare",        "profession",          "English playwright and poet born in 1564"),
    ("hamlet",             "author",              "William Shakespeare"),
    ("romeo and juliet",   "author",              "William Shakespeare"),
]


# ── DB initialisation ──────────────────────────────────────────────────────────

def _ensure_db() -> None:
    """Create the database and seed it if it doesn't exist yet."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(_DB_PATH))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                entity    TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value     TEXT NOT NULL
            )
        """)
        row_count = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if row_count == 0:
            con.executemany(
                "INSERT INTO facts (entity, attribute, value) VALUES (?, ?, ?)",
                _SEED_FACTS,
            )
    finally:
        con.close()


def _get_connection() -> duckdb.DuckDBPyConnection:
    _ensure_db()
    return duckdb.connect(str(_DB_PATH))


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class DuckDBCheckResult:
    match_found:    bool              = False
    match_quality:  float             = 0.0        # 0–1
    verdict:        str               = "not_found" # "supported" | "contradicted" | "not_found"
    evidence:       list[dict[str, Any]] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> list[str]:
    """
    Extract candidate entity keywords from a claim.
    Returns lower-cased tokens that are either:
      - 4+ characters
      - Capitalised (proper nouns) when ≥3 chars
      - Numbers (years, quantities)
    """
    words = re.findall(r"[A-Za-z]{3,}|\d{4}", text)
    stopwords = {
        "the", "and", "that", "this", "with", "from", "have", "been",
        "which", "their", "there", "about", "also", "into", "more",
        "its", "are", "was", "were", "has", "had", "for", "not",
    }
    keywords = []
    for w in words:
        lw = w.lower()
        if lw in stopwords:
            continue
        if len(lw) >= 4 or w[0].isupper():
            keywords.append(lw)
    return list(dict.fromkeys(keywords))  # preserve order, dedupe


def _rows_to_dicts(rows: list[tuple]) -> list[dict[str, Any]]:
    return [{"entity": r[0], "attribute": r[1], "value": r[2]} for r in rows]


def _detect_contradiction(claim: str, rows: list[dict]) -> bool:
    """
    Simple heuristic: look for numeric/date values in facts that differ from
    numbers/years mentioned in the claim.
    """
    claim_nums = set(re.findall(r"\b\d{4}\b|\b\d+\.?\d*\b", claim))
    for row in rows:
        fact_nums = set(re.findall(r"\b\d{4}\b|\b\d+\.?\d*\b", row["value"]))
        if claim_nums and fact_nums and claim_nums.isdisjoint(fact_nums):
            return True
    return False


# ── Main verification function ─────────────────────────────────────────────────

def verify_claim_against_duckdb(claim: str) -> DuckDBCheckResult:
    """
    Attempt to verify a single claim against the local facts DB.

    Strategy:
      1. Extract keywords from the claim.
      2. Query facts WHERE entity or value LIKE any keyword.
      3. Score overlap between the claim and each matching row.
      4. If best overlap >= SUPPORT_THRESHOLD → "supported".
      5. If rows found but overlap low and numeric contradiction detected → "contradicted".
      6. Otherwise → "not_found" (caller falls back to Wikipedia).
    """
    if not claim.strip():
        return DuckDBCheckResult()

    keywords = _extract_keywords(claim)
    if not keywords:
        return DuckDBCheckResult()

    try:
        con = _get_connection()
        try:
            # Build a LIKE filter for each keyword against entity + value columns
            conditions = " OR ".join(
                f"(LOWER(entity) LIKE '%' || ? || '%' OR LOWER(value) LIKE '%' || ? || '%')"
                for _ in keywords
            )
            params = [kw for kw in keywords for _ in range(2)]
            rows = con.execute(
                f"SELECT entity, attribute, value FROM facts WHERE {conditions} LIMIT 20",
                params,
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return DuckDBCheckResult()

        row_dicts = _rows_to_dicts(rows)

        # Score each row against the claim, pick the best
        scored = [
            (row, word_overlap(claim, f"{row['entity']} {row['attribute']} {row['value']}"))
            for row in row_dicts
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_row, best_score = scored[0]

        # Only keep rows with meaningful overlap as evidence
        evidence = [r for r, s in scored if s >= _CONTRADICT_THRESHOLD]

        if best_score >= _SUPPORT_THRESHOLD:
            return DuckDBCheckResult(
                match_found=True,
                match_quality=round(best_score, 3),
                verdict="supported",
                evidence=evidence[:5],
            )

        # Some rows matched but overlap is weak — check for numeric contradiction
        if evidence and _detect_contradiction(claim, evidence):
            return DuckDBCheckResult(
                match_found=True,
                match_quality=round(best_score, 3),
                verdict="contradicted",
                evidence=evidence[:5],
            )

        # Rows existed but weren't relevant enough — signal not_found so
        # the caller can fall back to Wikipedia
        return DuckDBCheckResult(
            match_found=False,
            match_quality=round(best_score, 3),
            verdict="not_found",
            evidence=[],
        )

    except Exception:
        # Never block the pipeline — treat DB errors as not_found
        return DuckDBCheckResult()
