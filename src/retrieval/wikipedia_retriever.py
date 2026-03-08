"""
Wikipedia Retriever
Fetches relevant Wikipedia summaries to ground factual checks.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import requests
from dotenv import load_dotenv

from utils.text_utils import word_overlap

load_dotenv()

WIKI_USER_AGENT = os.getenv("WIKI_USER_AGENT", "AIRealityCheck/1.0")
WIKI_API = "https://en.wikipedia.org/w/api.php"


@lru_cache(maxsize=256)
def search_wikipedia(query: str, sentences: int = 5) -> dict[str, str]:
    """
    Search Wikipedia and return (title, summary, url) for the best match.

    Returns:
        {"title": ..., "summary": ..., "url": ...} or empty dict on failure.
    """
    try:
        # Step 1: search for the best matching page title
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        }
        headers = {"User-Agent": WIKI_USER_AGENT}
        resp = requests.get(WIKI_API, params=search_params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return {}

        title = results[0]["title"]

        # Step 2: fetch the extract for that page
        extract_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exsentences": sentences,
            "exintro": True,
            "explaintext": True,
            "format": "json",
        }
        resp2 = requests.get(WIKI_API, params=extract_params, headers=headers, timeout=10)
        resp2.raise_for_status()
        pages = resp2.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        summary = page.get("extract", "")

        return {
            "title": title,
            "summary": summary,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        }
    except Exception:
        return {}


def extract_key_claims(text: str) -> list[str]:
    """
    Heuristically extract factual-sounding sentences from text.
    Real claim extraction would use NLP; this is a lightweight proxy.
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    # Patterns that suggest a factual claim
    factual_patterns = re.compile(
        r"\b(is|are|was|were|has|have|had|consists|contains|"
        r"founded|created|invented|discovered|born|died|won|"
        r"located|known as|refers to|defined as)\b",
        re.IGNORECASE,
    )
    for sent in sentences:
        if len(sent.split()) >= 6 and factual_patterns.search(sent):
            claims.append(sent.strip())
    return claims[:8]  # cap at 8 claims per evaluation


def verify_claim_against_wiki(claim: str) -> dict:
    """
    Attempt to verify a single claim against Wikipedia.

    Returns:
        {
            "claim": str,
            "wiki_title": str,
            "wiki_summary": str,
            "wiki_url": str,
            "verdict": "supported" | "unverified" | "no_source",
        }
    """
    # Build a search query from the claim (first 10 words)
    query = " ".join(claim.split()[:10])
    wiki = search_wikipedia(query)

    if not wiki:
        return {
            "claim": claim,
            "wiki_title": None,
            "wiki_summary": None,
            "wiki_url": None,
            "verdict": "no_source",
        }

    # Simple overlap heuristic — a real system would use embeddings
    overlap = word_overlap(claim, wiki["summary"])

    verdict = "supported" if overlap >= 0.25 else "unverified"

    return {
        "claim": claim,
        "wiki_title": wiki["title"],
        "wiki_summary": wiki["summary"][:300],
        "wiki_url": wiki["url"],
        "verdict": verdict,
    }
