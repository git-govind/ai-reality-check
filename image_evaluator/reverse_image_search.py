"""
reverse_image_search.py
------------------------
Step 5 of the Image Authenticity Evaluator pipeline — OPTIONAL fallback.

Attempts to locate this image (or near-copies) on the public web.  A
confirmed match against a known-legitimate source is strong evidence that
the image is real; significant differences between the query image and the
found source suggest cropping, colour-grading, or deepfake swapping.

Supported backends (tried in priority order)
--------------------------------------------
  1. Google Cloud Vision API  – requires env-var ``GOOGLE_API_KEY``
  2. Bing Visual Search API   – requires env-var ``BING_SEARCH_KEY``
  3. SerpApi (Google Lens)    – requires env-var ``SERPAPI_KEY``
  4. No backend available     – returns ReverseSearchResult(ran=False)

Perceptual hash similarity
--------------------------
When a result URL is returned by the API, the module optionally downloads
the source image and computes its perceptual hash (Average Hash / pHash)
against the query image to estimate how similar the two images are.

Dependencies: Pillow, NumPy (required for pHash); requests (stdlib/already
              in environment); specific API keys (optional)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .datatypes import ReverseSearchResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _google_api_key()  -> str: return os.getenv("GOOGLE_API_KEY",  "")
def _bing_search_key() -> str: return os.getenv("BING_SEARCH_KEY", "")
def _serpapi_key()     -> str: return os.getenv("SERPAPI_KEY",     "")

_REQUEST_TIMEOUT = 8          # seconds
_MAX_SOURCE_DL   = 5          # max source images to download for pHash check
_PHASH_BLOCK     = 8          # pHash block size (8×8 = 64-bit hash)


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------

def _average_hash(img: Image.Image, size: int = _PHASH_BLOCK) -> int:
    """
    Compute a simple Average Hash (aHash) as an integer.
    Two images with hamming_distance < 10 are considered near-identical.
    """
    gray  = img.convert("L").resize((size, size), Image.LANCZOS)
    arr   = np.array(gray, dtype=np.float32)
    mean  = arr.mean()
    bits  = (arr >= mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(h1: int, h2: int) -> int:
    """Bit-wise hamming distance between two integer hashes."""
    xor   = h1 ^ h2
    count = 0
    while xor:
        count += xor & 1
        xor  >>= 1
    return count


def _phash_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """Return 0.0 (completely different) … 1.0 (identical) perceptual similarity."""
    bits  = _PHASH_BLOCK * _PHASH_BLOCK
    dist  = _hamming_distance(_average_hash(img_a), _average_hash(img_b))
    return max(0.0, 1.0 - dist / bits)


def _download_image(url: str) -> Optional[Image.Image]:
    """Attempt to download *url* and return a PIL Image or None on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ImageAuthenticityBot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = resp.read(5 * 1024 * 1024)  # max 5 MB
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Google Cloud Vision — Web Detection
# ---------------------------------------------------------------------------

def _google_web_detection(image_bytes: bytes) -> tuple[Optional[Dict], Optional[str]]:
    """
    Call the Google Cloud Vision API web-detection endpoint.
    Returns (response_dict, None) on success or (None, error_message) on failure.
    Returns (None, None) when no key is configured.

    Environment variable required: ``GOOGLE_API_KEY``
    """
    if not _google_api_key():
        return None, None

    url     = (
        f"https://vision.googleapis.com/v1/images:annotate"
        f"?key={_google_api_key()}"
    )
    b64     = base64.b64encode(image_bytes).decode("utf-8")
    payload = json.dumps({
        "requests": [{
            "image":    {"content": b64},
            "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
        }]
    }).encode("utf-8")

    try:
        req  = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        try:
            body  = json.loads(e.read())
            msg   = body.get("error", {}).get("message", str(e))
        except Exception:
            msg = f"HTTP {e.code}: {e.reason}"
        return None, f"Google Vision API error: {msg}"
    except Exception as exc:
        return None, f"Google Vision API error: {exc}"


def _parse_google_response(resp: Dict) -> Tuple[List[str], float]:
    """
    Return (source_urls, best_similarity) from a Vision API web-detection response.

    Similarity tier mapping (spec §2):
      fullMatchingImages    → 1.00  (exact match)
      partialMatchingImages → 0.80  (high similarity)
      pagesWithMatchingImages only → 0.50  (partial match)
      nothing               → 0.00
    """
    try:
        wd      = resp["responses"][0]["webDetection"]
        pages   = wd.get("pagesWithMatchingImages",  [])
        urls    = [p["url"] for p in pages if "url" in p]

        if wd.get("fullMatchingImages"):
            return urls, 1.00   # exact match
        if wd.get("partialMatchingImages"):
            return urls, 0.80   # high similarity
        return urls, 0.50 if urls else 0.0   # partial (pages only)
    except (KeyError, IndexError):
        return [], 0.0


# ---------------------------------------------------------------------------
# Bing Visual Search
# ---------------------------------------------------------------------------

def _bing_visual_search(image_bytes: bytes) -> Optional[Dict]:
    """
    Call the Bing Visual Search API.
    Returns the parsed JSON response or None on any error.

    Environment variable required: ``BING_SEARCH_KEY``
    """
    if not _bing_search_key():
        return None

    boundary = "----BoundaryImageSearch"
    body     = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="image.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://api.bing.microsoft.com/v7.0/images/visualsearch",
            data=body,
            headers={
                "Ocp-Apim-Subscription-Key": _bing_search_key(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _parse_bing_response(resp: Dict) -> Tuple[List[str], float]:
    """Return (source_urls, best_similarity) from a Bing Visual Search response."""
    try:
        urls: List[str] = []
        for tag in resp.get("tags", []):
            for action in tag.get("actions", []):
                if action.get("actionType") in ("VisualSearch", "RelatedSearches"):
                    for item in action.get("data", {}).get("value", []):
                        if "hostPageUrl" in item:
                            urls.append(item["hostPageUrl"])
        return urls[:10], 0.50 if urls else 0.0   # partial match tier
    except Exception:
        return [], 0.0


# ---------------------------------------------------------------------------
# SerpApi (Google Lens)
# ---------------------------------------------------------------------------

def _serpapi_search(image_bytes: bytes) -> Optional[Dict]:
    """
    Upload image to Google Lens via SerpApi.
    Requires env-var ``SERPAPI_KEY``.
    """
    if not _serpapi_key():
        return None

    b64  = base64.b64encode(image_bytes).decode("utf-8")
    data = urllib.parse.urlencode({
        "engine":   "google_lens",
        "url":      f"data:image/jpeg;base64,{b64}",
        "api_key":  _serpapi_key(),
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://serpapi.com/search",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _parse_serpapi_response(resp: Dict) -> Tuple[List[str], float]:
    """Return (source_urls, best_similarity) from a SerpApi Google Lens response."""
    try:
        results = resp.get("visual_matches", [])
        urls    = [r["link"] for r in results if "link" in r]
        return urls[:10], 0.50 if urls else 0.0   # partial match tier
    except Exception:
        return [], 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_bytes: bytes) -> ReverseSearchResult:
    """
    Attempt a reverse image search for *image_bytes*.

    Returns immediately with ``ran=False`` if no API key is configured.
    Returns ``ran=True, found=False, error=<msg>`` if a key is set but the
    API call fails — so the caller can distinguish "not configured" from
    "configured but failed".

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.

    Returns
    -------
    ReverseSearchResult
    """
    # ── Short-circuit when no key is configured at all ─────────────────────
    if not any([_google_api_key(), _bing_search_key(), _serpapi_key()]):
        return ReverseSearchResult(ran=False)

    # ── Try backends in priority order ──────────────────────────────────────
    source_urls: List[str] = []
    api_similarity: float  = 0.0
    last_error: Optional[str] = None

    response, err = _google_web_detection(image_bytes)
    if response is not None:
        source_urls, api_similarity = _parse_google_response(response)
    elif err:
        last_error = err
        # Fall through to Bing / SerpApi only if Google key wasn't the problem
        response = _bing_visual_search(image_bytes)
        if response is not None:
            source_urls, api_similarity = _parse_bing_response(response)
            last_error = None
        else:
            response = _serpapi_search(image_bytes)
            if response is not None:
                source_urls, api_similarity = _parse_serpapi_response(response)
                last_error = None
    else:
        # Google key not set — try Bing then SerpApi
        response = _bing_visual_search(image_bytes)
        if response is not None:
            source_urls, api_similarity = _parse_bing_response(response)
        else:
            response = _serpapi_search(image_bytes)
            if response is not None:
                source_urls, api_similarity = _parse_serpapi_response(response)

    # ── If every backend failed (but at least one key was set) ──────────────
    if response is None and source_urls == []:
        return ReverseSearchResult(
            ran   = True,
            found = False,
            error = last_error or "API call failed — check key permissions",
        )

    if not source_urls:
        return ReverseSearchResult(found=False, similarity=0.0, ran=True)

    # ── pHash refinement ────────────────────────────────────────────────────
    try:
        query_img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        phash_sims: List[float] = []
        for url in source_urls[: _MAX_SOURCE_DL]:
            src_img = _download_image(url)
            if src_img is not None:
                phash_sims.append(_phash_similarity(query_img, src_img))

        phash_sim = float(max(phash_sims)) if phash_sims else api_similarity
    except Exception:
        phash_sim = api_similarity

    # Blend API signal and pHash
    similarity = 0.5 * api_similarity + 0.5 * phash_sim

    return ReverseSearchResult(
        found       = True,
        similarity  = float(np.clip(similarity, 0.0, 1.0)),
        source_urls = source_urls,
        ran         = True,
    )
