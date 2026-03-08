"""
image_text_consistency.py
--------------------------
Step 4 of the Image Authenticity Evaluator pipeline — OPTIONAL.

Only runs when a caption / prompt string is provided.  Measures how well
the image content aligns with the supplied text description.

Strategy
--------
  Model path (preferred)
      Uses the CLIP model via sentence-transformers (``clip-ViT-B-32``).
      Computes cosine similarity between the image embedding and the
      caption text embedding.

        score = cosine_similarity(image_emb, text_emb)
                mapped from the typical CLIP range (~0.15–0.35) to [0, 1].

      Additionally re-ranks multiple sub-phrases split from the caption
      to flag which objects / attributes are most poorly represented.

  Keyword heuristic path (fallback when CLIP is unavailable)
      Extracts meaningful tokens from the caption (nouns, colour words,
      numbers) and counts how many can be associated with dominant visual
      features (dominant colours, rough object detection via pixel statistics).
      This is coarse but requires zero extra dependencies.

  Skipped path
      If no caption is provided, returns ConsistencyResult(ran=False).

Dependencies: Pillow, NumPy (required); sentence-transformers (optional)
"""

from __future__ import annotations

import io
import re
from typing import List, Tuple

import numpy as np
from PIL import Image

from .datatypes import ConsistencyResult
from models.embeddings_registry import get_model as _get_embedding_model

# ---------------------------------------------------------------------------
# Colour names used in heuristic path
# ---------------------------------------------------------------------------

_COLOUR_MAP: dict[str, Tuple[int, int, int]] = {
    "red":     (200,  50,  50),
    "green":   ( 50, 180,  50),
    "blue":    ( 50,  50, 200),
    "yellow":  (220, 200,  50),
    "orange":  (220, 130,  50),
    "purple":  (130,  50, 180),
    "pink":    (220, 120, 150),
    "white":   (230, 230, 230),
    "black":   ( 30,  30,  30),
    "gray":    (128, 128, 128),
    "grey":    (128, 128, 128),
    "brown":   (120,  70,  40),
    "cyan":    ( 50, 200, 200),
    "magenta": (200,  50, 200),
}

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
    "at", "to", "for", "with", "and", "or", "but", "it", "its", "this",
    "that", "there", "their", "they", "has", "have", "be", "been",
    "by", "from", "as", "which", "who", "very", "quite", "just",
}

# ---------------------------------------------------------------------------
# Model-based consistency
# ---------------------------------------------------------------------------

_SPATIAL_WORDS = {
    "above", "below", "behind", "front", "left", "right",
    "inside", "outside", "floating", "upside", "vertical",
    "horizontal", "next to", "on top", "beneath", "beside",
}

_LIGHTING_WORDS = {
    "dark", "bright", "shadow", "sunlit", "moonlit", "foggy",
    "glowing", "backlit", "silhouette", "illuminated", "sunlight",
    "nighttime", "daytime", "overcast", "hazy", "neon",
}


# ---------------------------------------------------------------------------
# Model-based consistency with penalty scoring (spec §2)
# ---------------------------------------------------------------------------

def _clip_consistency(img: Image.Image, caption: str) -> Tuple[float, List[str]]:
    """
    Penalty-based consistency score using CLIP.

    Starts at 100 and applies the following deductions (spec §2):
      -40  Object mismatch        (overall CLIP similarity too low)
      -30  Impossible geometry    (spatial words in caption, low similarity)
      -20  Lighting/physics mismatch (lighting words in caption, low similarity)
      -10  Caption hallucination  (per poorly matched sub-phrase, max -30)

    Returns (score 0-1, issues).
    """
    clip = _get_embedding_model("clip-ViT-B-32")
    if not clip.available or clip.model is None:
        return 0.5, []

    try:
        from sentence_transformers import util  # type: ignore

        img_emb = clip.model.encode(img,     convert_to_tensor=True)
        cap_emb = clip.model.encode(caption, convert_to_tensor=True)
        overall = float(util.cos_sim(img_emb, cap_emb).item())

        # Map CLIP range ~[0.10, 0.35] → [0, 1]
        norm = float(np.clip((overall - 0.10) / 0.25, 0.0, 1.0))

        score_100 = 100.0
        issues: List[str] = []
        cap_lower = caption.lower()

        # ── Object mismatch  →  -40 ─────────────────────────────────────────
        if norm < 0.25:
            score_100 -= 40
            issues.append(
                f"Object mismatch: overall CLIP similarity too low "
                f"({overall:.3f}) — image content may not match caption"
            )

        # ── Impossible geometry  →  -30 ──────────────────────────────────────
        # Triggered when spatial positional claims exist but overall match is weak
        has_spatial = any(w in cap_lower for w in _SPATIAL_WORDS)
        if has_spatial and norm < 0.40:
            score_100 -= 30
            issues.append(
                "Impossible geometry: spatial positioning claim in caption "
                "is not supported by the image content"
            )

        # ── Lighting/physics mismatch  →  -20 ────────────────────────────────
        has_lighting = any(w in cap_lower for w in _LIGHTING_WORDS)
        if has_lighting and norm < 0.40:
            score_100 -= 20
            issues.append(
                "Lighting/physics mismatch: lighting or atmospheric condition "
                "in caption is inconsistent with the image"
            )

        # ── Caption hallucination  →  -10 per bad phrase (max -30) ───────────
        chunks = re.split(r"[,;]|\band\b|\bwith\b", caption)
        hallucination_penalty = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk.split()) < 2:
                continue
            c_emb  = clip.model.encode(chunk, convert_to_tensor=True)
            c_sim  = float(util.cos_sim(img_emb, c_emb).item())
            c_norm = float(np.clip((c_sim - 0.10) / 0.25, 0.0, 1.0))
            if c_norm < 0.20:
                if hallucination_penalty < 30:
                    score_100 -= 10
                    hallucination_penalty += 10
                issues.append(
                    f"Caption hallucination: '{chunk}' not visible in image "
                    f"(similarity={c_norm:.2f})"
                )

        score_100 = max(0.0, min(100.0, score_100))
        return score_100 / 100.0, issues

    except Exception as exc:
        return 0.5, [f"CLIP consistency error: {exc}"]


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _dominant_colours(img_rgb: np.ndarray, n: int = 5) -> List[Tuple[int, int, int]]:
    """Return n dominant RGB colours via k-means-style quantisation using numpy."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    # Simple random-sample centroid initialisation
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(pixels), size=min(n, len(pixels)), replace=False)
    centres = pixels[idx].copy()

    for _ in range(10):  # k-means iterations
        dists   = np.linalg.norm(pixels[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=2)
        labels  = np.argmin(dists, axis=1)
        for k in range(n):
            mask = labels == k
            if mask.any():
                centres[k] = pixels[mask].mean(axis=0)

    return [(int(c[0]), int(c[1]), int(c[2])) for c in centres]


def _nearest_colour_name(rgb: Tuple[int, int, int]) -> str:
    best, best_dist = "unknown", float("inf")
    for name, ref in _COLOUR_MAP.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref)) ** 0.5
        if dist < best_dist:
            best_dist, best = dist, name
    return best


def _keyword_consistency(img_rgb: np.ndarray, caption: str) -> Tuple[float, List[str]]:
    """
    Penalty-based keyword-matching heuristic when CLIP is unavailable.

    Deductions (spec §2):
      -40  Object mismatch        (caption colour words absent from dominant colours)
      -20  Lighting/physics mismatch (lighting words present but no matching colour)

    Returns (score 0-1, issues).
    """
    tokens   = re.findall(r"\b[a-z]+\b", caption.lower())
    keywords = {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}

    dom_colours  = _dominant_colours(img_rgb, n=4)
    colour_names = {_nearest_colour_name(c) for c in dom_colours}

    caption_colours = keywords & set(_COLOUR_MAP.keys())
    matched         = caption_colours & colour_names

    score_100 = 100.0
    issues: List[str] = []

    # Object mismatch  →  -40
    if caption_colours and not matched:
        score_100 -= 40
        issues.append(
            f"Object mismatch: caption mentions colours {caption_colours} "
            f"but dominant image colours are {colour_names}"
        )

    # Lighting/physics mismatch  →  -20
    cap_lower = caption.lower()
    has_lighting = any(w in cap_lower for w in _LIGHTING_WORDS)
    if has_lighting:
        night_words  = {"dark", "nighttime", "moonlit", "neon", "shadow"}
        bright_words = {"bright", "sunlit", "sunlight", "daytime", "illuminated"}
        night_claim  = any(w in cap_lower for w in night_words)
        bright_claim = any(w in cap_lower for w in bright_words)
        # "white" or "bright" dominant colour contradicts darkness claim
        if night_claim and ("white" in colour_names or "yellow" in colour_names):
            score_100 -= 20
            issues.append(
                "Lighting/physics mismatch: caption claims dark/night scene "
                "but image has bright dominant colours"
            )
        elif bright_claim and "black" in colour_names:
            score_100 -= 20
            issues.append(
                "Lighting/physics mismatch: caption claims bright scene "
                "but image is predominantly dark"
            )

    issues.insert(0, "Note: CLIP unavailable — using keyword heuristic only")
    score_100 = max(0.0, min(100.0, score_100))
    return score_100 / 100.0, issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_bytes: bytes, caption: str | None = None) -> ConsistencyResult:
    """
    Check image–text consistency.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.
    caption : str or None
        Optional caption / prompt associated with the image.
        If None, the step is skipped and ConsistencyResult(ran=False) is returned.

    Returns
    -------
    ConsistencyResult
    """
    if not caption or not caption.strip():
        return ConsistencyResult(score=0.5, issues=[], ran=False)

    try:
        img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(img, dtype=np.float32)
    except Exception as exc:
        return ConsistencyResult(
            score=0.5,
            issues=[f"Could not decode image: {exc}"],
            ran=True,
        )

    if _get_embedding_model("clip-ViT-B-32").available:
        score, issues = _clip_consistency(img, caption)
    else:
        score, issues = _keyword_consistency(img_rgb, caption)

    return ConsistencyResult(score=score, issues=issues, ran=True)
