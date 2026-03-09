"""
image_watermark_detector.py
----------------------------
AI watermark detection module for the Image Authenticity Evaluator pipeline.

Detection strategies (in priority order):
  1. Metadata-based   — software/generator EXIF tags already extracted by
                        metadata_checker (confidence 0.90)
  2. Visible text     — PNG/image text chunks + optional OCR backend hook
                        (confidence 0.80–0.85)
  3. SD invisible     — Stable Diffusion invisible-watermark stub
                        (returns None until the dependency is wired in)
  4. SynthID          — Google SynthID invisible-watermark stub
                        (returns None until the API/model is wired in)

Design principles
-----------------
* Watermark *presence* → strong positive AI signal (caller clamps ai_likelihood ≥ 85 %)
* Watermark *absence*  → neutral; never reduces ai_likelihood
* No hard dependencies on OCR or external APIs — hooks are pluggable via
  ``register_ocr_backend()``
* Fully unit-testable: all public functions accept plain bytes / dataclass args
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Known AI generator keyword lists
# ---------------------------------------------------------------------------

_AI_SOFTWARE_KEYWORDS: List[str] = [
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dall·e",
    "dalle",
    "adobe firefly",
    "firefly",
    "imagen",
    "dreamstudio",
    "leonardo.ai",
    "nightcafe",
    "wombo",
    "bing image creator",
    "runway",
    "pika",
    "sora",
    "getimg",
    "tensor.art",
    "novelai",
    "ideogram",
    "blue willow",
    "playground ai",
    "clipdrop",
    "hotpot.ai",
]

# Regex patterns applied to OCR output and PNG text chunks
_VISIBLE_WATERMARK_PATTERNS: List[str] = [
    r"generated\s+(?:by|with|using)",
    r"midjourney",
    r"dall[\-·]?e",
    r"stable\s+diffusion",
    r"ai[\s\-]generated",
    r"made\s+with\s+ai",
    r"@midjourney",
    r"adobe\s+firefly",
    r"firefly",
    r"imagen",
    r"dreamstudio",
    r"created\s+(?:by|with)\s+ai",
    r"ai\s+art",
]

# PNG / image info dict keys that may contain generator text
_TEXT_CHUNK_KEYS = (
    "parameters",   # Stable Diffusion AUTOMATIC1111 / ComfyUI standard
    "prompt",       # alternative key used by some UIs
    "workflow",     # ComfyUI workflow JSON (contains model names)
    "comment",
    "Comment",
    "description",
    "Description",
    "UserComment",
    "exif_comment",
)


# ---------------------------------------------------------------------------
# WatermarkResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class WatermarkResult:
    """Outcome of a single watermark detection run."""

    has_watermark: bool
    watermark_type: Optional[str]   # "visible_text" | "sd_invisible" | "synthid" | "metadata_tag"
    confidence: float               # 0.0 – 1.0
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OCR integration hook
# ---------------------------------------------------------------------------
# Register any OCR callable to enable text-based visible watermark detection.
# The callable must accept a PIL.Image and return a str (or None / "").
#
# Example (pytesseract):
#   import pytesseract
#   from image_evaluator.image_watermark_detector import register_ocr_backend
#   register_ocr_backend(lambda img: pytesseract.image_to_string(img))
#
# Example (easyocr):
#   import easyocr
#   reader = easyocr.Reader(["en"])
#   register_ocr_backend(lambda img: " ".join(reader.readtext(img, detail=0)))

_OCR_BACKEND: Optional[object] = None  # type: ignore[assignment]


def register_ocr_backend(fn) -> None:
    """Register an OCR backend: ``fn(pil_image: Image.Image) -> str | None``."""
    global _OCR_BACKEND
    _OCR_BACKEND = fn


def _ocr_hook(img: Image.Image) -> Optional[str]:
    """Invoke the registered OCR backend; return None if none is registered."""
    if _OCR_BACKEND is None:
        return None
    try:
        result = _OCR_BACKEND(img)  # type: ignore[call-arg]
        return str(result) if result else None
    except Exception:
        return None


def _downscale_for_ocr(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resize so the longest side is at most *max_side* pixels."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Invisible watermark detector stubs (extensible)
# ---------------------------------------------------------------------------

def detect_sd_invisible_watermark(image_bytes: bytes) -> Optional[WatermarkResult]:
    """
    Stub: Stable Diffusion invisible watermark detector.

    To implement, install the ``invisible-watermark`` package and replace
    this function body with real detection logic::

        from imwatermark import WatermarkDecoder
        import numpy as np
        from PIL import Image
        import io

        img_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        decoder = WatermarkDecoder("bytes", 48)
        wm = decoder.decode(img_np, "dwtDctSvd")
        if wm != b"\\x00" * 6:
            return WatermarkResult(
                has_watermark=True,
                watermark_type="sd_invisible",
                confidence=0.85,
                details={"raw_bytes": wm.hex()},
            )

    Returns None until a real implementation is provided.
    """
    return None


def detect_synthid_watermark(image_bytes: bytes) -> Optional[WatermarkResult]:
    """
    Stub: Google SynthID invisible watermark detector.

    To implement, integrate the SynthID API or open-source detector model::

        # Pseudocode — actual API subject to change
        from synthid_client import detect
        result = detect(image_bytes)
        if result.score > 0.7:
            return WatermarkResult(
                has_watermark=True,
                watermark_type="synthid",
                confidence=float(result.score),
                details={"raw_score": result.score},
            )

    Reference: https://deepmind.google/technologies/synthid/

    Returns None until a real implementation is provided.
    """
    return None


# ---------------------------------------------------------------------------
# Internal detection strategies
# ---------------------------------------------------------------------------

def _detect_from_metadata(metadata_result) -> Optional[WatermarkResult]:
    """
    Check a ``MetadataResult`` for AI generator tags.

    Uses ``detected_ai_generator`` (already set by ``metadata_checker``) as the
    primary signal, then scans all raw metadata fields as a fallback.
    """
    # Fast path: metadata_checker already identified an AI generator
    gen = getattr(metadata_result, "detected_ai_generator", "")
    if gen:
        return WatermarkResult(
            has_watermark=True,
            watermark_type="metadata_tag",
            confidence=0.90,
            details={"detected_generator": gen, "source": "detected_ai_generator"},
        )

    # Full scan of raw metadata fields
    raw: Dict[str, Any] = getattr(metadata_result, "raw_metadata", {})

    # Priority fields are checked first; remaining string fields follow
    priority_fields = [
        "Software", "Artist", "Copyright", "ImageDescription",
        "UserComment", "Comment", "parameters",
    ]
    scan_pairs: List[tuple] = []
    for key in priority_fields:
        if key in raw:
            scan_pairs.append((key, str(raw[key])))
    for key, val in raw.items():
        if key not in priority_fields and isinstance(val, str):
            scan_pairs.append((key, val))

    for field_name, field_val in scan_pairs:
        field_lower = field_val.lower()
        for kw in _AI_SOFTWARE_KEYWORDS:
            if kw in field_lower:
                return WatermarkResult(
                    has_watermark=True,
                    watermark_type="metadata_tag",
                    confidence=0.90,
                    details={"matched_keyword": kw, "field": field_name},
                )

    return None


def detect_visible_watermark(image_bytes: bytes) -> Optional[WatermarkResult]:
    """
    Search for visible AI watermarks using two sub-strategies:

    1. **PNG / image text chunks** — Stable Diffusion stores generation
       parameters under the ``"parameters"`` key; Midjourney and other tools
       embed similar text in ``"comment"``, ``"prompt"``, or ``"workflow"``
       fields.  These are checked without any external dependency.

    2. **OCR hook** — If a backend has been registered via
       :func:`register_ocr_backend`, the image is downscaled to ≤ 1024 px
       and passed to the backend.  The resulting text is matched against
       :data:`_VISIBLE_WATERMARK_PATTERNS` and :data:`_AI_SOFTWARE_KEYWORDS`.

    Returns ``None`` when no visible watermark is found.  Absence is neutral
    and is never interpreted as an authenticity signal by the caller.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None

    # ── 1. Image text chunks (PNG info dict) ─────────────────────────────────
    info: Dict[str, Any] = getattr(img, "info", {}) or {}

    for chunk_key in _TEXT_CHUNK_KEYS:
        if chunk_key not in info:
            continue
        chunk_val = str(info[chunk_key]).lower()

        for kw in _AI_SOFTWARE_KEYWORDS:
            if kw in chunk_val:
                return WatermarkResult(
                    has_watermark=True,
                    watermark_type="visible_text",
                    confidence=0.80,
                    details={
                        "source": "image_text_chunk",
                        "chunk_key": chunk_key,
                        "matched_keyword": kw,
                    },
                )
        for pattern in _VISIBLE_WATERMARK_PATTERNS:
            if re.search(pattern, chunk_val, re.IGNORECASE):
                return WatermarkResult(
                    has_watermark=True,
                    watermark_type="visible_text",
                    confidence=0.80,
                    details={
                        "source": "image_text_chunk",
                        "chunk_key": chunk_key,
                        "matched_pattern": pattern,
                    },
                )

    # ── 2. OCR hook ───────────────────────────────────────────────────────────
    ocr_text = _ocr_hook(_downscale_for_ocr(img))
    if ocr_text:
        ocr_lower = ocr_text.lower()
        for pattern in _VISIBLE_WATERMARK_PATTERNS:
            if re.search(pattern, ocr_lower, re.IGNORECASE):
                return WatermarkResult(
                    has_watermark=True,
                    watermark_type="visible_text",
                    confidence=0.85,
                    details={"source": "ocr", "matched_pattern": pattern},
                )
        for kw in _AI_SOFTWARE_KEYWORDS:
            if kw in ocr_lower:
                return WatermarkResult(
                    has_watermark=True,
                    watermark_type="visible_text",
                    confidence=0.85,
                    details={"source": "ocr", "matched_keyword": kw},
                )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_watermarks(
    image_bytes: bytes,
    metadata_result=None,
) -> WatermarkResult:
    """
    Run all watermark detection strategies and return the highest-confidence result.

    Strategy execution order
    ------------------------
    1. ``metadata_tag``  — re-uses already-extracted ``MetadataResult``; fast,
                           no image decode required (confidence 0.90)
    2. ``visible_text``  — PNG text chunks + optional OCR hook (confidence 0.80–0.85)
    3. ``sd_invisible``  — Stable Diffusion invisible-watermark stub (None until wired)
    4. ``synthid``       — Google SynthID stub (None until wired)

    Parameters
    ----------
    image_bytes :
        Raw image bytes (JPEG, PNG, WEBP, …).
    metadata_result :
        Optional ``MetadataResult`` returned by ``metadata_checker.run()``.
        Passed in to avoid re-parsing EXIF; if ``None``, metadata-based
        detection is skipped.

    Returns
    -------
    WatermarkResult
        Always returns a result.  ``has_watermark=False`` when no watermark is
        found — the caller must **not** treat absence as an authenticity signal.
    """
    candidates: List[WatermarkResult] = []

    # Strategy 1: metadata tags (fast — no image decode needed)
    if metadata_result is not None:
        result = _detect_from_metadata(metadata_result)
        if result is not None:
            candidates.append(result)

    # Strategy 2: visible text in image (PNG text chunks + OCR hook)
    try:
        result = detect_visible_watermark(image_bytes)
        if result is not None:
            candidates.append(result)
    except Exception:
        pass

    # Strategy 3: SD invisible watermark (stub — returns None until wired)
    try:
        result = detect_sd_invisible_watermark(image_bytes)
        if result is not None:
            candidates.append(result)
    except Exception:
        pass

    # Strategy 4: SynthID (stub — returns None until wired)
    try:
        result = detect_synthid_watermark(image_bytes)
        if result is not None:
            candidates.append(result)
    except Exception:
        pass

    if not candidates:
        return WatermarkResult(
            has_watermark=False,
            watermark_type=None,
            confidence=0.0,
            details={},
        )

    # Return the single highest-confidence detection
    return max(candidates, key=lambda r: r.confidence)
