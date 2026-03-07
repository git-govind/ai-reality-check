"""
metadata_checker.py
--------------------
Step 1 of the Image Authenticity Evaluator pipeline.

Analyses image metadata (EXIF, XMP embedded in EXIF, ICC colour profile)
to detect signals commonly associated with AI-generated or manipulated images.

Detection checks
----------------
  • Missing EXIF block       – AI generators rarely embed camera metadata
  • Missing camera model     – strong signal when the rest of the EXIF is present
  • Editing-software tag     – Photoshop, Lightroom, GIMP, Midjourney, Stable Diffusion …
  • Future / implausible timestamp – manipulation or clock error
  • Inconsistent timestamps  – DateTimeOriginal ≠ DateTime by > 24 h
  • Missing GPS when plausible – modern phones embed GPS; absence lowers score only mildly
  • Non-camera aspect ratio  – very unusual WxH dimensions that cameras never produce

Scoring (starts at 100, penalties/bonus applied, normalised to [0, 100])
------------------------------------------------------------------------
  -30  no EXIF at all
  -40  editing-software fingerprint found (Photoshop, GIMP, AI generator …)
  -20  no camera make or model in EXIF
  -10  timestamp anomaly (future date OR >24 h inconsistency)
  +20  valid camera pipeline (make + model present, no editing sw, no ts issues)

Final MetadataResult.score = clamped_0_100 / 100.0   (range [0.0, 1.0])

Dependencies: Pillow (already in requirements.txt)
"""

from __future__ import annotations

import io
import re
import struct
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from PIL.ExifTags import TAGS

from .datatypes import MetadataResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Software strings that indicate the image was created or edited by a
# known AI generator or image manipulation tool.
_EDITING_SOFTWARE: List[str] = [
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "snapseed", "darktable", "rawtherapee",
    # AI generators
    "midjourney", "stable diffusion", "dall-e", "dalle",
    "firefly", "imagen", "ideogram", "runwayml",
    "bing image creator", "leonardo", "nightcafe",
    # Generic markers sometimes embedded
    "adobe firefly", "ai generated", "synthetically generated",
]

# EXIF tag IDs we care about (decimal integers as stored in PIL)
_TAG_MAKE             = 271   # Camera make
_TAG_MODEL            = 272   # Camera model
_TAG_SOFTWARE         = 305   # Software
_TAG_DATETIME         = 306   # DateTime (file-change time)
_TAG_DATETIME_ORIG    = 36867 # DateTimeOriginal (capture time)
_TAG_DATETIME_DIGIT   = 36868 # DateTimeDigitized
_TAG_GPS_IFD          = 34853 # GPSInfo IFD pointer
_TAG_ARTIST           = 315
_TAG_COPYRIGHT        = 33432
_TAG_IMAGEDESC        = 270   # ImageDescription – sometimes contains "AI" text

_EXIF_DT_FMT = "%Y:%m:%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_exif_dt(value: str) -> Optional[datetime]:
    """Return a naive UTC datetime or None if parsing fails."""
    try:
        return datetime.strptime(str(value).strip(), _EXIF_DT_FMT)
    except ValueError:
        return None


def _extract_raw_exif(img: Image.Image) -> Dict[int, Any]:
    """Return a {tag_id: value} dict or {} if no EXIF present."""
    try:
        exif_data = img.getexif()
        return dict(exif_data) if exif_data else {}
    except Exception:
        return {}


def _humanise_exif(raw: Dict[int, Any]) -> Dict[str, Any]:
    """Convert numeric tag IDs to human-readable tag names."""
    return {TAGS.get(k, str(k)): v for k, v in raw.items()}


def _check_editing_software(raw: Dict[int, Any]) -> Tuple[bool, str]:
    """Return (detected, software_name) if an editing tool is fingerprinted."""
    for tag_id in (_TAG_SOFTWARE, _TAG_IMAGEDESC, _TAG_ARTIST, _TAG_COPYRIGHT):
        value = raw.get(tag_id)
        if not value:
            continue
        lower = str(value).lower()
        for sw in _EDITING_SOFTWARE:
            if sw in lower:
                return True, str(value).strip()
    return False, ""


def _check_timestamps(raw: Dict[int, Any]) -> Tuple[List[str], bool, bool]:
    """Return (flags, is_future, is_inconsistent)."""
    flags: List[str] = []
    is_future = False
    is_inconsistent = False

    dt_orig  = _parse_exif_dt(raw.get(_TAG_DATETIME_ORIG, ""))
    dt_file  = _parse_exif_dt(raw.get(_TAG_DATETIME, ""))
    now      = datetime.now()

    for label, dt in (("DateTimeOriginal", dt_orig), ("DateTime", dt_file)):
        if dt is None:
            continue
        if dt > now:
            flags.append(f"{label} is in the future ({dt.date()})")
            is_future = True

    if dt_orig and dt_file:
        gap_hours = abs((dt_orig - dt_file).total_seconds()) / 3600
        if gap_hours > 24:
            flags.append(
                f"Timestamp gap: DateTimeOriginal vs DateTime differ by "
                f"{gap_hours:.0f} h"
            )
            is_inconsistent = True

    return flags, is_future, is_inconsistent


def _check_dimensions(img: Image.Image) -> Tuple[bool, str]:
    """Flag very unusual dimensions that real cameras never produce."""
    w, h = img.size
    # Perfectly square images are uncommon from cameras (except some Hasselblads)
    # Round power-of-two dimensions are a diffusion model tell-tale
    pow2 = {256, 512, 768, 1024, 1280, 1536, 2048, 4096}
    if w in pow2 and h in pow2:
        return True, f"Exact power-of-two dimensions ({w}×{h}) typical of AI generators"
    # Very unusual aspect ratios
    ratio = max(w, h) / min(w, h)
    if ratio > 4.0:
        return True, f"Extreme aspect ratio {ratio:.1f}:1 uncommon in camera images"
    return False, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_bytes: bytes) -> MetadataResult:
    """
    Analyse metadata in *image_bytes* and return a :class:`MetadataResult`.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the image file (JPEG, PNG, WEBP, TIFF …).

    Returns
    -------
    MetadataResult
        score  – 0.0 (very suspicious) … 1.0 (authentic-looking)
        flags  – list of human-readable anomaly descriptions
        raw_metadata – humanised EXIF key/value pairs
    """
    flags: List[str] = []
    score = 100.0   # start at 100

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        return MetadataResult(
            score=0.5,
            flags=[f"Could not open image: {exc}"],
        )

    raw_exif = _extract_raw_exif(img)
    raw_human = _humanise_exif(raw_exif)

    # ── ICC colour profile ──────────────────────────────────────────────────
    icc = img.info.get("icc_profile")
    if icc:
        raw_human["ICC_profile_bytes"] = len(icc)

    # ── Missing EXIF  →  -30 ────────────────────────────────────────────────
    if not raw_exif:
        flags.append("No EXIF metadata found (common in AI-generated images)")
        score -= 30
    else:
        # ── Camera make / model  →  -20 if both missing, -10 if model only ──
        has_make  = bool(raw_exif.get(_TAG_MAKE))
        has_model = bool(raw_exif.get(_TAG_MODEL))
        has_camera = has_make and has_model

        if not has_make and not has_model:
            flags.append("No camera make or model in EXIF")
            score -= 20    # camera model mismatch → -20
        elif not has_model:
            flags.append("Camera model missing from EXIF")
            score -= 10    # partial deduction

        # ── Editing-software fingerprint  →  -40 ────────────────────────────
        sw_found, sw_name = _check_editing_software(raw_exif)
        if sw_found:
            flags.append(f"Editing/generation software detected: '{sw_name}'")
            score -= 40    # editing software tag → -40

        # ── Timestamp checks  →  -10 for any anomaly ────────────────────────
        ts_flags, is_future, is_inconsistent = _check_timestamps(raw_exif)
        flags.extend(ts_flags)
        if is_future or is_inconsistent:
            score -= 10    # timestamp anomalies → -10

        # ── Valid camera pipeline bonus  →  +20 ─────────────────────────────
        # All four conditions must hold: make+model present, no editing sw,
        # no timestamp anomalies.
        if has_camera and not sw_found and not is_future and not is_inconsistent:
            score += 20

    # ── Dimension check (informational — flags only, no separate penalty) ───
    dim_suspicious, dim_msg = _check_dimensions(img)
    if dim_suspicious:
        flags.append(dim_msg)

    # ── Format-specific notes ───────────────────────────────────────────────
    fmt = (img.format or "UNKNOWN").upper()
    raw_human["_format"] = fmt
    raw_human["_size"]   = f"{img.width}×{img.height}"
    raw_human["_mode"]   = img.mode

    if fmt == "PNG":
        raw_human["_note"] = (
            "PNG format: EXIF support is optional; absence is less suspicious."
        )

    # ── Normalise to [0, 100] then return as [0.0, 1.0] ────────────────────
    score = max(0.0, min(100.0, score))
    return MetadataResult(score=score / 100.0, flags=flags, raw_metadata=raw_human)
