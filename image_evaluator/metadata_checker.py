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
  -30  no EXIF at all for JPEG/WEBP/TIFF  (PNG exempt — EXIF is optional)
  -40  AI generator software tag  (Midjourney, Stable Diffusion, DALL-E, Firefly …)
  -15  photo editor software tag  (Photoshop, Lightroom, GIMP …)
  -20  no camera make or model in EXIF
  -10  timestamp anomaly (future date OR >24 h inconsistency)
  +20  valid camera pipeline (make + model present, no editing sw, no ts issues)
  +15  MakerNote present (camera-proprietary metadata block)
  +10  GPS coordinates embedded (only real cameras write GPS IFD)
  +10  EXIF thumbnail present (written by camera firmware, not by AI generators)

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

from config_loader import get_threshold

# ---------------------------------------------------------------------------
# Config-driven scoring constants (loaded once at import time)
# ---------------------------------------------------------------------------
_META_PENALTY_NO_EXIF       = get_threshold("image.metadata_penalty.no_exif")
_META_PENALTY_AI_GEN        = get_threshold("image.metadata_penalty.ai_generator")
_META_PENALTY_PHOTO_EDITOR  = get_threshold("image.metadata_penalty.photo_editor")
_META_PENALTY_NO_CAMERA     = get_threshold("image.metadata_penalty.no_camera")
_META_PENALTY_MODEL_ONLY    = get_threshold("image.metadata_penalty.camera_model_only")
_META_PENALTY_TIMESTAMP     = get_threshold("image.metadata_penalty.timestamp_anomaly")
_META_PENALTY_AI_DIMENSIONS = get_threshold("image.metadata_penalty.ai_dimensions")
_META_BONUS_CAMERA_PIPELINE = get_threshold("image.metadata_bonus.camera_pipeline")
_META_BONUS_MAKERNOTE       = get_threshold("image.metadata_bonus.makernote")
_META_BONUS_GPS             = get_threshold("image.metadata_bonus.gps")
_META_BONUS_THUMBNAIL       = get_threshold("image.metadata_bonus.thumbnail")
_META_BAND_NO_EXIF          = get_threshold("image.metadata_band.no_exif")
_META_BAND_ZERO_SIGNALS     = get_threshold("image.metadata_band.zero_signals")
_META_BAND_ONE_SIGNAL       = get_threshold("image.metadata_band.one_signal")
_META_BAND_TWO_SIGNALS      = get_threshold("image.metadata_band.two_signals")
_META_BAND_THREE_OR_MORE    = get_threshold("image.metadata_band.three_or_more")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Software strings that indicate the image was created by a known AI generator.
# Penalty: -40  (strong authenticity signal)
_AI_GENERATORS: List[str] = [
    "midjourney", "stable diffusion", "dall-e", "dalle",
    "firefly", "adobe firefly", "imagen", "ideogram", "runwayml",
    "bing image creator", "leonardo", "nightcafe",
    # Generic embedded markers
    "ai generated", "synthetically generated",
]

# Software strings that indicate a conventional photo editor was used.
# Penalty: -15  (editing is common; not a strong AI signal on its own)
_PHOTO_EDITORS: List[str] = [
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "snapseed", "darktable", "rawtherapee",
]

# Image formats where missing EXIF is normal and should NOT be penalised.
_EXIF_OPTIONAL_FORMATS = {"PNG", "GIF", "BMP", "WEBP"}

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
_TAG_MAKERNOTE        = 37500 # Camera manufacturer private data (never in AI images)

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


def _check_editing_software(raw: Dict[int, Any]) -> Tuple[bool, str, str]:
    """
    Detect editing / generation software fingerprints in EXIF tags.

    Returns
    -------
    (detected, software_name, category)
        detected       – True if any known software was found
        software_name  – raw tag value from the image
        category       – "ai_generator" | "photo_editor" | ""
    """
    for tag_id in (_TAG_SOFTWARE, _TAG_IMAGEDESC, _TAG_ARTIST, _TAG_COPYRIGHT):
        value = raw.get(tag_id)
        if not value:
            continue
        lower = str(value).lower()
        for sw in _AI_GENERATORS:
            if sw in lower:
                return True, str(value).strip(), "ai_generator"
        for sw in _PHOTO_EDITORS:
            if sw in lower:
                return True, str(value).strip(), "photo_editor"
    return False, "", ""


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


def _check_makernote(raw: Dict[int, Any]) -> bool:
    """
    Return True if a non-empty MakerNote block is present.

    MakerNote (tag 37500) is a camera-manufacturer private blob written by
    camera firmware to store proprietary settings (AF data, lens corrections,
    custom curves …).  No AI generator embeds this tag.
    """
    val = raw.get(_TAG_MAKERNOTE)
    if val is None:
        return False
    if isinstance(val, (bytes, bytearray)):
        return len(val) > 0
    return bool(val)


def _check_gps(img: Image.Image) -> bool:
    """
    Return True if GPS latitude and longitude are embedded in the EXIF GPS IFD.

    Uses PIL's ``get_ifd(0x8825)`` to read the GPS sub-IFD directly.
    Falls back to checking whether the GPS IFD pointer tag (34853) is present
    in IFD0, which is sufficient evidence that GPS data was written.
    """
    try:
        gps_ifd = img.getexif().get_ifd(0x8825)
        # GPSLatitude=2, GPSLongitude=4 — both must be present
        if gps_ifd and 2 in gps_ifd and 4 in gps_ifd:
            return True
    except Exception:
        pass
    # Fallback: GPS IFD pointer tag present in IFD0 is good enough evidence
    try:
        return bool(dict(img.getexif()).get(_TAG_GPS_IFD))
    except Exception:
        return False


def _check_thumbnail(img: Image.Image) -> bool:
    """
    Return True if an embedded JPEG thumbnail is present in EXIF IFD1.

    Camera firmware automatically generates a thumbnail and stores it in the
    EXIF IFD1 block; AI generators do not.  Detection strategy (two-tier):

    1. Try PIL's ``get_ifd(0x0001)`` and look for JPEGInterchangeFormat (513).
    2. Fallback: scan the raw EXIF bytes for a JPEG SOI marker (FF D8) after
       the 6-byte "Exif\\x00\\x00" header — reliable because thumbnails are
       always stored as JPEG within the EXIF blob.
    """
    try:
        ifd1 = img.getexif().get_ifd(0x0001)
        if isinstance(ifd1, dict) and (513 in ifd1 or 514 in ifd1):
            return True
    except Exception:
        pass
    # Fallback: look for JPEG SOI in raw EXIF bytes
    try:
        exif_bytes = img.info.get("exif") or b""
        # Skip "Exif\x00\x00" (6 bytes), then search for JPEG thumbnail SOI
        return b"\xff\xd8" in exif_bytes[6:]
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Confidence band estimation
# ---------------------------------------------------------------------------

def _compute_metadata_band(
    has_exif:      bool,
    has_camera:    bool,
    has_makernote: bool,
    has_gps:       bool,
    has_thumbnail: bool,
) -> float:
    """
    Estimate ± uncertainty on the metadata score as a fraction of [0, 1].

    More strong authenticity signals → narrower band (higher confidence).

    Typical values
    --------------
      0.25  no EXIF at all       (almost no signal)
      0.20  EXIF present, sparse (only make / model)
      0.15  one positive signal  (e.g. camera pair present)
      0.12  two positive signals
      0.08  three or more        (camera + MakerNote + GPS / thumbnail)
    """
    if not has_exif:
        return _META_BAND_NO_EXIF
    signal_count = sum([has_camera, has_makernote, has_gps, has_thumbnail])
    if signal_count >= 3:
        return _META_BAND_THREE_OR_MORE
    if signal_count >= 2:
        return _META_BAND_TWO_SIGNALS
    if signal_count >= 1:
        return _META_BAND_ONE_SIGNAL
    return _META_BAND_ZERO_SIGNALS


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

    # Signal-richness tracking — used to compute confidence_band at the end
    has_camera    = False
    had_makernote = False
    had_gps       = False
    had_thumbnail = False
    # Editing-signal accumulator — isolated from authenticity bonuses so that
    # image_scoring can use it directly in the editing_likelihood formula.
    edit_penalty    = 0.0
    # Early-exit signal — set to the generator name when AI software is detected
    # in metadata; the pipeline can then skip the AI classifier entirely.
    detected_ai_gen = ""

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        return MetadataResult(
            score=0.5,
            flags=[f"Could not open image: {exc}"],
        )

    # Detect format early — used for format-aware EXIF rules
    fmt = (img.format or "UNKNOWN").upper()

    raw_exif = _extract_raw_exif(img)
    raw_human = _humanise_exif(raw_exif)

    # ── ICC colour profile ──────────────────────────────────────────────────
    icc = img.info.get("icc_profile")
    if icc:
        raw_human["ICC_profile_bytes"] = len(icc)

    # ── Missing EXIF  →  -30 for JPEG/TIFF, exempt for PNG/GIF/BMP ──────────
    if not raw_exif:
        if fmt in _EXIF_OPTIONAL_FORMATS:
            flags.append(
                f"No EXIF metadata ({fmt} format — EXIF is optional, not penalised)"
            )
            # No score penalty for formats where EXIF is rarely embedded
        else:
            flags.append("No EXIF metadata found (common in AI-generated images)")
            score        -= _META_PENALTY_NO_EXIF
            edit_penalty += _META_PENALTY_NO_EXIF
    else:
        # ── Camera make / model  →  -20 if both missing, -10 if model only ──
        has_make  = bool(raw_exif.get(_TAG_MAKE))
        has_model = bool(raw_exif.get(_TAG_MODEL))
        has_camera = has_make and has_model
        if not has_make and not has_model:
            flags.append("No camera make or model in EXIF")
            score -= _META_PENALTY_NO_CAMERA    # camera model mismatch → -20
        elif not has_model:
            flags.append("Camera model missing from EXIF")
            score -= _META_PENALTY_MODEL_ONLY    # partial deduction

        # ── Editing-software fingerprint ─────────────────────────────────────
        # AI generators → -40  (strong forgery signal)
        # Photo editors → -15  (editing is common; low-confidence signal)
        sw_found, sw_name, sw_category = _check_editing_software(raw_exif)
        if sw_found:
            if sw_category == "ai_generator":
                flags.append(
                    f"AI generation software detected: '{sw_name}'"
                )
                score           -= _META_PENALTY_AI_GEN
                edit_penalty    += _META_PENALTY_AI_GEN
                detected_ai_gen  = sw_name   # enables early-exit in the pipeline
            else:  # photo_editor
                flags.append(
                    f"Photo editing software detected: '{sw_name}'"
                )
                score        -= _META_PENALTY_PHOTO_EDITOR
                edit_penalty += _META_PENALTY_PHOTO_EDITOR

        # ── Timestamp checks  →  -10 for any anomaly ────────────────────────
        ts_flags, is_future, is_inconsistent = _check_timestamps(raw_exif)
        flags.extend(ts_flags)
        if is_future or is_inconsistent:
            score        -= _META_PENALTY_TIMESTAMP    # timestamp anomalies → -10
            edit_penalty += _META_PENALTY_TIMESTAMP

        # ── Valid camera pipeline bonus  →  +20 ─────────────────────────────
        # All four conditions must hold: make+model present, no editing sw,
        # no timestamp anomalies.
        if has_camera and not sw_found and not is_future and not is_inconsistent:
            score += _META_BONUS_CAMERA_PIPELINE

        # ── Positive authenticity signals ────────────────────────────────────
        # MakerNote →  +15  (camera firmware private blob, never in AI output)
        had_makernote = _check_makernote(raw_exif)
        if had_makernote:
            score += _META_BONUS_MAKERNOTE
            flags.append(
                "MakerNote present (camera-proprietary metadata — strong authenticity signal)"
            )

        # GPS coordinates →  +10  (only embedded by real camera hardware)
        had_gps = _check_gps(img)
        if had_gps:
            score += _META_BONUS_GPS
            flags.append("GPS coordinates embedded (location data from camera sensor)")

        # EXIF thumbnail →  +10  (written by camera firmware during capture)
        had_thumbnail = _check_thumbnail(img)
        if had_thumbnail:
            score += _META_BONUS_THUMBNAIL
            flags.append(
                "EXIF thumbnail present (camera-generated preview — authenticity indicator)"
            )

    # ── Dimension check (flag + score penalty for AI-typical dimensions) ────
    dim_suspicious, dim_msg = _check_dimensions(img)
    if dim_suspicious:
        flags.append(dim_msg)
        score        -= _META_PENALTY_AI_DIMENSIONS
        edit_penalty += _META_PENALTY_AI_DIMENSIONS

    # ── Format / size metadata ───────────────────────────────────────────────
    raw_human["_format"] = fmt
    raw_human["_size"]   = f"{img.width}×{img.height}"
    raw_human["_mode"]   = img.mode

    # ── Normalise to [0, 100] then return as [0.0, 1.0] ────────────────────
    score = max(0.0, min(100.0, score))
    confidence_band = _compute_metadata_band(
        has_exif      = bool(raw_exif),
        has_camera    = has_camera,
        has_makernote = had_makernote,
        has_gps       = had_gps,
        has_thumbnail = had_thumbnail,
    )
    return MetadataResult(
        score                 = score / 100.0,
        flags                 = flags,
        raw_metadata          = raw_human,
        confidence_band       = confidence_band,
        editing_penalty       = min(100.0, edit_penalty),
        detected_ai_generator = detected_ai_gen,
    )
