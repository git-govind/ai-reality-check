"""
evaluate_image.py
------------------
Public API entry point for the Image Authenticity Evaluator.

Usage
-----
    from image_evaluator import evaluate_image

    with open("photo.jpg", "rb") as f:
        image_bytes = f.read()

    report = evaluate_image(image_bytes)
    print(report.grade, report.authenticity_score)
    print(report.summary)

    # With optional caption for consistency check:
    report = evaluate_image(image_bytes, caption="A red car parked outside a museum")

Pipeline steps (all isolated from the text-evaluation engine)
-------------------------------------------------------------
  1. metadata_checker     — EXIF / XMP / editing-software fingerprints
  2. pixel_forensics      — ELA, noise, FFT, JPEG quant tables
  3. ai_artifact_classifier — heuristic + optional CLIP-based AI detector
  4. image_text_consistency — optional; only runs when caption is provided
  5. reverse_image_search   — optional; only runs when an API key is configured
  ── image_scoring.aggregate() ────────────────────────────────────────────────
  → ImageEvaluationReport
"""

from __future__ import annotations

from typing import Optional

from . import (
    ai_artifact_classifier,
    image_scoring,
    image_text_consistency,
    image_watermark_detector,
    metadata_checker,
    pixel_forensics,
    reverse_image_search,
)
from .datatypes import ImageEvaluationReport
from profiler import get_timings, reset
from utils.logging_utils import timer
from config_loader import get_feature


def evaluate_image(
    image_bytes: bytes,
    caption: Optional[str] = None,
) -> ImageEvaluationReport:
    """
    Run the full image authenticity evaluation pipeline.

    This function is completely self-contained and does **not** call any
    component from the text-evaluation pipeline (factual_checker,
    consistency_checker, etc.).

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the image file (JPEG, PNG, WEBP, TIFF, BMP, GIF …).
        The caller is responsible for reading the file; this function never
        opens files from disk.

    caption : str or None, optional
        An optional textual description or prompt associated with the image.
        When provided, Step 4 (image_text_consistency) is executed.

    Returns
    -------
    ImageEvaluationReport
        All scores are in the 0–100 range.  See :mod:`.datatypes` for the
        full field reference.

    Notes
    -----
    Steps 4 and 5 are optional and degrade gracefully:
      • Step 4 is skipped when *caption* is None or empty.
      • Step 5 is skipped when no reverse-image-search API key is set.

    When a step is skipped its weight is redistributed proportionally among
    the remaining steps so the final score is always on the 0–100 scale.
    """
    reset()

    # ── Step 1: Metadata ────────────────────────────────────────────────────
    with timer("metadata"):
        meta_result = metadata_checker.run(image_bytes)

    # ── Step 1b: Watermark detection (optional, feature-flagged) ────────────
    # Runs after metadata so the already-extracted MetadataResult can be reused
    # (avoids a second EXIF parse).  Placed before pixel forensics so the scoring
    # engine can short-circuit further analysis when a definitive watermark is found.
    watermark_result = None
    if get_feature("image.enable_watermark_detection"):
        with timer("watermark"):
            watermark_result = image_watermark_detector.detect_watermarks(
                image_bytes, metadata_result=meta_result
            )

    # ── Step 2: Pixel forensics ─────────────────────────────────────────────
    with timer("pixel_forensics"):
        pixel_result = pixel_forensics.run(image_bytes)

    # ── Step 3: AI artifact classification ──────────────────────────────────
    with timer("ai_artifact"):
        ai_result = ai_artifact_classifier.run(image_bytes)

    # ── Step 4: Image–text consistency (optional) ───────────────────────────
    with timer("consistency"):
        consistency_result = image_text_consistency.run(image_bytes, caption=caption)

    # ── Step 5: Reverse image search (optional) ─────────────────────────────
    with timer("reverse_search"):
        reverse_result = reverse_image_search.run(image_bytes)

    # ── Scoring ─────────────────────────────────────────────────────────────
    with timer("aggregate"):
        report = image_scoring.aggregate(
            metadata       = meta_result,
            pixel          = pixel_result,
            ai_artifact    = ai_result,
            consistency    = consistency_result,
            reverse_search = reverse_result,
            watermark      = watermark_result,
        )

    report.metadata["timings"] = get_timings()
    return report
