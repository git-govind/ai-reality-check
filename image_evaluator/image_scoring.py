"""
image_scoring.py
-----------------
Scoring engine for the Image Authenticity Evaluator pipeline.

Aggregates results from all pipeline steps into a single
:class:`ImageEvaluationReport` using the exact weighted formula below.

Exact formula (spec §1)
-----------------------
  authenticity_score =
      0.20 * metadata_score          +
      0.25 * pixel_score             +
      0.35 * (1 - ai_generated_prob) +   (component already × 100 at the end)
      0.10 * consistency_score       +   (skipped → weight redistributed)
      0.10 * reverse_search_score        (skipped → weight redistributed)

All per-step scores are carried as floats in [0, 1] internally;
the final authenticity_score is scaled to [0, 100].

Derived metrics (spec §1)
-------------------------
  ai_likelihood       = ai_generated_prob × 100
  editing_likelihood  = 0.6 × ela_badness  +  0.4 × metadata_edit_penalty
                        ela_badness           = min(100, ela_p95 / 30 × 100)
                        metadata_edit_penalty = accumulated metadata editing signals
                        (missing EXIF +30, AI generator +40, photo editor +15,
                         timestamp anomaly +10; capped at 100)

Reverse-search score mapping (spec §2)
---------------------------------------
  ReverseSearchResult.similarity ≥ 0.90  →  score = 1.00  (exact match)
  ReverseSearchResult.similarity ≥ 0.70  →  score = 0.80  (high similarity)
  ReverseSearchResult.similarity ≥ 0.30  →  score = 0.50  (partial match)
  otherwise                              →  score = 0.00  (no match)

Weight redistribution
---------------------
  When optional steps are skipped (ran=False), their weight is distributed
  proportionally among the remaining active components so the final score
  is always scaled to a 0–100 range.

Grade thresholds
----------------
  A  ≥ 80   almost certainly authentic
  B  ≥ 65   probably authentic
  C  ≥ 50   uncertain
  D  ≥ 35   probably manipulated / AI-generated
  F  <  35  almost certainly manipulated / AI-generated
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image

from .datatypes import (
    AIArtifactResult,
    ConsistencyResult,
    ImageEvaluationReport,
    MetadataResult,
    PixelForensicsResult,
    ReverseSearchResult,
)

from config_loader import get_feature, get_threshold, get_weight
from explanation_generator import generate_explanation
from utils.scoring_utils import normalize_weights

# ---------------------------------------------------------------------------
# Adaptive weight presets by image type  (loaded from config/weights.yaml)
# ---------------------------------------------------------------------------
#
# photo        – real camera capture: default weights
# illustration – drawn / painted / rendered: pixel artifacts carry more signal,
#                AI classifier less reliable on stylised artwork
# screenshot   – desktop/app captures: no camera sensor noise, metadata-heavy,
#                AI detection almost meaningless

def _load_weights_by_type() -> dict[str, dict[str, float]]:
    _types = ["photo", "illustration", "screenshot"]
    _keys  = ["metadata", "pixel", "ai", "consistency", "reverse"]
    return {
        t: {k: get_weight(f"image.scoring_weights.{t}.{k}") for k in _keys}
        for t in _types
    }

_WEIGHTS_BY_TYPE: dict[str, dict[str, float]] = _load_weights_by_type()

# Grade thresholds (loaded from config/thresholds.yaml)
_IMG_GRADE_A = get_threshold("image.grade.a")  # 80
_IMG_GRADE_B = get_threshold("image.grade.b")  # 65
_IMG_GRADE_C = get_threshold("image.grade.c")  # 50
_IMG_GRADE_D = get_threshold("image.grade.d")  # 35

# AI band adjustments for OOD image types
_AI_BAND_ADJ_ILLUSTRATION = get_threshold("image.ai_band_adjustment.illustration")
_AI_BAND_ADJ_SCREENSHOT   = get_threshold("image.ai_band_adjustment.screenshot")

# Pixel-derived AI signal blend parameters.
# The pixel signal (noise block CV) is gated on the ML classifier's own ai_prob
# to prevent pixel noise statistics from overriding a clear ML "not-AI" verdict.
_PIXEL_AI_BLEND_CAP      = get_threshold("image.ai_pixel_blend.max_contribution")  # 0.50
_PIXEL_AI_GATE_LOW       = get_threshold("image.ai_pixel_blend.gate_low")           # 0.10
_PIXEL_AI_GATE_HIGH      = get_threshold("image.ai_pixel_blend.gate_high")          # 0.30
# Minimum ai_prob for the "AI detector" entry to appear in top_signals.
# Below this threshold a low AI probability is an AUTHENTICITY signal, not a concern.
_TOP_SIGNALS_AI_MIN_PROB = get_threshold("image.top_signals_ai_min_prob")           # 0.25

# ---------------------------------------------------------------------------
# Image-type detection
# ---------------------------------------------------------------------------

def detect_image_type(image_bytes: bytes) -> str:
    """
    Classify the image as ``"photo"``, ``"illustration"``, or ``"screenshot"``
    using three fast pixel-level signals.

    Signals
    -------
    flat_ratio
        Fraction of adjacent identical pixel pairs (horizontal + vertical).
        High in screenshots (solid UI fills) and vector artwork.
    unique_ratio
        Unique colours / total pixels.  Screenshots have very low ratios
        (few distinct palette entries); photos have very high ratios.
    noise_std
        Std-dev of the Gaussian high-pass residual on the luminance channel.
        Camera sensor noise produces a measurable floor (> ~2.5 DN);
        AI renders and screenshots tend to be much smoother.

    Decision rules
    --------------
    1. flat_ratio > 0.30  OR  unique_ratio < 0.10  →  screenshot
    2. flat_ratio < 0.05  AND noise_std > 2.5       →  photo
    3. otherwise                                     →  illustration
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "photo"  # safe default

    arr = np.array(img, dtype=np.float32)  # (H, W, 3)
    H, W, _ = arr.shape

    # ── flat_ratio ────────────────────────────────────────────────────────
    # Count adjacent pairs that are identical (within rounding to uint8)
    uint8 = arr.astype(np.uint8)
    h_diff = np.any(uint8[:, 1:] != uint8[:, :-1], axis=2)  # (H, W-1)
    v_diff = np.any(uint8[1:, :] != uint8[:-1, :], axis=2)  # (H-1, W)
    total_pairs = (H * (W - 1)) + ((H - 1) * W)
    flat_pairs  = total_pairs - int(h_diff.sum()) - int(v_diff.sum())
    flat_ratio  = flat_pairs / max(total_pairs, 1)

    # ── unique_ratio ──────────────────────────────────────────────────────
    # Downsample to at most 256×256 to keep hashing fast
    thumb = img.resize((min(W, 256), min(H, 256)), Image.BILINEAR)
    t_arr = np.array(thumb, dtype=np.uint8)
    # Pack RGB into a single int32 for fast unique count
    packed = (t_arr[:, :, 0].astype(np.int32) * 65536
              + t_arr[:, :, 1].astype(np.int32) * 256
              + t_arr[:, :, 2].astype(np.int32))
    unique_ratio = len(np.unique(packed)) / packed.size

    # ── noise_std ─────────────────────────────────────────────────────────
    # Simple 3×3 box-blur via integral image (no scipy dependency)
    def _box3(a: np.ndarray) -> np.ndarray:
        """3×3 uniform blur using cumulative sums (no edge roll artifacts)."""
        padded = np.pad(a, 1, mode="edge")
        out = np.zeros_like(a)
        for di in range(3):
            for dj in range(3):
                out += padded[di : di + a.shape[0], dj : dj + a.shape[1]]
        return out / 9.0

    gray = arr.mean(axis=2)                   # (H, W) luminance
    blurred = _box3(gray)
    residual = gray - blurred
    noise_std = float(np.std(residual))

    # ── Decision ──────────────────────────────────────────────────────────
    if flat_ratio > 0.30 or unique_ratio < 0.10:
        return "screenshot"
    if flat_ratio < 0.05 and noise_std > 2.5:
        return "photo"
    return "illustration"


# ---------------------------------------------------------------------------
# Top-signal helpers
# ---------------------------------------------------------------------------

# Metadata-flag substrings that indicate *positive* authenticity evidence.
# These are excluded from top_signals so the list focuses on suspicious signals.
_SKIP_FLAG_SUBSTRINGS: tuple[str, ...] = (
    "makernote present",
    "gps coordinates embedded",
    "exif thumbnail present",
    "not penalised",          # "EXIF is optional, not penalised" on PNG
)

# Ordered list of (substring, strength) for matching metadata flag text.
# First match wins; unmatched flags get the generic default (0.35).
_FLAG_STRENGTH_RULES: list[tuple[str, float]] = [
    ("ai generation software detected",  0.90),
    ("no exif metadata found",           0.70),
    ("power-of-two dimensions",          0.65),
    ("no camera make or model",          0.55),
    ("photo editing software detected",  0.55),
    ("camera model missing",             0.40),
    ("timestamp",                        0.40),
    ("aspect ratio",                     0.35),
]


def _top_signals(
    metadata:       MetadataResult,
    pixel:          PixelForensicsResult,
    ai_artifact:    AIArtifactResult,
    consistency:    ConsistencyResult,
    reverse_search: ReverseSearchResult,
    n: int = 3,
) -> list[str]:
    """
    Return the *n* strongest suspicious signals driving the evaluation verdict.

    Candidates are collected from every pipeline result, ranked by estimated
    strength (0–1), deduplicated, then the top *n* are returned as
    human-readable strings suitable for direct display.

    Positive authenticity signals (MakerNote, GPS, EXIF thumbnail) are
    intentionally excluded — they lower the suspicion score but are not
    themselves "suspicious signals" worth highlighting.
    """
    candidates: list[tuple[float, str]] = []   # (strength, label)

    # ── AI artifact probability ───────────────────────────────────────────────
    # Only list when raw ML probability is high enough to constitute a concern.
    # Values below _TOP_SIGNALS_AI_MIN_PROB (0.25) indicate the image is likely
    # real; listing them as a "concern" is misleading.
    if ai_artifact.ai_prob >= _TOP_SIGNALS_AI_MIN_PROB:
        candidates.append((
            ai_artifact.ai_prob,
            f"AI detector: {ai_artifact.ai_prob:.2f} probability",
        ))

    # ── Metadata: definitive AI generator (early-exit path) ──────────────────
    if metadata.detected_ai_generator:
        candidates.append((
            1.0,
            f"AI generator software confirmed: '{metadata.detected_ai_generator}'",
        ))

    # ── Metadata flags ────────────────────────────────────────────────────────
    ai_gen_flag_covered = bool(metadata.detected_ai_generator)
    for flag in metadata.flags:
        fl = flag.lower()
        # Skip positive authenticity indicators
        if any(sub in fl for sub in _SKIP_FLAG_SUBSTRINGS):
            continue
        # Avoid duplicating the ai_generator entry already added above
        if ai_gen_flag_covered and "ai generation software detected" in fl:
            continue
        strength = 0.35  # generic default for unrecognised flags
        for pattern, s in _FLAG_STRENGTH_RULES:
            if pattern in fl:
                strength = s
                break
        candidates.append((strength, flag))

    # ── Pixel: ELA compression artefacts ─────────────────────────────────────
    if pixel.ela_max_diff is not None:
        ela_bad = float(min(100.0, pixel.ela_max_diff / get_threshold("image.ela_p95_norm_cap") * 100.0))
        if ela_bad > 10.0:
            candidates.append((
                ela_bad / 100.0,
                f"ELA compression artefacts: {ela_bad:.0f}/100",
            ))

    # ── Pixel: JPEG Ghost ────────────────────────────────────────────────────
    if pixel.ghost_score is not None and pixel.ghost_score > 10.0:
        candidates.append((
            min(1.0, pixel.ghost_score / 100.0),
            f"JPEG ghost inconsistency score: {pixel.ghost_score:.0f}/100",
        ))

    # ── Pixel: noise uniformity  (low CV → unnaturally flat → AI signal) ─────
    # noise_uniformity is the CV of per-patch noise stds across colour channels.
    # Very low (~0) means all channels share identically smooth noise → AI fingerprint.
    if pixel.noise_uniformity is not None and pixel.noise_uniformity < 0.20:
        strength = max(0.0, 1.0 - pixel.noise_uniformity / 0.20)
        candidates.append((
            strength,
            f"Uniform color-channel noise correlation (CV={pixel.noise_uniformity:.2f})",
        ))

    # ── Pixel: noise block consistency  (high CV → spatially inhomogeneous) ──
    # noise_block_consistency is the CV of 16-block residual stds.
    # High CV → AI model produces heterogeneous noise (smooth + textured regions).
    if pixel.noise_block_consistency is not None and pixel.noise_block_consistency > 0.40:
        strength = min(1.0, pixel.noise_block_consistency / 0.80)
        candidates.append((
            strength,
            f"Spatially inconsistent noise pattern (block CV={pixel.noise_block_consistency:.2f})",
        ))

    # ── Pixel: FFT spectral peaks ─────────────────────────────────────────────
    if pixel.fft_peak_ratio is not None:
        fft_bad = float(min(100.0, pixel.fft_peak_ratio / get_threshold("image.fft_norm_cap") * 100.0))
        if fft_bad > 20.0:
            candidates.append((
                fft_bad / 100.0,
                f"Periodic FFT spectral peaks (ratio={pixel.fft_peak_ratio:.2f})",
            ))

    # ── Consistency ───────────────────────────────────────────────────────────
    if consistency.ran and consistency.score < 0.50:
        candidates.append((
            1.0 - consistency.score,
            f"Image–text inconsistency detected (score={consistency.score:.2f})",
        ))

    # ── Deduplicate, sort descending by strength, return top n ───────────────
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    unique: list[str] = []
    for _, label in candidates:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return unique[:n]


# ---------------------------------------------------------------------------
# Reverse-search tiered score (spec §2)
# ---------------------------------------------------------------------------

def _reverse_score(result: ReverseSearchResult) -> float:
    """
    Map ReverseSearchResult to an authenticity score in [0, 1].

    Tier mapping (spec §2):
      similarity ≥ 0.90  →  1.00  exact match       (→ 100 pts)
      similarity ≥ 0.70  →  0.80  high similarity   (→  80 pts)
      similarity ≥ 0.30  →  0.50  partial match     (→  50 pts)
      otherwise          →  0.00  no match           (→   0 pts)

    When the step was skipped (ran=False) returns None so the caller
    knows to exclude this component from the weighted average.
    """
    if not result.ran:
        return None  # will be redistributed

    if not result.found:
        return 0.0   # no match

    sim = result.similarity
    if sim >= 0.90:
        return 1.00  # exact match
    if sim >= 0.70:
        return 0.80  # high similarity
    if sim >= 0.30:
        return 0.50  # partial match
    return 0.00


# ---------------------------------------------------------------------------
# Pixel-forensics-derived AI signal  (supplements the ML classifier)
# ---------------------------------------------------------------------------

def _pixel_ai_signal(pixel: PixelForensicsResult) -> float:
    """
    Derive a supplementary AI-likelihood signal from pixel forensics.

    Uses **only** noise_block_consistency (16-block noise CV).

    FFT spectral peaks are intentionally excluded: JPEG 8×8 block-DCT
    quantisation routinely creates periodic spectral spikes in authentic
    camera photos (often with ratio > 0.25), making FFT an unreliable
    AI-vs-real discriminator at the ai_likelihood level.  FFT is already
    captured in pixel_score for detecting general editing artefacts.

    The noise_block_consistency signal (spatial CV of 16-block sensor noise)
    is more AI-specific: real camera PRNU is spatially stationary, while
    diffusion / GAN models synthesise textures region-by-region, producing
    heterogeneous noise levels across the frame.

    Returns a value in [0, 1]: higher means more AI-like based on pixel evidence.
    """
    block_cv    = pixel.noise_block_consistency or 0.0
    consist_sig = float(np.clip((block_cv - 0.15) / 0.40, 0.0, 1.0))
    return consist_sig


# ---------------------------------------------------------------------------
# Editing likelihood estimation  (formula-based, spec §3)
# ---------------------------------------------------------------------------
#
# editing_likelihood = clip(0.6 × ela_badness  +  0.4 × metadata_edit_penalty, 0, 100)
#
# ela_badness            – ELA 95th-pct residual mapped to [0, 100]:
#                          0 at p95=0 (no compression artefacts), 100 at p95≥30
#                          Same scale as _ela_badness() in pixel_forensics.py.
# metadata_edit_penalty  – editing-specific metadata signals, [0, 100]:
#                          +30 missing EXIF on JPEG/TIFF
#                          +40 AI-generator software tag
#                          +15 photo editing software tag
#                          +10 timestamp anomaly
#                          (returned as MetadataResult.editing_penalty)
#
# The 0.60 / 0.40 split reflects that ELA is the primary forensic signal for
# editing; metadata provides corroborating evidence but is easily stripped.


def _compute_editing_likelihood(
    meta:  MetadataResult,
    pixel: PixelForensicsResult,
) -> float:
    """
    Formula-based editing likelihood (0–100).

    editing_likelihood = 0.6 × ela_badness  +  0.4 × metadata_edit_penalty

    Both inputs are on the [0, 100] scale.
    """
    ela_bad   = float(min(100.0, (pixel.ela_max_diff or 0.0) / get_threshold("image.ela_p95_norm_cap") * 100.0))
    meta_edit = float(meta.editing_penalty)   # already clamped to [0, 100]
    return float(np.clip(
        get_weight("image.editing_likelihood.ela_weight")      * ela_bad
        + get_weight("image.editing_likelihood.metadata_weight") * meta_edit,
        0.0, 100.0,
    ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    metadata:       MetadataResult,
    pixel:          PixelForensicsResult,
    ai_artifact:    AIArtifactResult,
    consistency:    Optional[ConsistencyResult]    = None,
    reverse_search: Optional[ReverseSearchResult] = None,
    image_type:     str                            = "photo",
) -> ImageEvaluationReport:
    """
    Aggregate pipeline step results into an :class:`ImageEvaluationReport`.

    Implements the exact scoring formula from spec §1, with adaptive weights
    based on *image_type* (``"photo"``, ``"illustration"``, or ``"screenshot"``).

    Parameters
    ----------
    metadata       : MetadataResult
    pixel          : PixelForensicsResult
    ai_artifact    : AIArtifactResult
    consistency    : ConsistencyResult or None
    reverse_search : ReverseSearchResult or None
    image_type     : str  – ``"photo"`` | ``"illustration"`` | ``"screenshot"``

    Returns
    -------
    ImageEvaluationReport
    """
    # ── Defaults for optional steps ──────────────────────────────────────────
    if consistency is None:
        consistency = ConsistencyResult(score=0.5, ran=False)
    if reverse_search is None:
        reverse_search = ReverseSearchResult(ran=False)

    # ── Select base weights for image type ───────────────────────────────────
    type_key = image_type if image_type in _WEIGHTS_BY_TYPE else "photo"
    base_w   = _WEIGHTS_BY_TYPE[type_key]

    # ── Pixel-derived AI signal boost (ML-gated) ──────────────────────────────
    # Noise block inconsistency is a hallmark of AI generation; boost ai_prob
    # when it is elevated.  The boost is gated on the ML classifier's own
    # ai_prob so that a clear "not-AI" ML verdict (ai_prob < gate_low) is NOT
    # overridden by pixel statistics — this prevents JPEG compression artefacts
    # (which also elevate pixel signals) from producing false positives on
    # authentic camera photos.
    #   gate = 0  when ai_prob < gate_low  → trust the ML verdict, no pixel boost
    #   gate = 1  when ai_prob > gate_high → ML is uncertain, apply full boost
    _ml_gate = float(np.clip(
        (ai_artifact.ai_prob - _PIXEL_AI_GATE_LOW)
        / (_PIXEL_AI_GATE_HIGH - _PIXEL_AI_GATE_LOW),
        0.0, 1.0,
    ))
    _pixel_sig = _pixel_ai_signal(pixel)
    _effective_ai_prob = float(np.clip(
        ai_artifact.ai_prob
        + _ml_gate * _pixel_sig * _PIXEL_AI_BLEND_CAP * (1.0 - ai_artifact.ai_prob),
        0.0, 1.0,
    ))

    # ── Fixed components (always active) ─────────────────────────────────────
    weights: dict[str, float] = {
        "metadata": base_w["metadata"],
        "pixel":    base_w["pixel"],
        "ai":       base_w["ai"],
    }
    scores: dict[str, float] = {
        "metadata": metadata.score,                 # [0, 1]
        "pixel":    pixel.score,                    # [0, 1]
        "ai":       1.0 - _effective_ai_prob,       # [0, 1]
    }

    # ── Optional: consistency ────────────────────────────────────────────────
    if consistency.ran:
        weights["consistency"] = base_w["consistency"]
        scores["consistency"]  = consistency.score  # [0, 1]

    # ── Optional: reverse search ─────────────────────────────────────────────
    rev = _reverse_score(reverse_search)
    if rev is not None:
        weights["reverse"] = base_w["reverse"]
        scores["reverse"]  = rev                    # [0, 1]

    # ── Normalise weights to sum to 1.0 ──────────────────────────────────────
    norm_weights = normalize_weights(weights)

    # ── Exact spec formula: weighted sum × 100 ───────────────────────────────
    raw_score          = sum(scores[k] * norm_weights[k] for k in norm_weights)
    authenticity_score = float(raw_score * 100.0)

    # ── Derived metrics ──────────────────────────────────────────────────────
    ai_likelihood      = round(_effective_ai_prob * 100.0, 1)
    editing_likelihood = _compute_editing_likelihood(metadata, pixel)

    # ── Per-component confidence bands ───────────────────────────────────────
    # Adjust AI band upward when the AI classifier is less reliable for the
    # detected image type (illustrations / screenshots are out-of-distribution
    # for models trained on photograph vs diffusion-model datasets).
    ai_band = ai_artifact.confidence_band
    if type_key == "illustration":
        ai_band = min(1.0, ai_band + _AI_BAND_ADJ_ILLUSTRATION)
    elif type_key == "screenshot":
        ai_band = min(1.0, ai_band + _AI_BAND_ADJ_SCREENSHOT)

    bands: dict[str, float] = {
        "metadata": metadata.confidence_band,
        "pixel":    pixel.confidence_band,
        "ai":       ai_band,
    }
    if consistency.ran:
        bands["consistency"] = consistency.confidence_band
    if rev is not None:
        bands["reverse"] = 0.15   # fixed: reverse-search is moderately reliable

    # Propagate: authenticity_band = 100 × Σ(w_i × band_i)
    # This is a linear error-propagation approximation; each component's
    # uncertainty is weighted by the same normalised weight used in scoring.
    authenticity_band       = round(100.0 * sum(norm_weights[k] * bands[k] for k in norm_weights), 1)
    ai_likelihood_band      = round(ai_band * 100.0, 1)
    # editing_likelihood_band mirrors the formula weights 0.6/0.4:
    #   ELA     → pixel.confidence_band  (already captures format + size uncertainty)
    #   metadata → metadata.confidence_band (captures EXIF richness uncertainty)
    editing_likelihood_band = round(
        100.0 * (0.6 * pixel.confidence_band + 0.4 * metadata.confidence_band), 1
    )

    # ── Top signals ───────────────────────────────────────────────────────────
    top_sigs = _top_signals(metadata, pixel, ai_artifact, consistency, reverse_search)

    # ── Grade ─────────────────────────────────────────────────────────────────
    s = authenticity_score
    if s >= _IMG_GRADE_A:
        grade, summary = "A", "Image appears authentic — no significant anomalies detected."
    elif s >= _IMG_GRADE_B:
        grade, summary = "B", "Image is probably authentic with minor anomalies."
    elif s >= _IMG_GRADE_C:
        grade, summary = "C", "Image authenticity is uncertain — further review recommended."
    elif s >= _IMG_GRADE_D:
        grade, summary = "D", "Image shows significant manipulation or AI-generation signals."
    else:
        grade, summary = "F", "Image is very likely AI-generated or heavily manipulated."

    # ── Evidence bundle (spec §5) ─────────────────────────────────────────────
    evidence = {
        "image_type":               type_key,
        "authenticity_band":        authenticity_band,
        "ai_likelihood_band":       ai_likelihood_band,
        "editing_likelihood_band":  editing_likelihood_band,
        "metadata_flags":           metadata.flags,
        "metadata_editing_penalty": metadata.editing_penalty,
        "pixel_artifacts":          pixel.artifacts,
        "pixel_ela_p95":            pixel.ela_max_diff,
        "pixel_ghost_score":        pixel.ghost_score,
        "pixel_noise_cv":           pixel.noise_uniformity,
        "pixel_noise_block_cv":     pixel.noise_block_consistency,
        "pixel_fft_ratio":          pixel.fft_peak_ratio,
        "ai_artifact_features":     ai_artifact.features,
        "ai_artifact_features_dict": ai_artifact.features_dict,
        "ai_method":                ai_artifact.method,
        "ai_confidence_band":       ai_artifact.confidence_band,
        "ai_feature_vector":        ai_artifact.feature_vector,
        "ood_warning":              ai_artifact.ood_warning,
        "consistency_issues":       consistency.issues if consistency.ran else [],
        "consistency_ran":          consistency.ran,
        "reverse_search_hits":      reverse_search.source_urls,
        "reverse_search_ran":       reverse_search.ran,
        "reverse_search_found":     reverse_search.found,
        "reverse_search_error":     reverse_search.error,
        "component_scores": {
            k: round(scores[k] * 100, 1) for k in scores
        },
        "component_weights": {
            k: round(norm_weights[k], 3) for k in norm_weights
        },
    }

    report = ImageEvaluationReport(
        authenticity_score      = round(authenticity_score, 1),
        authenticity_band       = authenticity_band,
        ai_likelihood           = round(ai_likelihood,      1),
        ai_likelihood_band      = ai_likelihood_band,
        editing_likelihood      = round(editing_likelihood,  1),
        editing_likelihood_band = editing_likelihood_band,
        grade                   = grade,
        summary                 = summary,
        evidence                = evidence,
        top_signals             = top_sigs,
    )
    report.explanation = generate_explanation(report)

    if get_feature("debug"):
        report.metadata["debug"] = {
            "intermediate_scores": {
                "metadata_score":       round(metadata.score * 100, 1),
                "pixel_score":          round(pixel.score * 100, 1),
                "ai_prob_raw_pct":      round(ai_artifact.ai_prob * 100, 1),
                "pixel_ai_signal":      round(_pixel_ai_signal(pixel), 3),
                "effective_ai_prob_pct": round(_effective_ai_prob * 100, 1),
                "ai_score":             round((1.0 - _effective_ai_prob) * 100, 1),
                "consistency_score":    round(consistency.score * 100, 1) if consistency.ran else None,
                "reverse_score":        round((rev or 0) * 100, 1) if rev is not None else None,
            },
            "weights": {
                "base":             dict(base_w),
                "active":           dict(weights),
                "normalized":       {k: round(v, 4) for k, v in norm_weights.items()},
                "total_before_norm": round(sum(weights.values()), 4),
            },
            "raw_evidence": {
                "metadata_raw":            metadata.raw_metadata,
                "detected_ai_generator":   metadata.detected_ai_generator,
                "editing_penalty":         metadata.editing_penalty,
                "ela_max_diff":            pixel.ela_max_diff,
                "fft_peak_ratio":          pixel.fft_peak_ratio,
                "noise_uniformity":        pixel.noise_uniformity,
                "noise_block_consistency": pixel.noise_block_consistency,
                "ghost_score":             pixel.ghost_score,
                "ai_features_dict":        ai_artifact.features_dict,
                "ai_feature_vector":       ai_artifact.feature_vector,
                "ai_method":               ai_artifact.method,
            },
            "thresholds_used": {
                "grades": {
                    "A": _IMG_GRADE_A,
                    "B": _IMG_GRADE_B,
                    "C": _IMG_GRADE_C,
                    "D": _IMG_GRADE_D,
                },
                "base_weights_for_type": dict(base_w),
                "ai_band_adjustments": {
                    "illustration": _AI_BAND_ADJ_ILLUSTRATION,
                    "screenshot":   _AI_BAND_ADJ_SCREENSHOT,
                },
            },
            "module_decisions": {
                "image_type":              type_key,
                "consistency_ran":         consistency.ran,
                "reverse_search_ran":      reverse_search.ran,
                "reverse_in_scoring":      rev is not None,
                "early_exit_ai_generator": bool(metadata.detected_ai_generator),
                "ood_warning_triggered":   bool(ai_artifact.ood_warning),
                "ai_method":               ai_artifact.method,
                "ai_confidence_band":      round(ai_artifact.confidence_band, 3),
            },
        }

    return report
