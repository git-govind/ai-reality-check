"""
pixel_forensics.py
-------------------
Step 2 of the Image Authenticity Evaluator pipeline.

Performs classical digital-forensics analysis on pixel data to detect
evidence of editing or synthetic generation.

Techniques
----------
  ELA  (Error Level Analysis)
      Re-compresses the image as JPEG at a reduced quality and measures
      the pixel-level difference.  Authentic regions re-compress uniformly;
      spliced or heavily edited regions retain higher residual error.

  JPEG Ghost Analysis
      Re-compresses the image at multiple quality levels [50, 60, 70, 80, 90].
      For each 32×32 patch, detects which quality level produces the minimum
      residual — the patch's 'native' compression quality.  An unmodified image
      has one dominant native quality; spliced images have patches with different
      native qualities.  Measured as normalised Shannon entropy of the quality-
      index distribution across patches (0 = uniform / pristine, 100 = maximally
      inconsistent / edited).

  Noise Residual Analysis
      Subtracts a Gaussian-blurred version to isolate sensor noise.
      Real camera images have characteristic wide-band noise with quasi-
      Gaussian statistics.  AI-generated images are often unnaturally smooth
      (very low noise) or have spatially correlated noise patterns.

  Noise Pattern Consistency  (16-block)
      Divides the Gaussian residual into a 4×4 grid of 16 equal-area blocks.
      Computes std of residual within each block then measures the coefficient
      of variation (CV) across the 16 values.  Real cameras: sensor noise is
      spatially stationary (PRNU is deterministic) → low CV.  AI generators:
      smooth regions have near-zero residual, textured regions have high residual
      → high CV.  Contributes 40% of the combined noise badness signal.

  Frequency-Domain (FFT) Artifact Scan
      Computes the 2-D power spectrum.  Periodic grid artefacts (common in
      GAN upsampling and JPEG blocking) produce prominent spectral spikes.

  JPEG Quantisation Table Check
      Inspects the raw JFIF/JPEG quantisation tables.  Some AI tools embed
      tables with unusual quality factors or all-equal entries.

Scoring formula (spec §2)
-------------------------
  pixel_score = 100 − (ela_score × 0.30 + ghost_score × 0.20
                        + noise_score × 0.25 + fft_score × 0.25)

  Each sub-score is a "badness" value in [0, 100]:
    ela_score   = min(100, ela_p95 / 30 × 100)
    ghost_score = normalised Shannon entropy of quality-index distribution × 100
    noise_score = composite of low-std (< 2.0) and high-CV (> 0.60) badness
    fft_score   = min(100, norm_peak_ratio / 0.25 × 100)

  JPEG quant suspicion applies a small additive −5 outside the formula (informational).
  Final PixelForensicsResult.score = clamped_0_100 / 100.0  →  [0.0, 1.0]

Dependencies: Pillow, NumPy (both already in requirements.txt)
"""

from __future__ import annotations

import io
import struct
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .datatypes import PixelForensicsResult

from config_loader import get_threshold, get_weight
from utils.image_utils import coeff_of_variation, jpeg_recompress, load_image_rgb


# ---------------------------------------------------------------------------
# Constants  (loaded from config/thresholds.yaml and config/weights.yaml)
# ---------------------------------------------------------------------------

_ELA_HIGH_THRESHOLD  = get_threshold("image.ela_p95_norm_cap")        # 30.0
_ELA_MED_THRESHOLD   = 15.0     # informational threshold (not configurable)
_NOISE_CV_HIGH       = get_threshold("image.noise.high_cv_threshold")  # 0.60
_NOISE_STD_LOW       = get_threshold("image.noise.low_std_threshold")  # 2.0
_FFT_PEAK_HIGH       = get_threshold("image.fft_norm_cap")             # 0.25
_NOISE_BLOCK_CV_HIGH = get_threshold("image.noise.block_cv_high")      # 0.45
_ELA_QUALITY         = 75       # JPEG quality for re-compression in ELA
_PATCH_SIZE          = 32       # patch size for local noise estimation

_GHOST_QUALITIES     = [50, 60, 70, 80, 90]   # quality levels to probe for JPEG Ghost
_GHOST_PATCH         = 32                      # patch size in pixels for ghost analysis

_N_BLOCKS            = 4                       # 4×4 = 16 blocks for noise consistency


# ---------------------------------------------------------------------------
# ELA
# ---------------------------------------------------------------------------

def _ela(img_rgb: np.ndarray, quality: int = _ELA_QUALITY) -> np.ndarray:
    """
    Return per-pixel absolute error after JPEG re-compression.

    Parameters
    ----------
    img_rgb : np.ndarray, shape (H, W, 3), dtype float32
    quality : int  JPEG compression quality (0–95)

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype float32
    """
    pil    = Image.fromarray(img_rgb.astype(np.uint8))
    recomp = jpeg_recompress(pil, quality)
    return np.abs(img_rgb.astype(np.float32) - recomp)


def _run_ela(img_rgb: np.ndarray) -> Tuple[float, List[str]]:
    """Return (ela_percentile_95, flags)."""
    try:
        diff = _ela(img_rgb)
        p95  = float(np.percentile(diff, 95))
        flags: List[str] = []
        if p95 > _ELA_HIGH_THRESHOLD:
            flags.append(
                f"ELA: high residual error (p95={p95:.1f}) — strong editing signal"
            )
        elif p95 > _ELA_MED_THRESHOLD:
            flags.append(
                f"ELA: moderate residual error (p95={p95:.1f})"
            )
        return p95, flags
    except Exception as exc:
        return 0.0, [f"ELA skipped: {exc}"]


# ---------------------------------------------------------------------------
# Noise Residual Analysis
# ---------------------------------------------------------------------------

def _gaussian_blur_numpy(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Fast separable Gaussian blur without scipy dependency."""
    radius = int(3 * sigma) + 1
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Apply along H then W for each channel
    out = arr.copy().astype(np.float32)
    for c in range(arr.shape[2] if arr.ndim == 3 else 1):
        ch = out[:, :, c] if arr.ndim == 3 else out
        ch[:] = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, ch)
        ch[:] = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 0, ch)
    return out


def _run_noise_analysis(img_gray: np.ndarray) -> Tuple[float, float, List[str]]:
    """
    Return (noise_std_mean, noise_cv, flags).

    Estimates local noise by computing the std of the residual
    (image − Gaussian-blurred image) in non-overlapping patches.
    """
    try:
        blurred  = _gaussian_blur_numpy(img_gray[:, :, np.newaxis], sigma=1.5)[:, :, 0]
        residual = img_gray.astype(np.float32) - blurred

        H, W = residual.shape
        stds: List[float] = []
        for i in range(0, H - _PATCH_SIZE, _PATCH_SIZE):
            for j in range(0, W - _PATCH_SIZE, _PATCH_SIZE):
                patch = residual[i:i + _PATCH_SIZE, j:j + _PATCH_SIZE]
                stds.append(float(np.std(patch)))

        if not stds:
            return 0.0, 0.0, []

        std_arr = np.array(stds, dtype=np.float32)
        mean_std = float(np.mean(std_arr))
        cv       = coeff_of_variation(std_arr)            # coeff of variation

        flags: List[str] = []
        if mean_std < _NOISE_STD_LOW:
            flags.append(
                f"Noise: unusually low noise std ({mean_std:.2f}) — "
                "may indicate AI-generated or heavily processed image"
            )
        if cv > _NOISE_CV_HIGH:
            flags.append(
                f"Noise: high spatial variation of noise (CV={cv:.2f}) — "
                "inconsistent noise pattern across image regions"
            )

        return mean_std, cv, flags

    except Exception as exc:
        return 0.0, 0.0, [f"Noise analysis skipped: {exc}"]


# ---------------------------------------------------------------------------
# Noise Pattern Consistency  (16-block macro-level check)
# ---------------------------------------------------------------------------

def _run_noise_consistency(img_gray: np.ndarray) -> Tuple[float, float, List[str]]:
    """
    Sensor noise consistency check across a 4×4 grid of 16 equal-area blocks.

    Algorithm
    ---------
    1. Compute the full-image Gaussian residual (one blur pass, not per-block).
    2. Divide the residual into 16 equal-area blocks using numpy reshape trick.
    3. For each block compute the std of residual values → 16 noise-level estimates.
    4. Compute the coefficient of variation (CV = std / mean) of those 16 estimates.

    Physical basis
    --------------
    Real camera sensors exhibit spatially stationary thermal / shot / read noise
    (PRNU is deterministic and smooth, not random).  Diffusion / GAN models
    synthesise textures region-by-region with independent noise statistics — smooth
    faces get near-zero residual, hair / foliage get high residual — producing a
    systematic spatial variation that real sensors do not.

    Returns
    -------
    (block_var, block_cv, flags)
        block_var – variance of the 16 block noise stds (raw, squared units)
        block_cv  – CV of the 16 block noise stds (normalised, key discriminator)
        flags     – human-readable anomaly descriptions
    """
    try:
        blurred  = _gaussian_blur_numpy(img_gray[:, :, np.newaxis], sigma=1.5)[:, :, 0]
        residual = img_gray.astype(np.float32) - blurred

        H, W = residual.shape
        bh = H // _N_BLOCKS   # block height (pixels)
        bw = W // _N_BLOCKS   # block width  (pixels)

        if bh < 16 or bw < 16:
            return 0.0, 0.0, []   # image too small for 16-block analysis

        # ── Reshape into (N, bh, N, bw) — vectorised, no per-block loops ────
        # Element [i, j, k, l] maps to residual pixel at (i*bh+j, k*bw+l),
        # so std over axes (1, 3) gives the noise std for block (i, k).
        cropped = residual[: _N_BLOCKS * bh, : _N_BLOCKS * bw]
        blocks  = cropped.reshape(_N_BLOCKS, bh, _N_BLOCKS, bw)

        # block_stds shape: (N, N) → 16 values
        block_stds = np.std(blocks, axis=(1, 3))
        flat_stds  = block_stds.ravel().astype(np.float64)

        mean_std  = float(np.mean(flat_stds))
        block_var = float(np.var(flat_stds))
        block_cv  = coeff_of_variation(flat_stds)

        flags: List[str] = []
        if block_cv > _NOISE_BLOCK_CV_HIGH:
            flags.append(
                f"Noise consistency: high variance across 16 blocks "
                f"(CV={block_cv:.2f}, var={block_var:.2f}) — "
                "spatially inconsistent sensor noise (AI synthesis or heavy editing)"
            )

        return block_var, block_cv, flags

    except Exception as exc:
        return 0.0, 0.0, [f"Noise consistency skipped: {exc}"]


# ---------------------------------------------------------------------------
# FFT Artifact Scan
# ---------------------------------------------------------------------------

def _run_fft(img_gray: np.ndarray) -> Tuple[float, List[str]]:
    """
    Return (peak_ratio, flags).

    Computes the 2-D power spectrum of the grayscale image.
    A high ratio of spectral peak-to-mean indicates periodic artefacts
    (common in GAN upsampling, checkerboard patterns, or lossy compression).
    """
    try:
        f  = np.fft.fft2(img_gray.astype(np.float32))
        fs = np.fft.fftshift(f)
        power = np.abs(fs) ** 2

        # Mask out DC component (centre)
        H, W = power.shape
        cy, cx = H // 2, W // 2
        r = max(5, min(H, W) // 20)
        ys, xs = np.ogrid[:H, :W]
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
        power_no_dc = power.copy()
        power_no_dc[mask] = 0.0

        peak  = float(np.max(power_no_dc))
        mean  = float(np.mean(power_no_dc[~mask]))
        ratio = peak / (mean + 1e-6)
        # Normalise to [0, 1] range using log-scale heuristic
        norm_ratio = min(1.0, np.log1p(ratio) / np.log1p(1e6))

        flags: List[str] = []
        if norm_ratio > _FFT_PEAK_HIGH:
            flags.append(
                f"FFT: strong periodic artefacts detected "
                f"(normalised peak ratio={norm_ratio:.3f}) — "
                "possible GAN upsampling or JPEG grid artefacts"
            )

        return norm_ratio, flags

    except Exception as exc:
        return 0.0, [f"FFT analysis skipped: {exc}"]


# ---------------------------------------------------------------------------
# JPEG Ghost Analysis
# ---------------------------------------------------------------------------

def _run_jpeg_ghost(img_rgb: np.ndarray) -> Tuple[float, float, List[str]]:
    """
    JPEG Ghost analysis (Farid 2009 — multi-quality residual inconsistency).

    Algorithm
    ---------
    1. Re-compress the image at each quality level in ``_GHOST_QUALITIES``.
    2. For every non-overlapping 32×32 patch, record which quality level
       produces the minimum mean absolute residual — the patch's inferred
       'native' compression quality.
    3. Build a histogram of native-quality indices across all patches.
    4. Compute normalised Shannon entropy of that histogram:
         0   — all patches share one quality level  → pristine / unedited
         100 — patches spread across all levels     → heavy editing / splicing
    5. Also compute the spatial coefficient-of-variation of residuals at the
       dominant quality level; high spatial CV reinforces the editing signal.

    Returns
    -------
    (ghost_score, spatial_cv, flags)
        ghost_score  – [0, 100]   badness value (0 = clean, 100 = edited)
        spatial_cv   – [0, ∞)    spatial residual variation at dominant quality
        flags        – human-readable anomaly descriptions
    """
    try:
        pil_img       = Image.fromarray(img_rgb.astype(np.uint8))
        orig_gray     = np.mean(img_rgb, axis=2).astype(np.float32)
        H, W          = orig_gray.shape
        n_q           = len(_GHOST_QUALITIES)

        # ── Residual map for every quality level ─────────────────────────────
        residual_maps: List[np.ndarray] = []
        for q in _GHOST_QUALITIES:
            recomp_gray = np.mean(jpeg_recompress(pil_img, q), axis=2)
            residual_maps.append(np.abs(orig_gray - recomp_gray))

        # ── Vectorised per-patch mean using reshape trick ─────────────────────
        ph = H // _GHOST_PATCH   # number of full patches along H
        pw = W // _GHOST_PATCH   # number of full patches along W
        if ph < 2 or pw < 2:
            return 0.0, 0.0, []   # image too small for meaningful analysis

        # patch_residuals: (n_q, ph, pw)
        patch_residuals = np.stack([
            rmap[:ph * _GHOST_PATCH, :pw * _GHOST_PATCH]
            .reshape(ph, _GHOST_PATCH, pw, _GHOST_PATCH)
            .mean(axis=(1, 3))
            for rmap in residual_maps
        ], axis=0)

        # ── Quality-index distribution ────────────────────────────────────────
        min_q_map = np.argmin(patch_residuals, axis=0)          # (ph, pw)
        counts    = np.bincount(min_q_map.ravel(), minlength=n_q).astype(np.float64)
        probs     = counts / (counts.sum() + 1e-10)

        # ── Normalised Shannon entropy ─────────────────────────────────────────
        nz        = probs[probs > 0]
        entropy   = float(-np.sum(nz * np.log(nz)))
        max_ent   = float(np.log(n_q))
        ghost_score = float(np.clip(entropy / (max_ent + 1e-10) * 100.0, 0.0, 100.0))

        # ── Spatial CV at dominant quality level ──────────────────────────────
        dom_qi     = int(np.argmax(counts))
        dom_pats   = patch_residuals[dom_qi]                     # (ph, pw)
        spatial_cv = coeff_of_variation(dom_pats)

        flags: List[str] = []
        if ghost_score > 70:
            flags.append(
                f"JPEG Ghost: high quality-entropy ({ghost_score:.1f}/100, "
                f"spatial CV={spatial_cv:.2f}) — "
                "regions have inconsistent compression history (splicing/composite)"
            )
        elif ghost_score > 40:
            flags.append(
                f"JPEG Ghost: moderate inconsistency ({ghost_score:.1f}/100, "
                f"spatial CV={spatial_cv:.2f}) — possible editing artefacts"
            )

        return ghost_score, spatial_cv, flags

    except Exception as exc:
        return 0.0, 0.0, [f"JPEG Ghost skipped: {exc}"]


# ---------------------------------------------------------------------------
# JPEG Quantisation Table Check
# ---------------------------------------------------------------------------

_MARKER_SOF = {0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3}
_MARKER_DQT = 0xFFDB


def _read_jpeg_quant_tables(image_bytes: bytes) -> List[List[int]]:
    """
    Parse JPEG DQT (Define Quantisation Table) markers.
    Returns a list of quantisation tables, each a flat list of 64 ints.
    """
    tables: List[List[int]] = []
    pos = 0
    data = image_bytes

    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        return tables  # not a JPEG

    pos = 2
    while pos + 3 < len(data):
        if data[pos] != 0xFF:
            break
        marker = (data[pos] << 8) | data[pos + 1]
        seg_len = (data[pos + 2] << 8) | data[pos + 3]
        seg_end = pos + 2 + seg_len

        if marker == _MARKER_DQT:
            seg = data[pos + 4: seg_end]
            offset = 0
            while offset < len(seg):
                precision_and_id = seg[offset]
                precision = (precision_and_id >> 4)  # 0=8bit, 1=16bit
                offset += 1
                n = 64 * (2 if precision else 1)
                if offset + n > len(seg):
                    break
                if precision == 0:
                    table = list(seg[offset: offset + 64])
                else:
                    table = [
                        (seg[offset + i * 2] << 8) | seg[offset + i * 2 + 1]
                        for i in range(64)
                    ]
                tables.append(table)
                offset += n

        pos = seg_end

    return tables


def _run_quant_check(image_bytes: bytes) -> Tuple[bool, List[str]]:
    """Return (suspicious, flags)."""
    try:
        tables = _read_jpeg_quant_tables(image_bytes)
        if not tables:
            return False, []

        flags: List[str] = []
        suspicious = False

        for i, table in enumerate(tables):
            # All-identical coefficients is unusual
            if len(set(table)) == 1:
                flags.append(
                    f"JPEG quant table {i}: all-uniform coefficients ({table[0]}) "
                    "— unusual for standard cameras"
                )
                suspicious = True

            # Very low quantisation coefficients (near-lossless AI output forced to JPEG)
            if max(table) < 4:
                flags.append(
                    f"JPEG quant table {i}: near-lossless quantisation "
                    "— atypical for camera JPEGs"
                )
                suspicious = True

        return suspicious, flags

    except Exception:
        return False, []


# ---------------------------------------------------------------------------
# Sub-score → badness converters  (0 = clean, 100 = heavily suspicious)
# ---------------------------------------------------------------------------

def _ela_badness(ela_p95: float) -> float:
    """
    Map the ELA 95th-percentile residual to a badness score [0, 100].
    Threshold anchors: clean ≤ 0 → 0, severe ≥ 30 → 100.
    """
    return float(min(100.0, ela_p95 / 30.0 * 100.0))


def _noise_badness(noise_std: float, noise_cv: float) -> float:
    """
    Map noise statistics to a badness score [0, 100].

    Two independent signals are combined:
      low_std_bad  – std < _NOISE_STD_LOW (2.0) → smooth/synthetic (0–50)
      high_cv_bad  – CV  > _NOISE_CV_HIGH (0.60) → inconsistent (0–50)
    Combine as max(a, b) + 0.5 * min(a, b) to reward when both fire.
    """
    low_std_bad = float(max(0.0, 50.0 * (1.0 - noise_std / _NOISE_STD_LOW))) \
        if noise_std < _NOISE_STD_LOW else 0.0
    high_cv_bad = float(max(0.0, 50.0 * (noise_cv - _NOISE_CV_HIGH) / 0.40)) \
        if noise_cv > _NOISE_CV_HIGH else 0.0
    return float(min(100.0, max(low_std_bad, high_cv_bad) + 0.5 * min(low_std_bad, high_cv_bad)))


def _fft_badness(norm_ratio: float) -> float:
    """
    Map the FFT normalised peak ratio to a badness score [0, 100].
    Threshold anchors: clean ≤ 0 → 0, suspicious ≥ _FFT_PEAK_HIGH (0.25) → 100.
    """
    return float(min(100.0, norm_ratio / _FFT_PEAK_HIGH * 100.0))


def _ghost_badness(ghost_score: float) -> float:
    """
    Ghost score is already a [0, 100] badness value (pass-through with clamp).
    0 = all patches agree on one native quality (pristine).
    100 = patches maximally disagree (heavily edited / spliced).
    """
    return float(np.clip(ghost_score, 0.0, 100.0))


def _noise_consistency_badness(block_cv: float) -> float:
    """
    Map block-level noise CV to a badness score [0, 100].

    Threshold anchors (empirical):
      CV < 0.15  → sensor-uniform (real camera) → 0
      CV ≥ 0.55  → spatially inconsistent (AI / composite) → 100
    Linear interpolation between the two anchors.
    """
    return float(np.clip((block_cv - 0.15) / 0.40 * 100.0, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Confidence band estimation
# ---------------------------------------------------------------------------

def _compute_pixel_band(fmt: str, width: int, height: int) -> float:
    """
    Estimate ± uncertainty on the pixel forensics score as a fraction of [0, 1].

    Reliability depends on image format and pixel count.

    Format influence
    ----------------
    JPEG  – ELA + JPEG Ghost both meaningful → narrow base band (0.12)
    PNG   – lossless: ELA produces artifactual residual; Ghost is meaningless → 0.20
    Other – intermediate (0.17)

    Size influence
    --------------
    < 256×256  – too few patches for statistical confidence → + 0.10
    < 512×512  – borderline → + 0.05
    """
    upper = fmt.upper()
    if upper in {"JPEG", "JPG"}:
        base = get_threshold("image.pixel_band.jpeg")
    elif upper == "PNG":
        base = get_threshold("image.pixel_band.png")
    else:
        base = get_threshold("image.pixel_band.other")

    if width * height < get_threshold("image.pixel_band.tiny_area"):
        base += get_threshold("image.pixel_band.tiny_image_penalty")
    elif width * height < get_threshold("image.pixel_band.small_area"):
        base += get_threshold("image.pixel_band.small_image_penalty")

    return min(base, 0.30)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_bytes: bytes) -> PixelForensicsResult:
    """
    Run all pixel-forensics checks on *image_bytes*.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the image file.

    Returns
    -------
    PixelForensicsResult
    """
    try:
        raw_img  = Image.open(io.BytesIO(image_bytes))
        img_fmt  = raw_img.format or ""          # capture before convert() clears it
        img      = raw_img.convert("RGB")
        img_rgb  = np.array(img, dtype=np.float32)
        img_gray = np.mean(img_rgb, axis=2)
    except Exception as exc:
        return PixelForensicsResult(
            score=0.5,
            artifacts=[f"Could not decode image for pixel analysis: {exc}"],
        )

    all_flags: List[str] = []

    # ── ELA ─────────────────────────────────────────────────────────────────
    ela_p95, ela_flags = _run_ela(img_rgb)
    all_flags.extend(ela_flags)

    # ── Noise (patch-level) ──────────────────────────────────────────────────
    noise_std, noise_cv, noise_flags = _run_noise_analysis(img_gray)
    all_flags.extend(noise_flags)

    # ── Noise consistency (16-block macro-level) ─────────────────────────────
    block_var, block_cv, consist_flags = _run_noise_consistency(img_gray)
    all_flags.extend(consist_flags)

    # ── FFT ─────────────────────────────────────────────────────────────────
    fft_ratio, fft_flags = _run_fft(img_gray)
    all_flags.extend(fft_flags)

    # ── JPEG Ghost ───────────────────────────────────────────────────────────
    ghost_score, ghost_cv, ghost_flags = _run_jpeg_ghost(img_rgb)
    all_flags.extend(ghost_flags)

    # ── JPEG quant ──────────────────────────────────────────────────────────
    quant_suspicious, quant_flags = _run_quant_check(image_bytes)
    all_flags.extend(quant_flags)

    # ── Scoring formula:
    #    pixel_score = 100 − (ela × 0.30 + ghost × 0.20 + noise × 0.25 + fft × 0.25)
    #    noise = 60% patch-level badness + 40% 16-block consistency badness
    ela_bad         = _ela_badness(ela_p95)
    ghost_bad       = _ghost_badness(ghost_score)
    patch_noise_bad = _noise_badness(noise_std, noise_cv)
    consist_bad     = _noise_consistency_badness(block_cv)
    noise_bad       = (get_weight("image.pixel_formula.noise_patch") * patch_noise_bad
                       + get_weight("image.pixel_formula.noise_block") * consist_bad)
    fft_bad         = _fft_badness(fft_ratio)

    pixel_score_100 = 100.0 - (
        ela_bad   * get_weight("image.pixel_formula.ela")   +
        ghost_bad * get_weight("image.pixel_formula.ghost") +
        noise_bad * get_weight("image.pixel_formula.noise") +
        fft_bad   * get_weight("image.pixel_formula.fft")
    )

    # JPEG quant anomaly: extra −5 (informational; outside weighted formula)
    if quant_suspicious:
        pixel_score_100 -= 5.0

    score = max(0.0, min(100.0, pixel_score_100)) / 100.0

    return PixelForensicsResult(
        score                   = score,
        artifacts               = all_flags,
        ela_max_diff            = ela_p95,
        fft_peak_ratio          = fft_ratio,
        noise_uniformity        = noise_cv,
        ghost_score             = ghost_score,
        noise_block_consistency = block_cv,
        confidence_band         = _compute_pixel_band(img_fmt, img.width, img.height),
    )
