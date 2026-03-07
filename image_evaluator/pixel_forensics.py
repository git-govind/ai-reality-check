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

  Noise Residual Analysis
      Subtracts a Gaussian-blurred version to isolate sensor noise.
      Real camera images have characteristic wide-band noise with quasi-
      Gaussian statistics.  AI-generated images are often unnaturally smooth
      (very low noise) or have spatially correlated noise patterns.

  Frequency-Domain (FFT) Artifact Scan
      Computes the 2-D power spectrum.  Periodic grid artefacts (common in
      GAN upsampling and JPEG blocking) produce prominent spectral spikes.

  JPEG Quantisation Table Check
      Inspects the raw JFIF/JPEG quantisation tables.  Some AI tools embed
      tables with unusual quality factors or all-equal entries.

Scoring formula (spec §2)
-------------------------
  pixel_score = 100 − (ela_score × 0.40 + noise_score × 0.30 + fft_score × 0.30)

  Each sub-score is a "badness" value in [0, 100]:
    ela_score   = min(100, ela_p95 / 30 × 100)
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ELA_QUALITY         = 75       # JPEG quality for re-compression in ELA
_ELA_HIGH_THRESHOLD  = 30.0     # 95th-pct diff → strong manipulation signal
_ELA_MED_THRESHOLD   = 15.0     # 95th-pct diff → moderate signal
_NOISE_CV_HIGH       = 0.60     # coefficient of variation → unusual uniformity
_NOISE_STD_LOW       = 2.0      # std below this → suspiciously smooth
_FFT_PEAK_HIGH       = 0.25     # peak / mean power ratio → periodic artefacts
_PATCH_SIZE          = 32       # patch size for local noise estimation


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
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
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
        cv       = float(np.std(std_arr) / (mean_std + 1e-6))  # coeff of variation

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
        img      = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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

    # ── Noise ───────────────────────────────────────────────────────────────
    noise_std, noise_cv, noise_flags = _run_noise_analysis(img_gray)
    all_flags.extend(noise_flags)

    # ── FFT ─────────────────────────────────────────────────────────────────
    fft_ratio, fft_flags = _run_fft(img_gray)
    all_flags.extend(fft_flags)

    # ── JPEG quant ──────────────────────────────────────────────────────────
    quant_suspicious, quant_flags = _run_quant_check(image_bytes)
    all_flags.extend(quant_flags)

    # ── Exact spec formula:
    #    pixel_score = 100 − (ela_score × 0.40 + noise_score × 0.30 + fft_score × 0.30)
    ela_bad   = _ela_badness(ela_p95)
    noise_bad = _noise_badness(noise_std, noise_cv)
    fft_bad   = _fft_badness(fft_ratio)

    pixel_score_100 = 100.0 - (ela_bad * 0.40 + noise_bad * 0.30 + fft_bad * 0.30)

    # JPEG quant anomaly: extra −5 (informational; outside weighted formula)
    if quant_suspicious:
        pixel_score_100 -= 5.0

    score = max(0.0, min(100.0, pixel_score_100)) / 100.0

    return PixelForensicsResult(
        score            = score,
        artifacts        = all_flags,
        ela_max_diff     = ela_p95,
        fft_peak_ratio   = fft_ratio,
        noise_uniformity = noise_cv,
    )
