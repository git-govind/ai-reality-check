"""
ai_artifact_classifier.py
--------------------------
Step 3 of the Image Authenticity Evaluator pipeline.

Estimates the probability that an image was produced by a generative AI
(GAN, diffusion model, VAE, etc.) by combining multiple independent signals:

Heuristic layer (always runs, no downloads required)
-----------------------------------------------------
  1. Texture smoothness  – AI images have unnaturally smooth textures
                           measured as mean local variance over 16-px patches.
  2. Colour coherence    – Diffusion models produce globally coherent palettes;
                           measured as low entropy in the HSV hue histogram.
  3. Edge sharpness      – AI images often have crisp but geometrically
                           improbable edges; measured via Sobel gradient stats.
  4. High-freq / low-freq ratio  – Sensor noise gives real photos a distinct
                                   high-frequency fingerprint; AI images lack it.
  5. Saturation skew     – AI generators trend toward aesthetically pleasing
                           hyper-saturated colours; measured via HSV S-histogram.
  6. Spatial symmetry    – Faces / objects in AI images exhibit slight bilateral
                           symmetry not present in candid photos.

Model layer (optional, requires sentence-transformers with CLIP)
----------------------------------------------------------------
  If ``sentence_transformers`` and ``PIL`` are available and the CLIP model
  ``clip-ViT-B-32`` can be loaded, an additional model-based score is
  computed by projecting the image into CLIP's vision embedding space and
  measuring its cosine distance to a reference "photographic" anchor.
  This is a lightweight proxy; a production system should use a dedicated
  deepfake/AI-detector model (e.g., Grounding-DINO, CNNDetect, UnivFD).

  Set the environment variable ``SKIP_CLIP=1`` to always use heuristics only.

Combination
-----------
  ai_prob = weighted average of active sub-scores, with the model score
  receiving a higher weight when available.

Dependencies: Pillow, NumPy (required); sentence-transformers (optional)
"""

from __future__ import annotations

import io
import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from .datatypes import AIArtifactResult

# ---------------------------------------------------------------------------
# CLIP model (optional)
# ---------------------------------------------------------------------------

_CLIP_MODEL = None
_CLIP_LOADED = False
_CLIP_AVAILABLE = False


def _try_load_clip() -> None:
    global _CLIP_MODEL, _CLIP_LOADED, _CLIP_AVAILABLE
    if _CLIP_LOADED:
        return
    _CLIP_LOADED = True

    if os.getenv("SKIP_CLIP", "").strip() == "1":
        return

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _CLIP_MODEL = SentenceTransformer("clip-ViT-B-32")
        _CLIP_AVAILABLE = True
    except Exception:
        pass  # silently fall back to heuristics


# ---------------------------------------------------------------------------
# Heuristic sub-classifiers
# ---------------------------------------------------------------------------

def _score_texture_smoothness(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Measure local variance in 16×16 patches.
    Very low mean variance → suspiciously smooth → higher AI probability.
    Returns (ai_sub_prob 0-1, description).
    """
    H, W   = img_gray.shape
    patch  = 16
    variances: List[float] = []
    for i in range(0, H - patch, patch):
        for j in range(0, W - patch, patch):
            variances.append(float(np.var(img_gray[i:i + patch, j:j + patch])))

    if not variances:
        return 0.5, "texture: insufficient image size"

    mean_var = np.mean(variances)
    # Empirically: real photos have mean_var > 300; AI smoothness < 80
    # Sigmoid-like mapping into [0, 1]
    sub_prob = float(1.0 / (1.0 + (mean_var / 80.0) ** 1.5))
    label = f"texture smoothness: mean patch variance={mean_var:.1f}"
    return sub_prob, label


def _score_colour_coherence(img_hsv: np.ndarray) -> Tuple[float, str]:
    """
    Analyse hue entropy.  Low hue entropy → few dominant hues →
    often indicates AI stylisation or synthetic scenes.
    Returns (ai_sub_prob 0-1, description).
    """
    hue = img_hsv[:, :, 0].flatten()
    hist, _ = np.histogram(hue, bins=36, range=(0, 180))
    hist     = hist.astype(np.float32) + 1e-9
    hist    /= hist.sum()
    entropy = float(-np.sum(hist * np.log2(hist)))
    # Full entropy ≈ 5.17 (log2(36)); lower entropy = fewer hues
    norm_entropy = entropy / 5.17
    # Very low entropy (< 0.5) = suspicious
    sub_prob = max(0.0, 1.0 - norm_entropy * 1.5)
    label    = f"colour coherence: hue entropy={entropy:.2f}"
    return sub_prob, label


def _score_edge_sharpness(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Sobel gradient statistics.
    AI images sometimes have implausibly sharp edges everywhere.
    A very high 95th-percentile and low std of the gradient → suspicious.
    Returns (ai_sub_prob 0-1, description).
    """
    # Simple Sobel via numpy convolution
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T

    def _conv2d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided
        ph, pw = k.shape[0] // 2, k.shape[1] // 2
        padded = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
        H, W = a.shape
        kH, kW = k.shape
        shape   = (H, W, kH, kW)
        strides = padded.strides + padded.strides
        patches = as_strided(padded, shape=shape, strides=strides)
        return np.einsum("ijkl,kl->ij", patches, k)

    gx     = _conv2d_same(img_gray, sobel_x)
    gy     = _conv2d_same(img_gray, sobel_y)
    mag    = np.sqrt(gx ** 2 + gy ** 2)
    p95    = float(np.percentile(mag, 95))
    cv     = float(np.std(mag) / (np.mean(mag) + 1e-6))

    # Very high p95 with low CV = sharp edges everywhere = suspicious
    sharpness_prob = min(1.0, p95 / 400.0)
    uniformity_pen = max(0.0, 1.0 - cv)
    sub_prob = sharpness_prob * uniformity_pen * 0.5
    label = f"edge sharpness: p95={p95:.1f}, CV={cv:.2f}"
    return sub_prob, label


def _score_hf_lf_ratio(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    High-frequency / low-frequency power ratio via FFT.
    Real camera images have a distinct high-frequency noise fingerprint.
    AI images tend to be cleaner → lower HF/LF ratio.
    Returns (ai_sub_prob 0-1, description).
    """
    F    = np.fft.fft2(img_gray.astype(np.float32))
    Fs   = np.fft.fftshift(F)
    power = np.abs(Fs) ** 2

    H, W = power.shape
    cy, cx = H // 2, W // 2
    ys, xs = np.ogrid[:H, :W]
    dist   = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_mid  = min(H, W) / 4

    lf_power = float(np.mean(power[dist <= r_mid]))
    hf_power = float(np.mean(power[dist  > r_mid]))
    ratio    = hf_power / (lf_power + 1e-9)

    # Low HF/LF (< 0.01) → suspiciously smooth spectrum → AI
    sub_prob = float(1.0 / (1.0 + (ratio / 0.005) ** 1.2))
    label    = f"HF/LF ratio: {ratio:.5f}"
    return sub_prob, label


def _score_saturation_skew(img_hsv: np.ndarray) -> Tuple[float, str]:
    """
    Analyse saturation histogram skewness.
    AI generators trend toward hyper-saturated images.
    Returns (ai_sub_prob 0-1, description).
    """
    sat   = img_hsv[:, :, 1].flatten().astype(np.float32)
    mean_ = float(np.mean(sat))
    # High mean saturation > 140 (out of 255) is unusual in real photos
    sub_prob = min(1.0, max(0.0, (mean_ - 80) / 120.0))
    label    = f"saturation: mean={mean_:.1f}/255"
    return sub_prob, label


def _score_symmetry(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Measure bilateral (left-right) symmetry.
    A high symmetry score on a non-portrait photo is suspicious.
    Returns (ai_sub_prob 0-1, description).
    """
    H, W  = img_gray.shape
    left  = img_gray[:, : W // 2].astype(np.float32)
    right = img_gray[:, W - W // 2:][:, ::-1].astype(np.float32)
    diff  = np.mean(np.abs(left - right))
    # Low diff → high symmetry → suspicious (normalise by max possible ~127)
    sub_prob = max(0.0, 1.0 - diff / 60.0)
    label    = f"bilateral symmetry: mean diff={diff:.2f}"
    return sub_prob, label


# ---------------------------------------------------------------------------
# FreqNet-inspired frequency-domain classifier (spec §3)
# ---------------------------------------------------------------------------

def _score_freqnet(img_gray: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
    """
    FreqNet-inspired frequency-domain AI artifact detector.

    Inspired by: Frank et al. (2020) "Leveraging Frequency Analysis for Deep
    Fake Image Recognition".  Implemented in pure NumPy — no additional
    dependencies required.

    Key insight: GAN / diffusion-generated images exhibit characteristic
    spectral artifacts compared with real camera photographs:
      1. Spectral roll-off – AI images have steeper power drop from low to
         high frequencies (too clean, lack sensor-noise fingerprint).
      2. Mid-frequency anomaly – upsampling artifacts leave residual patterns
         in the mid-frequency band.
      3. Spectral entropy – AI images have lower spectral entropy (more
         regular periodic structure from generator architecture).
      4. Peak prominence – GAN upsampling grids create sharp spectral peaks
         not present in real photographs.

    Returns (ai_sub_prob 0-1, description, features_dict).
    """
    F      = np.fft.fft2(img_gray.astype(np.float32))
    Fs     = np.fft.fftshift(F)
    power  = np.abs(Fs) ** 2 + 1e-10

    H, W   = power.shape
    cy, cx = H // 2, W // 2
    ys, xs = np.ogrid[:H, :W]
    dist   = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    r_dc   = max(3, min(H, W) // 30)   # exclude DC component
    r_lf   = min(H, W) / 8             # low-freq radius
    r_mf   = min(H, W) / 4             # mid-freq radius
    r_hf   = min(H, W) / 2             # high-freq radius

    lf_mask  = (dist >  r_dc) & (dist <= r_lf)
    mf_mask  = (dist >  r_lf) & (dist <= r_mf)
    hf_mask  = (dist >  r_mf) & (dist <= r_hf)
    all_mask = (dist >  r_dc) & (dist <= r_hf)

    log_pwr  = np.log10(power)
    lf_mean  = float(np.mean(log_pwr[lf_mask]))
    mf_mean  = float(np.mean(log_pwr[mf_mask]))
    hf_mean  = float(np.mean(log_pwr[hf_mask]))

    # ── Signal 1: spectral roll-off ─────────────────────────────────────────
    # Real photos: gradual roll-off (low value).
    # AI images: steeper drop → higher value → higher AI sub-prob.
    rolloff      = (lf_mean - hf_mean) / (abs(lf_mean) + 1e-9)
    rolloff_prob = float(np.clip((rolloff - 0.15) / 0.20, 0.0, 1.0))

    # ── Signal 2: mid-frequency anomaly ────────────────────────────────────
    # AI images leave structured residuals in mid-freq from upsampling.
    expected_mf  = (lf_mean + hf_mean) / 2.0
    mid_anomaly  = abs(mf_mean - expected_mf) / (abs(lf_mean) + 1e-9)
    anomaly_prob = float(np.clip(mid_anomaly * 3.0, 0.0, 1.0))

    # ── Signal 3: spectral entropy ──────────────────────────────────────────
    # AI images exhibit lower entropy (more regular periodic structure).
    p_vals    = power[all_mask]
    p_norm    = p_vals / (p_vals.sum() + 1e-10)
    spec_ent  = float(-np.sum(p_norm * np.log(p_norm + 1e-10)))
    max_ent   = float(np.log(p_norm.size))
    norm_ent  = spec_ent / (max_ent + 1e-9)
    ent_prob  = float(np.clip(1.0 - norm_ent * 1.2, 0.0, 1.0))

    # ── Signal 4: peak prominence ───────────────────────────────────────────
    # GAN upsampling grids produce sharp isolated peaks in the spectrum.
    lp_vals   = log_pwr[all_mask]
    peak_prom = (float(np.max(lp_vals)) - float(np.mean(lp_vals))) / \
                (abs(float(np.mean(lp_vals))) + 1e-9)
    peak_prob = float(np.clip((peak_prom - 0.5) / 1.0, 0.0, 1.0))

    # ── Combine equal-weight average ────────────────────────────────────────
    ai_sub = float(np.mean([rolloff_prob, anomaly_prob, ent_prob, peak_prob]))

    features_dict: Dict[str, Any] = {
        "freqnet_rolloff":      round(rolloff,      4),
        "freqnet_mid_anomaly":  round(mid_anomaly,  4),
        "freqnet_spec_entropy": round(norm_ent,     4),
        "freqnet_peak_prom":    round(peak_prom,    4),
        "freqnet_rolloff_prob": round(rolloff_prob, 3),
        "freqnet_anomaly_prob": round(anomaly_prob, 3),
        "freqnet_entropy_prob": round(ent_prob,     3),
        "freqnet_peak_prob":    round(peak_prob,    3),
        "freqnet_ai_sub":       round(ai_sub,       3),
    }
    label = (
        f"FreqNet: rolloff={rolloff:.3f}, mid_anomaly={mid_anomaly:.3f}, "
        f"spec_entropy={norm_ent:.3f}, peak_prom={peak_prom:.3f} "
        f"→ ai_sub={ai_sub:.2f}"
    )
    return ai_sub, label, features_dict


# ---------------------------------------------------------------------------
# CLIP-based score (optional)
# ---------------------------------------------------------------------------

def _score_clip(img: Image.Image) -> Tuple[float, str]:
    """
    Project the image into CLIP embedding space.
    Compute cosine similarity against a set of reference prompts.
    Low similarity to "real photograph" prompts → higher AI suspicion.
    Returns (ai_sub_prob 0-1, description).
    """
    if not _CLIP_AVAILABLE or _CLIP_MODEL is None:
        return 0.5, "CLIP: not available"

    try:
        # CLIP SentenceTransformer can encode PIL images directly
        img_emb = _CLIP_MODEL.encode(img, convert_to_tensor=True)

        real_prompts = [
            "a real photograph taken with a camera",
            "a candid photo of a real person",
            "a documentary photograph",
        ]
        ai_prompts = [
            "an AI-generated digital artwork",
            "a photorealistic image made by a diffusion model",
            "a synthetic computer-generated image",
        ]

        from sentence_transformers import util  # type: ignore
        real_embs = _CLIP_MODEL.encode(real_prompts, convert_to_tensor=True)
        ai_embs   = _CLIP_MODEL.encode(ai_prompts,   convert_to_tensor=True)

        real_sim = float(util.cos_sim(img_emb, real_embs).mean())
        ai_sim   = float(util.cos_sim(img_emb,   ai_embs).mean())

        # Convert similarity gap to probability
        total    = real_sim + ai_sim
        ai_prob  = ai_sim / (total + 1e-6)
        label    = f"CLIP: real_sim={real_sim:.3f}, ai_sim={ai_sim:.3f}"
        return ai_prob, label

    except Exception as exc:
        return 0.5, f"CLIP: error ({exc})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_bytes: bytes) -> AIArtifactResult:
    """
    Classify whether *image_bytes* was AI-generated.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.

    Returns
    -------
    AIArtifactResult
        ai_prob  – 0.0 (almost certainly real) … 1.0 (almost certainly AI)
        features – list of signal descriptions
        method   – "heuristic" or "heuristic+model"
    """
    _try_load_clip()

    try:
        img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(img, dtype=np.float32)
        img_gray = np.mean(img_rgb, axis=2)
        img_hsv  = np.array(img.convert("HSV"), dtype=np.float32)
    except Exception as exc:
        return AIArtifactResult(
            ai_prob=0.5,
            features=[f"Could not decode image: {exc}"],
            method="heuristic",
        )

    features: List[str] = []
    sub_probs: List[float] = []
    fdict: Dict[str, Any] = {}

    # ── Heuristic sub-scores ─────────────────────────────────────────────────
    heuristic_labels = [
        "texture", "colour", "edge", "hf_lf", "saturation", "symmetry"
    ]
    for name, scorer in zip(heuristic_labels, (
        lambda: _score_texture_smoothness(img_gray),
        lambda: _score_colour_coherence(img_hsv),
        lambda: _score_edge_sharpness(img_gray),
        lambda: _score_hf_lf_ratio(img_gray),
        lambda: _score_saturation_skew(img_hsv),
        lambda: _score_symmetry(img_gray),
    )):
        try:
            prob, label = scorer()
            sub_probs.append(prob)
            features.append(f"{label} → ai_sub={prob:.2f}")
            fdict[name] = round(prob, 3)
        except Exception as exc:
            features.append(f"sub-scorer error ({name}): {exc}")

    heuristic_prob = float(np.mean(sub_probs)) if sub_probs else 0.5
    method = "heuristic"

    # ── FreqNet-inspired frequency-domain scorer ─────────────────────────────
    try:
        freq_prob, freq_label, freq_dict = _score_freqnet(img_gray)
        features.append(freq_label)
        fdict.update(freq_dict)
        # Blend: FreqNet gets 30 %, heuristics 70 %
        combined_prob = 0.70 * heuristic_prob + 0.30 * freq_prob
        method = "heuristic+freqnet"
    except Exception as exc:
        combined_prob = heuristic_prob
        features.append(f"FreqNet scorer error: {exc}")

    # ── CLIP model score (optional) ──────────────────────────────────────────
    if _CLIP_AVAILABLE:
        clip_prob, clip_label = _score_clip(img)
        features.append(clip_label)
        fdict["clip_ai_prob"] = round(clip_prob, 3)
        # CLIP gets 30 %, combined (heuristic+freqnet) gets 70 %
        ai_prob = 0.70 * combined_prob + 0.30 * clip_prob
        method  = "heuristic+freqnet+model"
    else:
        ai_prob = combined_prob

    ai_prob = float(np.clip(ai_prob, 0.0, 1.0))
    fdict["ai_prob_final"] = round(ai_prob, 3)

    return AIArtifactResult(
        ai_prob=ai_prob,
        features=features,
        features_dict=fdict,
        method=method,
    )
