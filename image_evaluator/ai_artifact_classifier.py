"""
ai_artifact_classifier.py
--------------------------
Step 3 of the Image Authenticity Evaluator pipeline.

Estimates the probability that an image was produced by a generative AI
(GAN, diffusion model, VAE, etc.).

Classifier layers  (priority order)
-------------------------------------
1. HuggingFace trained binary classifier  ← PRIMARY
     Preferred : umm-maybe/AI-image-detector
                 ViT fine-tuned on a broad real-vs-AI dataset (~343 MB).
     Alternative: Organika/sdxl-detector
                 ResNet50 fine-tuned specifically on SDXL outputs.
     Loaded lazily on first call; cached for the lifetime of the process.
     Requires ``transformers`` to be installed (already in requirements.txt).
     Set env-var ``SKIP_HF_MODEL=1`` to force heuristic-only mode.

2. FreqNet-inspired frequency-domain scorer  ← ALWAYS RUNS (supporting signal)
     Pure NumPy FFT analysis — zero additional dependencies.
     Captures spectral roll-off, mid-freq anomalies, spectral entropy,
     and GAN upsampling peak prominence.

3. Six pixel-level heuristic scorers  ← ALWAYS RUNS (supporting signal)
     texture smoothness · colour coherence · edge sharpness ·
     HF/LF ratio · saturation skew · bilateral symmetry.

Note on CLIP
-------------
The CLIP "real photograph vs AI-generated image" prompt comparison has been
removed from this module.  CLIP was designed for visual-text alignment, not
image authenticity detection.  Cosine similarity against crafted prompts does
not reliably separate authentic from synthetic images.  CLIP remains in
image_text_consistency.py for its legitimate purpose.

Combination
-----------
  Model available:
      ai_prob         = 0.80 × model_prob  +  0.20 × heuristic_blend
      confidence_band = min(model_prob, 1 − model_prob)   ← logit margin

  Model absent (heuristic only):
      ai_prob         = 0.70 × heuristic_mean  +  0.30 × freqnet_prob
      confidence_band = std(all_sub_scores)    ← agreement spread

Output
------
  AIArtifactResult.ai_prob          – probability [0, 1]
  AIArtifactResult.confidence_band  – ± uncertainty radius
  AIArtifactResult.feature_vector   – all numeric sub-scores as a flat list
  AIArtifactResult.features_dict    – same signals as a structured dict
  AIArtifactResult.features         – human-readable descriptions (UI)
  AIArtifactResult.method           – short string describing what ran

Dependencies: Pillow, NumPy (required); transformers (optional for model)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .datatypes import AIArtifactResult

from config_loader import get_feature, get_threshold, get_weight
from models.image_model_registry import get_model as _get_image_model
from utils.image_utils import load_image_rgb

# ---------------------------------------------------------------------------
# HuggingFace trained classifier  (primary)
# ---------------------------------------------------------------------------

# AI-label keywords — used to identify which output label represents "AI-generated"
_AI_LABEL_KEYWORDS = frozenset({
    "artificial", "fake", "generated", "ai", "synthetic",
    "sdxl", "machine", "aigc", "deepfake",
})


# ---------------------------------------------------------------------------
# CLIP-based out-of-distribution style detector
# ---------------------------------------------------------------------------
#
# The binary AI-vs-real classifiers in this module are trained on photographic
# images (real camera shots vs diffusion/GAN outputs).  They are unreliable
# for image styles that fall outside that photographic distribution:
#   anime, 3D renders, digital illustrations, screenshots.
#
# We use CLIP zero-shot classification to detect these OOD styles at the
# same time the main classifier runs, then surface a warning via
# AIArtifactResult.ood_warning.
#
# Model: openai/clip-vit-base-patch32  (~350 MB; loaded lazily, cached)
# Env-var SKIP_CLIP_OOD=1 disables loading (useful in low-memory environments).
# If the model cannot be loaded for any reason, OOD detection is skipped
# silently — no warning is emitted, which is the safe fallback.

# One text prompt per style class.
# The first entry ("photo") is the in-distribution baseline.
_OOD_PROMPTS: List[str] = [
    "a photo taken with a camera",
    "anime or manga artwork",
    "a 3D rendered image or CGI animation",
    "a digital illustration or vector artwork",
    "a screenshot of a user interface or computer screen",
]
_OOD_CATEGORIES: List[str] = [
    "photo",
    "anime",
    "3d_render",
    "illustration",
    "screenshot",
]
_OOD_WARNING: dict[str, str] = {
    "anime":        "Anime artwork detected — AI detector may be unreliable for this image type.",
    "3d_render":    "3D render / CGI detected — AI detector may be unreliable for this image type.",
    "illustration": "Digital illustration detected — AI detector may be unreliable for this image type.",
    "screenshot":   "Screenshot detected — AI detector may be unreliable for this image type.",
}

# Minimum softmax score for an OOD class to trigger a warning.
# With 5 classes uniform = 0.20; setting 0.32 requires the OOD class to be
# clearly preferred over the photo baseline before we emit a warning.
_OOD_THRESHOLD = get_threshold("image.ood_threshold")

# Blend weights for ai_prob combination (loaded once at import time)
_W_MODEL      = get_weight("image.ai_classifier.model_weight")
_W_HEURISTIC  = get_weight("image.ai_classifier.heuristic_weight")
_W_HEUR_MEAN  = get_weight("image.ai_classifier.heuristic_mean")
_W_FREQNET    = get_weight("image.ai_classifier.freqnet_weight")



def _detect_ood_style(img: Image.Image) -> Tuple[str, str]:
    """
    Use CLIP zero-shot classification to detect out-of-distribution image styles.

    Parameters
    ----------
    img : PIL.Image.Image (already converted to RGB)

    Returns
    -------
    (ood_category, warning_message)
        ood_category     – one of "anime" | "3d_render" | "illustration" |
                           "screenshot", or "" for in-distribution photographs.
        warning_message  – human-readable warning string, or "" if none.
    """
    clip = _get_image_model("clip_ood")
    if not clip.available or clip.model is None or clip.processor is None:
        return "", ""

    try:
        import torch  # type: ignore

        inputs = clip.processor(
            text=_OOD_PROMPTS,
            images=img,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs      = clip.model(**inputs)
        # logits_per_image: (1, num_prompts)
        probs        = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        best_idx   = int(max(range(len(probs)), key=lambda i: probs[i]))
        best_score = float(probs[best_idx])
        best_cat   = _OOD_CATEGORIES[best_idx]

        if best_cat != "photo" and best_score >= _OOD_THRESHOLD:
            warning = _OOD_WARNING.get(
                best_cat,
                f"{best_cat.replace('_', ' ').title()} detected — "
                "AI detector may be unreliable for this image type.",
            )
            return best_cat, warning

        return "", ""
    except Exception:
        return "", ""  # never raise — OOD check is best-effort only


def _run_hf_model(img: Image.Image) -> Tuple[float, float, str]:
    """
    Run the loaded HuggingFace pipeline on *img*.

    Returns
    -------
    (ai_prob, confidence_band, model_label)
        ai_prob         – [0, 1]  probability of being AI-generated
        confidence_band – [0, 0.5]  ± uncertainty radius from logit margin
                          small (→ 0) when the model is confident,
                          large (→ 0.5) when near the decision boundary
        model_label     – human-readable label string from the model
    """
    hf = _get_image_model("hf_classifier")
    try:
        results = hf.pipe(img)   # list of {"label": str, "score": float}

        # Find the AI-class score by matching label keywords
        for item in results:
            if any(kw in item["label"].lower() for kw in _AI_LABEL_KEYWORDS):
                p_ai   = float(item["score"])
                band   = float(min(p_ai, 1.0 - p_ai))
                label  = f"{hf.model_id}: {item['label']}={p_ai:.3f}"
                return p_ai, band, label

        # Fallback: assume top label is "real", complement is AI probability
        top     = max(results, key=lambda x: x["score"])
        p_real  = float(top["score"])
        p_ai    = 1.0 - p_real
        band    = float(min(p_ai, p_real))
        label   = f"{hf.model_id}: ~ai_prob={p_ai:.3f} (label '{top['label']}' treated as real)"
        return p_ai, band, label

    except Exception as exc:
        return 0.5, 0.5, f"{hf.model_id}: inference error — {exc}"


# ---------------------------------------------------------------------------
# Heuristic sub-classifiers  (supporting signals — always run)
# ---------------------------------------------------------------------------

def _score_texture_smoothness(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Mean local variance over 16×16 patches.
    Very low mean variance → suspiciously smooth → higher AI sub-prob.
    """
    H, W   = img_gray.shape
    patch  = 16
    variances: List[float] = []
    for i in range(0, H - patch, patch):
        for j in range(0, W - patch, patch):
            variances.append(float(np.var(img_gray[i:i + patch, j:j + patch])))

    if not variances:
        return 0.5, "texture: insufficient image size"

    mean_var = float(np.mean(variances))
    sub_prob = float(1.0 / (1.0 + (mean_var / 80.0) ** 1.5))
    return sub_prob, f"texture smoothness: mean_var={mean_var:.1f}"


def _score_colour_coherence(img_hsv: np.ndarray) -> Tuple[float, str]:
    """
    Hue histogram entropy.  Low entropy → few dominant hues → AI-like.
    """
    hue  = img_hsv[:, :, 0].flatten()
    hist = np.histogram(hue, bins=36, range=(0, 180))[0].astype(np.float32) + 1e-9
    hist /= hist.sum()
    entropy      = float(-np.sum(hist * np.log2(hist)))
    norm_entropy = entropy / 5.17   # max ≈ log2(36)
    sub_prob     = float(max(0.0, 1.0 - norm_entropy * 1.5))
    return sub_prob, f"colour coherence: hue_entropy={entropy:.2f}"


def _score_edge_sharpness(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Sobel gradient statistics.
    Implausibly sharp edges everywhere (high p95, low CV) → suspicious.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T

    def _conv2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided
        ph, pw  = k.shape[0] // 2, k.shape[1] // 2
        padded  = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
        H2, W2  = a.shape
        kH, kW  = k.shape
        patches = as_strided(
            padded,
            shape=(H2, W2, kH, kW),
            strides=padded.strides + padded.strides,
        )
        return np.einsum("ijkl,kl->ij", patches, k)

    mag      = np.sqrt(_conv2d(img_gray, sobel_x) ** 2 + _conv2d(img_gray, sobel_y) ** 2)
    p95      = float(np.percentile(mag, 95))
    cv       = float(np.std(mag) / (np.mean(mag) + 1e-6))
    sub_prob = float(min(1.0, p95 / 400.0) * max(0.0, 1.0 - cv) * 0.5)
    return sub_prob, f"edge sharpness: p95={p95:.1f}, CV={cv:.2f}"


def _score_hf_lf_ratio(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    HF/LF power ratio via FFT.
    AI images lack camera sensor noise → unnaturally low HF/LF ratio.
    """
    Fs    = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float32)))
    power = np.abs(Fs) ** 2
    H, W  = power.shape
    cy, cx = H // 2, W // 2
    dist  = np.sqrt((np.ogrid[:H][0] - cy) ** 2 + (np.ogrid[:W][0] - cx).reshape(1, -1) ** 2)
    r_mid = min(H, W) / 4
    ratio = float(np.mean(power[dist > r_mid])) / (float(np.mean(power[dist <= r_mid])) + 1e-9)
    sub_prob = float(1.0 / (1.0 + (ratio / 0.005) ** 1.2))
    return sub_prob, f"HF/LF ratio: {ratio:.5f}"


def _score_saturation_skew(img_hsv: np.ndarray) -> Tuple[float, str]:
    """
    Mean saturation.  AI generators trend toward hyper-saturated images.
    """
    mean_sat = float(np.mean(img_hsv[:, :, 1]))
    sub_prob = float(min(1.0, max(0.0, (mean_sat - 80) / 120.0)))
    return sub_prob, f"saturation: mean={mean_sat:.1f}/255"


def _score_symmetry(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Bilateral (left-right) symmetry.
    Unnaturally high symmetry in a non-portrait shot is suspicious.
    """
    H, W  = img_gray.shape
    left  = img_gray[:, : W // 2].astype(np.float32)
    right = img_gray[:, W - W // 2:][:, ::-1].astype(np.float32)
    diff  = float(np.mean(np.abs(left - right)))
    sub_prob = float(max(0.0, 1.0 - diff / 60.0))
    return sub_prob, f"bilateral symmetry: mean_diff={diff:.2f}"


# ---------------------------------------------------------------------------
# FreqNet-inspired frequency-domain classifier  (supporting signal)
# ---------------------------------------------------------------------------

def _score_freqnet(img_gray: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
    """
    FreqNet-inspired frequency-domain AI artifact detector.

    Inspired by: Frank et al. (2020) "Leveraging Frequency Analysis for Deep
    Fake Image Recognition".  Pure NumPy — no additional dependencies.

    Four spectral signals:
      1. Spectral roll-off    – steeper drop in AI images (lack sensor noise)
      2. Mid-freq anomaly     – residual patterns from diffusion upsampling
      3. Spectral entropy     – lower in AI (more regular periodic structure)
      4. Peak prominence      – GAN upsampling grids create sharp spectral peaks

    Returns (ai_sub_prob 0-1, description, features_dict).
    """
    F      = np.fft.fft2(img_gray.astype(np.float32))
    Fs     = np.fft.fftshift(F)
    power  = np.abs(Fs) ** 2 + 1e-10

    H, W   = power.shape
    cy, cx = H // 2, W // 2
    ys, xs = np.ogrid[:H], np.ogrid[:W]
    dist   = np.sqrt((ys - cy) ** 2 + (xs.reshape(1, -1) - cx) ** 2)

    r_dc   = max(3, min(H, W) // 30)
    r_lf   = min(H, W) / 8
    r_mf   = min(H, W) / 4
    r_hf   = min(H, W) / 2

    lf_mask  = (dist >  r_dc) & (dist <= r_lf)
    mf_mask  = (dist >  r_lf) & (dist <= r_mf)
    hf_mask  = (dist >  r_mf) & (dist <= r_hf)
    all_mask = (dist >  r_dc) & (dist <= r_hf)

    log_pwr = np.log10(power)
    lf_mean = float(np.mean(log_pwr[lf_mask]))
    mf_mean = float(np.mean(log_pwr[mf_mask]))
    hf_mean = float(np.mean(log_pwr[hf_mask]))

    # Signal 1: spectral roll-off
    rolloff      = (lf_mean - hf_mean) / (abs(lf_mean) + 1e-9)
    rolloff_prob = float(np.clip((rolloff - 0.15) / 0.20, 0.0, 1.0))

    # Signal 2: mid-frequency anomaly
    mid_anomaly  = abs(mf_mean - (lf_mean + hf_mean) / 2.0) / (abs(lf_mean) + 1e-9)
    anomaly_prob = float(np.clip(mid_anomaly * 3.0, 0.0, 1.0))

    # Signal 3: spectral entropy
    p_vals   = power[all_mask]
    p_norm   = p_vals / (p_vals.sum() + 1e-10)
    spec_ent = float(-np.sum(p_norm * np.log(p_norm + 1e-10)))
    norm_ent = spec_ent / (float(np.log(p_norm.size)) + 1e-9)
    ent_prob = float(np.clip(1.0 - norm_ent * 1.2, 0.0, 1.0))

    # Signal 4: peak prominence
    lp_vals   = log_pwr[all_mask]
    peak_prom = (float(np.max(lp_vals)) - float(np.mean(lp_vals))) / \
                (abs(float(np.mean(lp_vals))) + 1e-9)
    peak_prob = float(np.clip((peak_prom - 0.5) / 1.0, 0.0, 1.0))

    ai_sub = float(np.mean([rolloff_prob, anomaly_prob, ent_prob, peak_prob]))

    fdict: Dict[str, Any] = {
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
    return ai_sub, label, fdict


# ---------------------------------------------------------------------------
# Bayer-pattern inter-channel correlation  (supporting signal)
# ---------------------------------------------------------------------------

def _score_bayer_correlation(img_rgb: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
    """
    Bayer-pattern inter-channel correlation heuristic.

    Real camera sensors use a Bayer colour filter array (RGGB); the
    demosaicing step (bilinear / AHD / VNG interpolation) introduces
    well-characterised cross-channel dependencies:

      R↔G correlation  ~0.80–0.93  (green is densest in Bayer grid)
      G↔B correlation  ~0.78–0.92
      R↔B correlation  ~0.68–0.88  (lowest — R and B never co-located)
      std across patches  ~0.05–0.18  (moderate spatial variation)

    Diffusion / GAN models synthesise all colour channels simultaneously
    from the same latent, producing two detectable anomalies:

      1. Uniformity  — patch correlations are nearly identical across the
         frame (std < 0.03) because there is no real demosaicing noise.
      2. Magnitude   — smooth AI outputs drive all three correlations
         toward 0.95+ (over-smooth, unphysically uniform).

    Score interpretation: higher value → more AI-like.

    Returns (ai_sub_prob 0-1, description, features_dict).
    """
    H, W, _ = img_rgb.shape
    patch    = 16

    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    rg_corrs: List[float] = []
    gb_corrs: List[float] = []
    rb_corrs: List[float] = []

    for i in range(0, H - patch, patch):
        for j in range(0, W - patch, patch):
            pr = r[i:i + patch, j:j + patch].flatten()
            pg = g[i:i + patch, j:j + patch].flatten()
            pb = b[i:i + patch, j:j + patch].flatten()

            # Skip near-constant patches to avoid numerical instability
            if np.std(pr) < 1e-6 or np.std(pg) < 1e-6 or np.std(pb) < 1e-6:
                continue

            rg = float(np.corrcoef(pr, pg)[0, 1])
            gb = float(np.corrcoef(pg, pb)[0, 1])
            rb = float(np.corrcoef(pr, pb)[0, 1])

            if np.isfinite(rg):
                rg_corrs.append(rg)
            if np.isfinite(gb):
                gb_corrs.append(gb)
            if np.isfinite(rb):
                rb_corrs.append(rb)

    if len(rg_corrs) < 4:
        fdict_out: Dict[str, Any] = {"bayer_ai_sub": 0.5}
        return 0.5, "bayer_correlation: insufficient patches", fdict_out

    rg_mean = float(np.mean(rg_corrs))
    gb_mean = float(np.mean(gb_corrs))
    rb_mean = float(np.mean(rb_corrs))
    rg_std  = float(np.std(rg_corrs))
    gb_std  = float(np.std(gb_corrs))
    rb_std  = float(np.std(rb_corrs))

    mean_corr = (rg_mean + gb_mean + rb_mean) / 3.0
    mean_std  = (rg_std  + gb_std  + rb_std)  / 3.0

    # Signal 1: uniformity anomaly
    #   Real cameras: mean_std ~ 0.05–0.18  → uniformity_prob → 0
    #   AI models:    mean_std < 0.03       → uniformity_prob → 1
    uniformity_prob = float(np.clip(1.0 - mean_std / 0.06, 0.0, 1.0))

    # Signal 2: magnitude anomaly (over-smooth AI → very high correlations)
    #   Real cameras: mean_corr ~ 0.75–0.92 → magnitude_prob → 0
    #   AI models:    mean_corr > 0.95       → magnitude_prob → 1
    magnitude_prob = float(np.clip((mean_corr - 0.92) / 0.07, 0.0, 1.0))

    ai_sub = float(np.mean([uniformity_prob, magnitude_prob]))

    fdict_out = {
        "bayer_rg_mean":          round(rg_mean,          4),
        "bayer_gb_mean":          round(gb_mean,          4),
        "bayer_rb_mean":          round(rb_mean,          4),
        "bayer_rg_std":           round(rg_std,           4),
        "bayer_gb_std":           round(gb_std,           4),
        "bayer_rb_std":           round(rb_std,           4),
        "bayer_mean_corr":        round(mean_corr,        4),
        "bayer_mean_std":         round(mean_std,         4),
        "bayer_uniformity_prob":  round(uniformity_prob,  3),
        "bayer_magnitude_prob":   round(magnitude_prob,   3),
        "bayer_ai_sub":           round(ai_sub,           3),
    }
    label = (
        f"bayer_correlation: RG={rg_mean:.3f}±{rg_std:.3f}, "
        f"GB={gb_mean:.3f}±{gb_std:.3f}, RB={rb_mean:.3f}±{rb_std:.3f} "
        f"(uniformity={uniformity_prob:.2f}, magnitude={magnitude_prob:.2f}) "
        f"→ ai_sub={ai_sub:.2f}"
    )
    return ai_sub, label, fdict_out


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
        ai_prob         – [0, 1]
        confidence_band – ± uncertainty radius
        feature_vector  – all numeric sub-scores as a flat list
        features_dict   – structured dict of all signals
        features        – human-readable descriptions
        method          – what classifier(s) ran
    """
    try:
        img, img_rgb, img_gray = load_image_rgb(image_bytes)
        img_hsv  = np.array(img.convert("HSV"), dtype=np.float32)
    except Exception as exc:
        return AIArtifactResult(
            ai_prob=0.5,
            confidence_band=0.5,
            features=[f"Could not decode image: {exc}"],
            method="error",
        )

    features:   List[str]        = []
    fdict:      Dict[str, Any]   = {}
    sub_scores: List[float]      = []

    # ── 1. Six heuristic sub-scores (always run) ─────────────────────────────
    _heuristics = [
        ("texture",    lambda: _score_texture_smoothness(img_gray)),
        ("colour",     lambda: _score_colour_coherence(img_hsv)),
        ("edge",       lambda: _score_edge_sharpness(img_gray)),
        ("hf_lf",      lambda: _score_hf_lf_ratio(img_gray)),
        ("saturation", lambda: _score_saturation_skew(img_hsv)),
        ("symmetry",   lambda: _score_symmetry(img_gray)),
    ]
    for name, scorer in _heuristics:
        try:
            prob, label = scorer()
            sub_scores.append(prob)
            features.append(f"{label} → ai_sub={prob:.2f}")
            fdict[name] = round(prob, 3)
        except Exception as exc:
            features.append(f"heuristic error ({name}): {exc}")

    heuristic_mean = float(np.mean(sub_scores)) if sub_scores else 0.5

    # ── 2. FreqNet frequency-domain scorer (always run) ──────────────────────
    try:
        freq_prob, freq_label, freq_dict = _score_freqnet(img_gray)
        sub_scores.append(freq_prob)
        features.append(freq_label)
        fdict.update(freq_dict)
    except Exception as exc:
        freq_prob = heuristic_mean
        features.append(f"FreqNet error: {exc}")

    # ── 3. Bayer-pattern inter-channel correlation (always run) ──────────────
    try:
        bayer_prob, bayer_label, bayer_dict = _score_bayer_correlation(img_rgb)
        sub_scores.append(bayer_prob)
        features.append(bayer_label)
        fdict.update(bayer_dict)
    except Exception as exc:
        features.append(f"Bayer correlation error: {exc}")

    # ── 4. HuggingFace trained classifier (primary, when available) ──────────
    hf = _get_image_model("hf_classifier")
    if hf.available and hf.pipe is not None:
        model_prob, model_band, model_label = _run_hf_model(img)
        features.append(model_label)
        fdict["model_ai_prob"]      = round(model_prob, 3)
        fdict["model_id"]           = hf.model_id
        sub_scores.append(model_prob)

        # Model dominates (80%), heuristics are supporting evidence (20%)
        heuristic_blend = _W_HEUR_MEAN * heuristic_mean + _W_FREQNET * freq_prob
        ai_prob         = _W_MODEL * model_prob + _W_HEURISTIC * heuristic_blend

        # Confidence band from model logit margin (tight when model is decisive)
        confidence_band = model_band
        method          = f"model:{hf.model_id}+heuristic"
    else:
        # Heuristic-only fallback
        heuristic_blend = _W_HEUR_MEAN * heuristic_mean + _W_FREQNET * freq_prob
        ai_prob         = heuristic_blend

        # Confidence band from disagreement among sub-scores
        confidence_band = float(np.std(sub_scores)) if len(sub_scores) > 1 else 0.25
        method          = "heuristic+freqnet"

    ai_prob         = float(np.clip(ai_prob,         0.0, 1.0))
    confidence_band = float(np.clip(confidence_band, 0.0, 0.5))

    fdict["ai_prob_final"]   = round(ai_prob,         3)
    fdict["confidence_band"] = round(confidence_band, 3)

    # ── OOD style detection ───────────────────────────────────────────────────
    ood_cat, ood_warning = _detect_ood_style(img)
    if ood_cat:
        fdict["ood_category"] = ood_cat

    return AIArtifactResult(
        ai_prob         = ai_prob,
        confidence_band = confidence_band,
        feature_vector  = [round(s, 4) for s in sub_scores],
        features        = features,
        features_dict   = fdict,
        method          = method,
        ood_warning     = ood_warning,
    )
