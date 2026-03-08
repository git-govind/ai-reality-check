"""
datatypes.py
------------
Shared dataclasses for the Image Authenticity Evaluator pipeline.

Each Result type is the output contract for its corresponding pipeline step.
All fields have safe defaults so callers can pattern-match on .ran or .found
to tell apart "ran and found nothing" from "was skipped entirely".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation_report_base import EvaluationReportBase


# ---------------------------------------------------------------------------
# Per-step result types
# ---------------------------------------------------------------------------

@dataclass
class MetadataResult:
    """Output from metadata_checker.run().

    score            – 0.0 (highly suspicious) … 1.0 (looks authentic)
    flags            – human-readable list of detected anomalies
    raw              – raw key/value metadata extracted from the image
    confidence_band  – ± uncertainty radius around score (fraction of [0,1] range)
                       Narrow when many strong authenticity signals are present;
                       wide when EXIF is absent or sparse.
    editing_penalty        – [0, 100] editing-signal portion of the metadata score.
                             Accumulates only the signals that indicate the image was
                             edited or processed (AI generator, photo editor, missing EXIF
                             on JPEG, timestamp anomaly).  Used as the metadata input to
                             the editing-likelihood formula in image_scoring.
    detected_ai_generator  – name of the AI generator found in metadata (empty string if
                             none detected).  Non-empty value is a definitive confirmation
                             that the image is AI-generated; the pipeline can skip the
                             AI classifier and set ai_prob = 1.0 directly.
    """
    score: float
    flags: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_band: float = 0.15
    editing_penalty: float = 0.0
    detected_ai_generator: str = ""   # non-empty when an AI generator is found in
                                      # EXIF/metadata (e.g. "Midjourney", "DALL-E").
                                      # Enables early-exit optimisation: skip the AI
                                      # classifier and set ai_prob = 1.0 directly.


@dataclass
class PixelForensicsResult:
    """Output from pixel_forensics.run().

    score            – 0.0 (heavy manipulation) … 1.0 (no anomalies)
    artifacts        – list of detected artifact descriptions
    ela_max_diff     – 95th-percentile pixel error from ELA (None if not run)
    fft_peak_ratio   – high-freq / low-freq power ratio (None if not run)
    noise_uniformity – coefficient of variation of local noise patches (None if not run)
    ghost_score      – JPEG Ghost inconsistency score 0–100 (None if not run)
                       0 = all patches share one native quality (pristine)
                       100 = patches maximally disagree (spliced / composite)
    confidence_band  – ± uncertainty radius around score (fraction of [0,1] range)
                       Narrow for JPEG (ELA + Ghost both meaningful);
                       wide for PNG/small images (lossless → ELA less informative).
    """
    score: float
    artifacts: List[str] = field(default_factory=list)
    ela_max_diff: Optional[float] = None
    fft_peak_ratio: Optional[float] = None
    noise_uniformity: Optional[float] = None
    ghost_score: Optional[float] = None
    noise_block_consistency: Optional[float] = None  # CV of 16-block noise stds; low = authentic
    confidence_band: float = 0.20


@dataclass
class AIArtifactResult:
    """Output from ai_artifact_classifier.run().

    ai_prob         – 0.0 (almost certainly real) … 1.0 (almost certainly AI-generated)
    features        – signal descriptions as a list of strings (used by the UI)
    features_dict   – same signals as a structured dict
    method          – e.g. "heuristic+freqnet", "model:umm-maybe/AI-image-detector+heuristic"
    confidence_band – ± uncertainty radius around ai_prob
                      derived from model logit margin when a trained model runs,
                      or from heuristic sub-score std-dev when falling back.
                      Interpretation: true probability is likely within
                      [ai_prob − confidence_band, ai_prob + confidence_band].
    feature_vector  – ordered list of all numeric sub-scores; useful for
                      downstream calibration or ensemble models (None if single
                      heuristic path only).
    """
    ai_prob:         float
    features:        List[str]             = field(default_factory=list)
    features_dict:   Dict[str, Any]        = field(default_factory=dict)
    method:          str                   = "heuristic"
    confidence_band: float                 = 0.0
    feature_vector:  Optional[List[float]] = None
    ood_warning:     str                   = ""  # non-empty when image style is outside
                                                  # the AI detector's training distribution
                                                  # (anime, 3D render, illustration, screenshot)


@dataclass
class ConsistencyResult:
    """Output from image_text_consistency.run().

    score            – 0.0 (inconsistent) … 1.0 (fully consistent)
    issues           – list of detected inconsistency descriptions
    ran              – False when the step was skipped (no caption supplied or no model available)
    confidence_band  – ± uncertainty radius; LLM-based scores carry ~0.12 uncertainty.
    """
    score: float = 0.5
    issues: List[str] = field(default_factory=list)
    ran: bool = False
    confidence_band: float = 0.12


@dataclass
class ReverseSearchResult:
    """Output from reverse_image_search.run().

    found       – True if a matching source image was located online
    similarity  – 0.0 … 1.0 perceptual similarity to closest match
    source_urls – list of URLs where matching content was found
    ran         – False when the step was skipped (no API key configured)
    error       – set when a key was configured but the API call failed
    """
    found: bool = False
    similarity: float = 0.0
    source_urls: List[str] = field(default_factory=list)
    ran: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Final aggregated report
# ---------------------------------------------------------------------------

@dataclass
class ImageEvaluationReport(EvaluationReportBase):
    """Final output from image_scoring.aggregate().

    Inherits from EvaluationReportBase: metadata, scores, issues, evidence, grade.

    authenticity_score  – 0–100  (higher = more likely to be an unedited real photo)
    authenticity_band   – ± points uncertainty on authenticity_score
    ai_likelihood       – 0–100  (higher = more likely AI-generated)
    ai_likelihood_band  – ± percentage-point uncertainty on ai_likelihood
    editing_likelihood  – 0–100  (higher = more likely manually edited/manipulated)
    editing_likelihood_band – ± percentage-point uncertainty on editing_likelihood
    grade               – letter grade (inherited from base; set by aggregate())
    summary             – one-sentence human-readable verdict
    evidence            – structured dict grouping signals (inherited from base)
    top_signals         – top 3 strongest anomaly signals for the UI summary panel
    """
    authenticity_score: float = 0.0
    ai_likelihood: float = 0.0
    editing_likelihood: float = 0.0
    authenticity_band: float = 0.0
    ai_likelihood_band: float = 0.0
    editing_likelihood_band: float = 0.0
    # grade and evidence are inherited from EvaluationReportBase
    summary: str = ""
    top_signals: List[str] = field(default_factory=list)

    def grade_label(self) -> str:
        """Recompute grade from authenticity_score (A/B/C/D/F)."""
        s = self.authenticity_score
        if s >= 80:
            return "A"
        if s >= 65:
            return "B"
        if s >= 50:
            return "C"
        if s >= 35:
            return "D"
        return "F"
