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


# ---------------------------------------------------------------------------
# Per-step result types
# ---------------------------------------------------------------------------

@dataclass
class MetadataResult:
    """Output from metadata_checker.run().

    score   – 0.0 (highly suspicious) … 1.0 (looks authentic)
    flags   – human-readable list of detected anomalies
    raw     – raw key/value metadata extracted from the image
    """
    score: float
    flags: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PixelForensicsResult:
    """Output from pixel_forensics.run().

    score            – 0.0 (heavy manipulation) … 1.0 (no anomalies)
    artifacts        – list of detected artifact descriptions
    ela_max_diff     – 95th-percentile pixel error from ELA (None if not run)
    fft_peak_ratio   – high-freq / low-freq power ratio (None if not run)
    noise_uniformity – coefficient of variation of local noise patches (None if not run)
    """
    score: float
    artifacts: List[str] = field(default_factory=list)
    ela_max_diff: Optional[float] = None
    fft_peak_ratio: Optional[float] = None
    noise_uniformity: Optional[float] = None


@dataclass
class AIArtifactResult:
    """Output from ai_artifact_classifier.run().

    ai_prob       – 0.0 (almost certainly real) … 1.0 (almost certainly AI-generated)
    features      – signal descriptions as a list of strings (used by the UI)
    features_dict – same signals as a structured dict  (spec §3 requirement)
    method        – "heuristic", "heuristic+freqnet", or "heuristic+model"
    """
    ai_prob:       float
    features:      List[str]         = field(default_factory=list)
    features_dict: Dict[str, Any]    = field(default_factory=dict)
    method:        str               = "heuristic"


@dataclass
class ConsistencyResult:
    """Output from image_text_consistency.run().

    score  – 0.0 (inconsistent) … 1.0 (fully consistent)
    issues – list of detected inconsistency descriptions
    ran    – False when the step was skipped (no caption supplied or no model available)
    """
    score: float = 0.5
    issues: List[str] = field(default_factory=list)
    ran: bool = False


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
class ImageEvaluationReport:
    """Final output from image_scoring.aggregate().

    authenticity_score  – 0–100  (higher = more likely to be an unedited real photo)
    ai_likelihood       – 0–100  (higher = more likely AI-generated)
    editing_likelihood  – 0–100  (higher = more likely manually edited/manipulated)
    grade               – letter grade derived from authenticity_score
    summary             – one-sentence human-readable verdict
    evidence            – structured dict grouping signals from each pipeline step
    """
    authenticity_score: float
    ai_likelihood: float
    editing_likelihood: float
    grade: str = "?"
    summary: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

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
