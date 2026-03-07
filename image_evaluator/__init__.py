"""
image_evaluator
---------------
Standalone Image Authenticity Evaluator module.

Public API
----------
    evaluate_image(image_bytes, caption=None) -> ImageEvaluationReport

This module is fully independent of the text-evaluation pipeline.
"""

from .evaluate_image import evaluate_image
from .datatypes import (
    AIArtifactResult,
    ConsistencyResult,
    ImageEvaluationReport,
    MetadataResult,
    PixelForensicsResult,
    ReverseSearchResult,
)

__all__ = [
    "evaluate_image",
    "ImageEvaluationReport",
    "MetadataResult",
    "PixelForensicsResult",
    "AIArtifactResult",
    "ConsistencyResult",
    "ReverseSearchResult",
]
