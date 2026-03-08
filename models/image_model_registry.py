"""
models/image_model_registry.py
-------------------------------
Global lazy-loading registry for heavy image classification models used by
``image_evaluator/ai_artifact_classifier.py``.

Models
------
  ``"hf_classifier"``
      HuggingFace ``transformers`` image-classification pipeline.
      Tries ``umm-maybe/AI-image-detector`` first, falls back to
      ``Organika/sdxl-detector``.  Device is always CPU (``device=-1``).
      Disabled by ``SKIP_HF_MODEL=1`` env-var or ``image.hf_model_enabled: false``.

  ``"clip_ood"``
      ``openai/clip-vit-base-patch32`` raw CLIPModel + CLIPProcessor used for
      out-of-distribution style detection (anime, 3D render, illustration,
      screenshot).  Disabled by ``SKIP_CLIP_OOD=1`` or
      ``image.clip_ood_enabled: false``.

Public API
----------
  get_model(name: str) -> HFClassifierEntry | ClipOodEntry
      Return the cached model entry, loading on first call.  Thread-safe.

  is_available(name: str) -> bool
      ``True`` when the named model loaded successfully.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, List, Tuple

_lock:    threading.Lock     = threading.Lock()
_registry: dict[str, Any]   = {}
_loaded:   set[str]         = set()


# ---------------------------------------------------------------------------
# Typed model entries
# ---------------------------------------------------------------------------

@dataclass
class HFClassifierEntry:
    """Result of loading the HuggingFace image-classification pipeline."""
    pipe:      Any  = None   # transformers pipeline or None
    model_id:  str  = ""     # which candidate was loaded
    available: bool = False  # True when loading succeeded


@dataclass
class ClipOodEntry:
    """Result of loading the CLIP OOD-style detector."""
    model:     Any  = None   # CLIPModel or None
    processor: Any  = None   # CLIPProcessor or None
    available: bool = False  # True when loading succeeded


# ---------------------------------------------------------------------------
# Candidate model IDs (tried in order)
# ---------------------------------------------------------------------------

_HF_CANDIDATES: List[Tuple[str, str]] = [
    ("umm-maybe/AI-image-detector", "artificial"),
    ("Organika/sdxl-detector",      "artificial"),
]

_CLIP_OOD_MODEL_ID = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# Loader functions (called at most once per model per process)
# ---------------------------------------------------------------------------

def _load_hf_classifier() -> HFClassifierEntry:
    from config_loader import get_feature

    entry = HFClassifierEntry()
    if (
        os.getenv("SKIP_HF_MODEL", "").strip() == "1"
        or not get_feature("image.hf_model_enabled")
    ):
        return entry

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
    except ImportError:
        return entry

    for model_id, _ in _HF_CANDIDATES:
        try:
            pipe = hf_pipeline(
                "image-classification",
                model=model_id,
                device=-1,       # CPU only — no GPU dependency
                framework="pt",
            )
            entry.pipe      = pipe
            entry.model_id  = model_id
            entry.available = True
            return entry
        except Exception:
            continue   # try next candidate

    return entry


def _load_clip_ood() -> ClipOodEntry:
    from config_loader import get_feature

    entry = ClipOodEntry()
    if (
        os.getenv("SKIP_CLIP_OOD", "").strip() == "1"
        or not get_feature("image.clip_ood_enabled")
    ):
        return entry

    try:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        entry.processor = CLIPProcessor.from_pretrained(_CLIP_OOD_MODEL_ID)
        entry.model     = CLIPModel.from_pretrained(_CLIP_OOD_MODEL_ID)
        entry.available = True
    except Exception:
        pass   # graceful degradation — OOD check is best-effort

    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(name: str) -> Any:
    """Return (or lazily load) the named model entry.

    Parameters
    ----------
    name : ``"hf_classifier"`` | ``"clip_ood"``

    Returns
    -------
    ``HFClassifierEntry`` when *name* is ``"hf_classifier"``.
    ``ClipOodEntry``       when *name* is ``"clip_ood"``.

    Raises
    ------
    KeyError
        When *name* is not a recognised model key.
    """
    if name not in _loaded:
        with _lock:
            if name not in _loaded:   # double-checked locking
                if name == "hf_classifier":
                    _registry[name] = _load_hf_classifier()
                elif name == "clip_ood":
                    _registry[name] = _load_clip_ood()
                else:
                    raise KeyError(
                        f"image_model_registry: unknown model {name!r}. "
                        f"Valid keys: 'hf_classifier', 'clip_ood'."
                    )
                _loaded.add(name)
    return _registry[name]


def is_available(name: str) -> bool:
    """Return ``True`` when *name* loaded successfully."""
    return get_model(name).available
