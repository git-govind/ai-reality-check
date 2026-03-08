"""
models/embeddings_registry.py
------------------------------
Global lazy-loading registry for sentence / multi-modal embedding models
used by ``image_evaluator/image_text_consistency.py``.

Models
------
  ``"clip-ViT-B-32"``
      ``sentence_transformers.SentenceTransformer("clip-ViT-B-32")`` — the
      CLIP model wrapped for convenient encode() calls on both PIL images
      and text strings.  Disabled by ``SKIP_CLIP=1`` env-var.

Public API
----------
  get_model(name: str) -> EmbeddingModelEntry
      Return the cached entry, loading on first call.  Thread-safe.

  is_available(name: str) -> bool
      ``True`` when the named model loaded successfully.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

_lock:    threading.Lock                    = threading.Lock()
_registry: dict[str, "EmbeddingModelEntry"] = {}
_loaded:   set[str]                         = set()


# ---------------------------------------------------------------------------
# Typed model entry
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingModelEntry:
    """Result of loading a SentenceTransformer embedding model."""
    model:     Any  = None   # SentenceTransformer instance or None
    available: bool = False  # True when loading succeeded


# ---------------------------------------------------------------------------
# Loader function (called at most once per model per process)
# ---------------------------------------------------------------------------

def _load_clip_st() -> EmbeddingModelEntry:
    entry = EmbeddingModelEntry()

    if os.getenv("SKIP_CLIP", "").strip() == "1":
        return entry

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        entry.model     = SentenceTransformer("clip-ViT-B-32")
        entry.available = True
    except Exception:
        pass   # graceful degradation — heuristic fallback activates

    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(name: str) -> EmbeddingModelEntry:
    """Return (or lazily load) the named embedding model.

    Parameters
    ----------
    name : ``"clip-ViT-B-32"``

    Returns
    -------
    EmbeddingModelEntry

    Raises
    ------
    KeyError
        When *name* is not a recognised model key.
    """
    if name not in _loaded:
        with _lock:
            if name not in _loaded:   # double-checked locking
                if name == "clip-ViT-B-32":
                    _registry[name] = _load_clip_st()
                else:
                    raise KeyError(
                        f"embeddings_registry: unknown model {name!r}. "
                        f"Valid keys: 'clip-ViT-B-32'."
                    )
                _loaded.add(name)
    return _registry[name]


def is_available(name: str) -> bool:
    """Return ``True`` when *name* loaded successfully."""
    return get_model(name).available
