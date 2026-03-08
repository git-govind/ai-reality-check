"""
config_loader.py
----------------
Lightweight config reader for the AI Reality Check pipeline.

The three YAML files in the ``config/`` directory are loaded lazily on the
first access and cached for the lifetime of the process — there is no
file I/O after the first call to any of the three public functions.

Usage
-----
    from config_loader import get_feature, get_threshold, get_weight

    if get_feature("image.hf_model_enabled"):
        ...

    w = get_weight("text.scoring.accuracy")    # → 0.30
    t = get_threshold("text.grade.a")          # → 90

Dotted paths
------------
Keys are dotted strings that map to the YAML hierarchy::

    "image.scoring_weights.photo.metadata"
    →  config["image"]["scoring_weights"]["photo"]["metadata"]

A ``KeyError`` with the full dotted path is raised when a key is not found.

Dependencies
------------
PyYAML (``pip install pyyaml``).  Already listed in requirements.txt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False

# Resolve config/ relative to this file so it works from any working directory.
_CONFIG_DIR: Path = Path(__file__).parent / "config"

# In-memory caches, populated on first access.
_features:   dict[str, Any] | None = None
_thresholds: dict[str, Any] | None = None
_weights:    dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(filename: str) -> dict[str, Any]:
    """Read and parse *filename* from the config directory."""
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required by config_loader.  "
            "Install it with:  pip install pyyaml"
        )
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected location: {_CONFIG_DIR / filename}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(store: dict[str, Any], key: str) -> Any:
    """
    Traverse *store* using a dotted *key* path.

    Raises ``KeyError`` with the full path when any segment is missing.
    """
    parts = key.split(".")
    node  = store
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Config key not found: {key!r}")
        node = node[part]
    return node


def _features_store() -> dict[str, Any]:
    global _features
    if _features is None:
        _features = _load_yaml("features.yaml")
    return _features


def _thresholds_store() -> dict[str, Any]:
    global _thresholds
    if _thresholds is None:
        _thresholds = _load_yaml("thresholds.yaml")
    return _thresholds


def _weights_store() -> dict[str, Any]:
    global _weights
    if _weights is None:
        _weights = _load_yaml("weights.yaml")
    return _weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature(key: str) -> Any:
    """
    Return the value of a feature flag from ``config/features.yaml``.

    Parameters
    ----------
    key : str
        Dotted path, e.g. ``"image.hf_model_enabled"``.

    Returns
    -------
    Any
        Typically ``bool``.

    Raises
    ------
    KeyError
        When the dotted path does not exist.
    """
    return _resolve(_features_store(), key)


def get_threshold(key: str) -> Any:
    """
    Return a threshold value from ``config/thresholds.yaml``.

    Parameters
    ----------
    key : str
        Dotted path, e.g. ``"text.grade.a"``.

    Returns
    -------
    Any
        Typically ``float`` or ``int``.

    Raises
    ------
    KeyError
        When the dotted path does not exist.
    """
    return _resolve(_thresholds_store(), key)


def get_weight(key: str) -> Any:
    """
    Return a weight value from ``config/weights.yaml``.

    Parameters
    ----------
    key : str
        Dotted path, e.g. ``"text.scoring.accuracy"``.

    Returns
    -------
    Any
        Typically ``float``.

    Raises
    ------
    KeyError
        When the dotted path does not exist.
    """
    return _resolve(_weights_store(), key)
