"""
utils/cache_utils.py
---------------------
Cache-key generation helpers for the AI Reality Check pipeline.

Public API
----------
  make_cache_key(*parts)  – SHA-256 hex-digest key from bytes / str parts.

Notes
-----
Calling ``make_cache_key(a, b)`` is equivalent to
``hashlib.sha256(a + b).hexdigest()`` when both parts are bytes objects,
because SHA-256's incremental ``update()`` is identical to hashing the
concatenation.
"""
from __future__ import annotations

import hashlib


def make_cache_key(*parts: "bytes | str") -> str:
    """
    Compute a 64-character SHA-256 hex-digest cache key from one or more
    *parts*.

    Parameters
    ----------
    *parts : bytes or str
        Any number of byte strings or unicode strings.
        Strings are UTF-8 encoded before hashing.

    Returns
    -------
    str  — 64-character lowercase hex digest.

    Example
    -------
    ::

        key = make_cache_key(image_bytes, caption_text)
        if key in cache:
            return cache[key]
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8") if isinstance(part, str) else part)
    return h.hexdigest()
