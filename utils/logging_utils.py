"""
utils/logging_utils.py
-----------------------
Logging and timing helpers for the AI Reality Check pipeline.

Public API
----------
  timer(label)  – context manager that wraps profiler.start_timer /
                  profiler.end_timer for cleaner pipeline code.

Usage
-----
    from utils.logging_utils import timer

    with timer("factual"):
        result = factual_checker.run(response)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from profiler import end_timer, start_timer


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager that calls :func:`profiler.start_timer` on entry and
    :func:`profiler.end_timer` on exit (even if an exception is raised).

    Parameters
    ----------
    label : str
        The timer label forwarded to the profiler.  Must match between
        start and end calls (handled automatically by this context manager).

    Example
    -------
    ::

        with timer("metadata"):
            meta_result = metadata_checker.run(image_bytes)
    """
    start_timer(label)
    try:
        yield
    finally:
        end_timer(label)
