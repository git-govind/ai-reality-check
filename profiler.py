"""
profiler.py
-----------
Lightweight per-step wall-clock timer for the AI Reality Check pipeline.

Each pipeline call gets an isolated timer state via ``threading.local()``, so
concurrent Streamlit sessions never mix their timings.

Public API
----------
  reset()               – clear all state for the current thread (call at
                          the top of each pipeline run)
  start_timer(label)    – record wall-clock start for *label*
  end_timer(label)      – stop the timer; returns elapsed milliseconds
  get_timings()         – return ``{label: ms, …, "total_ms": ms}`` copy

Usage
-----
    from profiler import reset, start_timer, end_timer, get_timings

    reset()
    start_timer("step_a")
    result = do_step_a()
    end_timer("step_a")

    start_timer("step_b")
    result = do_step_b()
    end_timer("step_b")

    timings = get_timings()
    # {"step_a": 42.3, "step_b": 17.1, "total_ms": 59.4}
"""

from __future__ import annotations

import threading
import time
from typing import Dict

_local: threading.local = threading.local()


# ---------------------------------------------------------------------------
# Internal state accessor
# ---------------------------------------------------------------------------

def _state() -> dict:
    """Return the per-thread profiler state dict, initialising on first access."""
    if not hasattr(_local, "starts"):
        _local.starts: Dict[str, float] = {}
        _local.timings: Dict[str, float] = {}
    return _local.__dict__


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset() -> None:
    """Clear all recorded timings for the current thread.

    Call this at the top of each pipeline run so that timings from a previous
    evaluation do not bleed into the current one.
    """
    s = _state()
    s["starts"]  = {}
    s["timings"] = {}


def start_timer(label: str) -> None:
    """Record the wall-clock start time for *label*.

    If *label* is already running it is silently restarted.
    """
    _state()["starts"][label] = time.perf_counter()


def end_timer(label: str) -> float:
    """Stop the timer for *label* and return the elapsed time in milliseconds.

    The result is stored internally and included in the next :func:`get_timings`
    call.  If :func:`start_timer` was never called for *label*, returns ``0.0``
    and stores nothing.
    """
    s = _state()
    t0 = s["starts"].pop(label, None)
    if t0 is None:
        return 0.0
    elapsed_ms = round((time.perf_counter() - t0) * 1_000, 1)
    s["timings"][label] = elapsed_ms
    return elapsed_ms


def get_timings() -> Dict[str, float]:
    """Return a snapshot of all completed timings for the current thread.

    The returned dict maps each label to its elapsed milliseconds, plus a
    synthetic ``"total_ms"`` key that sums all recorded steps.  The dict is
    a copy; mutating it does not affect the profiler state.
    """
    s  = _state()
    result: Dict[str, float] = dict(s.get("timings", {}))
    result["total_ms"] = round(
        sum(v for k, v in result.items() if k != "total_ms"), 1
    )
    return result
