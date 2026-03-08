"""
models/llm_registry.py
-----------------------
Global cache of LLM clients for the AI Reality Check text pipeline.

Each unique model display name gets one ``LLMClient`` instance for the
lifetime of the process.  The client wraps the existing
``src.llm.response_generator`` logic and reuses a single
``requests.Session`` for HTTP connection pooling.

Public API
----------
  get_model(name: str) -> LLMClient
      Return (or create) a cached client for *name*.

  LLMClient.generate(prompt, system_prompt, temperature, max_tokens) -> str
      Send a generation request through the configured backend.

  LLMClient.generate_critique(original_prompt, ai_response) -> str
      High-level critique wrapper used by the consistency checker.
"""

from __future__ import annotations

import threading
from typing import Any

import requests

_lock:  threading.Lock        = threading.Lock()
_cache: dict[str, "LLMClient"] = {}


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------

class LLMClient:
    """Stateful wrapper around response_generator with a cached HTTP session."""

    def __init__(self, model_display_name: str) -> None:
        self.model_display_name = model_display_name
        # Persistent session for HTTP keep-alive across consecutive calls.
        self._session: requests.Session = requests.Session()

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send *prompt* to the active LLM backend and return the response."""
        from src.llm.response_generator import DEFAULT_SYSTEM_PROMPT, generate_response

        result = generate_response(
            prompt=prompt,
            model_display_name=self.model_display_name,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(result)

    # ── Critique wrapper (used by consistency checker) ────────────────────────

    def generate_critique(
        self,
        original_prompt: str,
        ai_response: str,
    ) -> str:
        """Run the structured critique prompt and return the critique text."""
        from src.llm.response_generator import generate_critique

        return generate_critique(
            original_prompt=original_prompt,
            ai_response=ai_response,
            model_display_name=self.model_display_name,
        )


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------

def get_model(name: str) -> LLMClient:
    """Return (or create) a cached :class:`LLMClient` for model *name*.

    Parameters
    ----------
    name : str
        A model display name such as ``"GPT-4o Mini"`` or ``"Mistral 7B"``.
        Any string is accepted — the client resolves the backend at call time.

    Returns
    -------
    LLMClient
    """
    if name not in _cache:
        with _lock:
            if name not in _cache:          # double-checked locking
                _cache[name] = LLMClient(name)
    return _cache[name]
