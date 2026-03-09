"""
LLM Response Generator
Supports three backends: Ollama (local), OpenAI API, and Demo (mock) mode.

Backend priority:
  1. OPENAI_API_KEY set  →  OpenAI backend
  2. Ollama reachable    →  Ollama backend
  3. fallback            →  Demo / mock mode
"""
from __future__ import annotations

import os
import textwrap
from typing import Iterator

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── Model registries ───────────────────────────────────────────────────────────

OLLAMA_MODELS: dict[str, str] = {
    "Llama 3 (8B)":    "llama3",
    "Llama 3 (70B)":   "llama3:70b",
    "Mistral 7B":      "mistral",
    "Phi-3 Mini":      "phi3",
    "Phi-3 Medium":    "phi3:medium",
    "Gemma 2 (2B)":    "gemma2:2b",
    "Gemma 2 (9B)":    "gemma2",
}

OPENAI_MODELS: dict[str, str] = {
    "GPT-4o":           "gpt-4o",
    "GPT-4o Mini":      "gpt-4o-mini",
    "GPT-4 Turbo":      "gpt-4-turbo",
    "GPT-3.5 Turbo":    "gpt-3.5-turbo",
}

DEMO_MODELS: dict[str, str] = {
    "Demo Model (mock)": "demo",
}

# Unified registry used by the rest of the app
MODEL_REGISTRY: dict[str, str] = {**OLLAMA_MODELS, **OPENAI_MODELS, **DEMO_MODELS}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless AI assistant. "
    "Answer questions accurately and concisely."
)

# ── Demo responses ─────────────────────────────────────────────────────────────

_DEMO_RESPONSES = [
    textwrap.dedent("""\
        The water cycle, also known as the hydrological cycle, describes the continuous movement
        of water through Earth's systems. Water evaporates from oceans and lakes, rises as water
        vapor, condenses into clouds, and falls as precipitation (rain or snow). This process is
        driven by solar energy and gravity. The water cycle is essential for distributing
        freshwater across the planet and regulating climate. Scientists estimate that a single
        water molecule completes a full cycle approximately every 9 days on average, though this
        varies significantly depending on where the molecule resides — glaciers may hold water
        for thousands of years, while atmospheric moisture cycles in about 10 days."""),
    textwrap.dedent("""\
        Photosynthesis is the biological process by which plants, algae, and some bacteria convert
        light energy — usually from the sun — into chemical energy stored as glucose. The overall
        reaction can be summarized as: 6CO2 + 6H2O + light → C6H12O6 + 6O2. This process occurs
        in the chloroplasts, specifically using the pigment chlorophyll which absorbs red and blue
        light most efficiently. Photosynthesis has two stages: the light-dependent reactions
        (which produce ATP and NADPH) and the Calvin cycle (which fixes carbon dioxide into
        glucose). Without photosynthesis, nearly all life on Earth would not exist, as it forms
        the base of most food chains."""),
    textwrap.dedent("""\
        Artificial intelligence (AI) refers to the simulation of human intelligence processes by
        computer systems. These processes include learning (the acquisition of information and
        rules for using the information), reasoning (using rules to reach conclusions), and
        self-correction. AI applications include expert systems, natural language processing (NLP),
        speech recognition, and machine vision. The field was founded by Alan Turing, who proposed
        the famous Turing Test in 1950 as a criterion of intelligence. Modern AI is largely built
        on machine learning and deep learning techniques, where systems learn patterns from large
        datasets rather than following explicitly programmed rules. As of 2024, large language
        models like GPT-4 represent the state of the art in generative AI."""),
    textwrap.dedent("""\
        Climate change refers to long-term shifts in global temperatures and weather patterns.
        While some climate change is natural, since the 1800s human activities — primarily the
        burning of fossil fuels — have been the main driver of climate change. This releases
        greenhouse gases like CO2 and methane into the atmosphere, trapping heat and causing
        global temperatures to rise. The Intergovernmental Panel on Climate Change (IPCC) has
        found that global average temperatures have already increased by approximately 1.1°C
        above pre-industrial levels. Consequences include more frequent extreme weather events,
        rising sea levels, and disruptions to ecosystems. The Paris Agreement aims to limit
        warming to 1.5°C above pre-industrial levels."""),
]

_demo_index = 0


def _get_demo_response(prompt: str) -> str:
    global _demo_index
    # Try to pick a relevant response based on keywords
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ("water", "rain", "ocean", "cycle", "evapor")):
        return _DEMO_RESPONSES[0]
    if any(w in prompt_lower for w in ("photo", "plant", "chloro", "glucose", "sunlight")):
        return _DEMO_RESPONSES[1]
    if any(w in prompt_lower for w in ("ai", "artificial", "machine", "neural", "language model")):
        return _DEMO_RESPONSES[2]
    if any(w in prompt_lower for w in ("climate", "co2", "carbon", "warming", "fossil")):
        return _DEMO_RESPONSES[3]
    # Round-robin fallback
    resp = _DEMO_RESPONSES[_demo_index % len(_DEMO_RESPONSES)]
    _demo_index += 1
    return resp


# ── Backend detection ──────────────────────────────────────────────────────────

def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def detect_backend() -> str:
    """Return active backend: 'openai' | 'ollama' | 'demo'."""
    if OPENAI_API_KEY:
        return "openai"
    if _ollama_reachable():
        return "ollama"
    return "demo"


def list_available_models() -> list[str]:
    backend = detect_backend()
    if backend == "openai":
        return list(OPENAI_MODELS.keys())
    if backend == "ollama":
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            pulled = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
            available = [
                display for display, tag in OLLAMA_MODELS.items()
                if tag.split(":")[0] in pulled
            ]
            return available if available else list(OLLAMA_MODELS.keys())
        except Exception:
            return list(OLLAMA_MODELS.keys())
    # demo
    return list(DEMO_MODELS.keys())


# ── Main generation function ───────────────────────────────────────────────────

def generate_response(
    prompt: str,
    model_display_name: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
) -> str | Iterator[str]:
    backend = detect_backend()

    if backend == "openai":
        return _openai_response(prompt, model_display_name, system_prompt, temperature, max_tokens)

    if backend == "ollama":
        try:
            return _ollama_response(prompt, model_display_name, system_prompt, temperature, max_tokens, stream)
        except Exception as exc:
            # Re-raise with the cleaned-up message so the UI can display it clearly.
            # The caller (page code) should catch this and show st.error() rather than
            # letting Streamlit render a raw traceback.
            raise RuntimeError(str(exc)) from exc

    # demo mode
    return _get_demo_response(prompt)


# ── Ollama backend ─────────────────────────────────────────────────────────────

def _ollama_response(
    prompt: str,
    model_display_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> str | Iterator[str]:
    model_tag = OLLAMA_MODELS.get(model_display_name, "mistral")
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": stream,
    }
    if stream:
        return _ollama_stream(payload)
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    if not resp.ok:
        _raise_ollama_error(resp)
    return resp.json()["message"]["content"]


def _raise_ollama_error(resp: requests.Response) -> None:
    """Extract Ollama's error detail and raise a descriptive RuntimeError."""
    try:
        detail = resp.json().get("error", resp.text) or resp.text
    except Exception:
        detail = resp.text or f"HTTP {resp.status_code}"
    # Suggest the fix for the most common cause (model not pulled)
    hint = ""
    if "not found" in detail.lower() or "pull" in detail.lower():
        model = detail.split("'")[1] if "'" in detail else "the model"
        hint = f" — run: ollama pull {model}"
    raise RuntimeError(f"Ollama error: {detail}{hint}")


def _ollama_stream(payload: dict) -> Iterator[str]:
    import json
    with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=120) as resp:
        if not resp.ok:
            _raise_ollama_error(resp)
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break


# ── OpenAI backend ─────────────────────────────────────────────────────────────

def _openai_response(
    prompt: str,
    model_display_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    model_id = OPENAI_MODELS.get(model_display_name, "gpt-4o-mini")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Critique (used by consistency checker) ─────────────────────────────────────

def generate_critique(
    original_prompt: str,
    ai_response: str,
    model_display_name: str,
) -> str:
    critique_prompt = (
        f"You are a critical reviewer. A user asked:\n\n"
        f'"""{original_prompt}"""\n\n'
        f"An AI responded:\n\n"
        f'"""{ai_response}"""\n\n'
        "Carefully analyze the response. Identify:\n"
        "1. Any factual errors or unsupported claims\n"
        "2. Logical inconsistencies or contradictions\n"
        "3. Missing important information\n"
        "4. Any biased or unsafe statements\n\n"
        "Be specific and concise. Rate overall quality 0-10."
    )
    backend = detect_backend()
    if backend == "demo":
        return (
            "Demo mode: LLM critique unavailable. "
            "The response appears factually reasonable with no major inconsistencies detected. "
            "Quality Score: 7/10"
        )
    return generate_response(  # type: ignore[return-value]
        critique_prompt,
        model_display_name,
        system_prompt="You are an expert AI evaluator. Be precise and critical.",
        temperature=0.3,
    )
