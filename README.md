# VeritasIQ
> Where Intelligence Meets Integrity

**Live app:** [veritasiq.streamlit.app](https://veritasiq.streamlit.app/)

An open-source tool that evaluates the **correctness, safety, and reliability** of AI model responses **and** the **authenticity of images** (AI-generated vs. real), using two fully independent evaluation pipelines.

---

## Architecture

```
app.py  (st.navigation router)
       │
       ├─────────────────────────────────────────────────────────┐
       │                                                         │
       ▼                                                         ▼
TEXT EVALUATION PIPELINE                          IMAGE EVALUATION PIPELINE
(pages/2_Text_Evaluator.py)                       (pages/3_Image_Evaluator.py)
       │                                                         │
       ▼                                                         ▼
LLM Response Engine                               Image Authenticator
  Ollama / OpenAI / Demo                            ├── Metadata Checker
  models/llm_registry.py                            │   EXIF · XMP · software · timestamps
       │                                            │
       │                                            ├── AI Watermark Detector
       │                                            │   metadata tags · PNG text chunks
       │                                            │   OCR hook · floor ≥ 85 % if found
       │                                            │
       ▼                                            ├── Pixel Forensics
Evaluation Engine                                   │   ELA · JPEG Ghost · noise residual
  ├── Factual Checker (DuckDB + Wikipedia)          │   noise block consistency (16-block)
  ├── Consistency Checker (heuristic + LLM)         │   FFT spectral peaks · JPEG quant tables
  ├── Bias & Safety Checker (rules + LLM)           │
  └── Clarity Scorer (completeness + clarity)       ├── AI Artifact Classifier
       │                                            │   HuggingFace binary classifier (primary)
       ▼                                            │   FreqNet spectral scorer (secondary)
Text Scoring Engine                                 │   6 pixel heuristics (tertiary)
  Accuracy / Safety / Bias / Clarity                │
  Confidence Grade A–F                              ├── Image-Text Consistency
  Weights from config/weights.yaml                  │   CLIP cosine similarity
       │                                            │   (or keyword heuristic fallback)
       │                                            │   models/embeddings_registry.py
       │                                            │
       │                                            ├── Reverse Image Search
       │                                            │   Google Vision / Bing / SerpApi
       │                                            │
       │                                            └── Image Scoring Engine
       │                                                  Image-type detection (photo /
       │                                                  illustration / screenshot)
       │                                                  Adaptive weights per image type
       │                                                  authenticity_score  Grade A–F
       │                                                  ai_likelihood · editing_likelihood
       │                                                  Confidence bands per component
       │
       └──────────────────────────────────────────────────┘
                                    │
                                    ▼
                         Dashboard (pages/1_Dashboard.py)
                           📝 Prompt Evaluations tab
                           🖼 Image Evaluations tab
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              utils/           models/          config/
      Shared helper funcs   Model registries   YAML-driven
      text · image ·        llm · image ·      thresholds ·
      scoring · logging ·   embeddings         weights ·
      cache                                    features
```

---

## Features

| Feature | Pipeline | Status |
|---------|----------|--------|
| Prompt → AI Response | Text | ✅ |
| Factual claim verification (DuckDB + Wikipedia) | Text | ✅ |
| LLM self-critique (consistency) | Text | ✅ |
| Bias & safety rule engine | Text | ✅ |
| Completeness & clarity scoring | Text | ✅ |
| Weighted confidence score (YAML-configurable) | Text | ✅ |
| Side-by-side model comparison | Text | ✅ |
| Export evaluation as JSON | Text | ✅ |
| Stress Test Mode | Text | ✅ |
| EXIF / XMP / ICC metadata analysis | Image | ✅ |
| Error Level Analysis (ELA) | Image | ✅ |
| JPEG Ghost multi-quality analysis | Image | ✅ |
| Noise residual & 16-block consistency | Image | ✅ |
| FFT spectral anomaly detection | Image | ✅ |
| JPEG quantisation table check | Image | ✅ |
| HuggingFace AI image detector (primary) | Image | ✅ |
| FreqNet-inspired spectral classifier | Image | ✅ |
| Image-type detection (photo / illustration / screenshot) | Image | ✅ |
| Adaptive scoring weights per image type | Image | ✅ |
| Per-component confidence bands | Image | ✅ |
| ML-gated pixel AI boost (noise-block CV → ai_likelihood) | Image | ✅ |
| Illustration AI gate (lowers gate threshold; includes FFT for PNG/lossless) | Image | ✅ |
| Photo conditional AI gate (no_exif + block_cv ≥ 0.45 or ELA ≥ 20) | Image | ✅ |
| Image-type fix: flat_ratio cap 0.25 (snow / sky / water scene tolerance) | Image | ✅ |
| AI watermark detection (metadata tags, PNG text chunks, OCR hook, invisible-watermark stubs) | Image | ✅ |
| Watermark scoring floor: ai_likelihood ≥ 85 % when AI watermark confirmed | Image | ✅ |
| Power-of-two dimension penalty (AI-typical WxH) | Image | ✅ |
| CLIP image-text consistency check | Image | ✅ |
| Reverse image search (Google / Bing / SerpApi) | Image | ✅ |
| Authenticity grade A–F | Image | ✅ |
| AI likelihood & editing likelihood scores | Image | ✅ |
| Human-readable explanation for every verdict | Both | ✅ |
| Unified Dashboard (text + image history) | Both | ✅ |
| YAML-configurable thresholds, weights & feature flags | Both | ✅ |

---

## Quick Start

> **Try it now:** [veritasiq.streamlit.app](https://veritasiq.streamlit.app/) — no local setup required.
> The hosted app runs the Image Evaluator in full. The Prompt Evaluator requires a connected LLM (see below).

### 1. Prerequisites

- Python 3.10+
- **Prompt Evaluator (text):** requires an LLM backend to generate and evaluate responses.
  - [Ollama](https://ollama.ai) installed locally with at least one model pulled, e.g. `ollama pull mistral` — **or**
  - An `OPENAI_API_KEY` set in `.env` for the OpenAI backend.
  - Without either, the app falls back to **Demo mode** (pre-canned mock responses — no real evaluation).
- **Image Evaluator:** works fully without any LLM or API key.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set OLLAMA_BASE_URL, OPENAI_API_KEY, GOOGLE_API_KEY, etc.
```

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Documentation

Two interactive HTML reference guides are included in the repository root:

| File | Purpose |
|------|---------|
| [`static/architecture.html`](static/architecture.html) | Full system architecture diagram — all modules, data flows, scoring formulas, and config keys for both the text and image pipelines. Open directly in any browser. |
| [`static/pipeline_guide.html`](static/pipeline_guide.html) | Step-by-step pipeline guide — detailed breakdown of every evaluation technique (ELA, JPEG Ghost, AI watermark detection, ML gates, …) with code-level formulas, signal explanations, and output field reference. |

---

## Project Structure

```
ai-reality-check/
├── app.py                              # Navigation router (st.navigation)
├── config_loader.py                    # Loads YAML config files at import time
├── evaluation_report_base.py           # Shared base dataclass for all reports
├── explanation_generator.py            # Generates human-readable verdicts
├── profiler.py                         # Per-step timing utilities
│
├── config/                             # YAML-driven runtime configuration
│   ├── thresholds.yaml                 # Grade cut-offs, ELA/noise/FFT limits
│   ├── weights.yaml                    # Scoring weights (text + image)
│   └── features.yaml                   # Feature flags (debug, LLM checks…)
│
├── models/                             # Lazy-loading model registries
│   ├── llm_registry.py                 # LLM client factory (Ollama / OpenAI)
│   ├── image_model_registry.py         # HuggingFace image model loader
│   └── embeddings_registry.py          # CLIP / sentence-transformers loader
│
├── utils/                              # Shared helper functions
│   ├── text_utils.py                   # word_overlap · split_sentences · parse_llm_score
│   │                                   # extract_issue_bullets
│   ├── image_utils.py                  # load_image_rgb · jpeg_recompress
│   │                                   # coeff_of_variation
│   ├── scoring_utils.py                # clamp · letter_grade · normalize_weights
│   │                                   # weighted_average · score_to_color
│   ├── logging_utils.py                # timer(label) context manager
│   └── cache_utils.py                  # make_cache_key (SHA-256)
│
├── pages/
│   ├── 1_Dashboard.py                  # Tabbed dashboard: text + image history
│   ├── 2_Text_Evaluator.py             # Text evaluation UI  ("VeritasIQ")
│   └── 3_Image_Evaluator.py            # Image evaluation UI ("Image Evaluator")
│
├── src/
│   ├── llm/
│   │   └── response_generator.py       # LLM query interface (Ollama / OpenAI / Demo)
│   ├── evaluation/
│   │   ├── factual_checker.py          # DuckDB-first + Wikipedia fact check
│   │   ├── consistency_checker.py      # Heuristic + LLM consistency
│   │   ├── bias_safety_checker.py      # Rule + LLM bias/safety detection
│   │   └── clarity_scorer.py           # Completeness & readability
│   ├── scoring/
│   │   └── scoring_engine.py           # Weighted text scoring → EvaluationReport
│   ├── retrieval/
│   │   ├── duckdb_retriever.py         # Local DuckDB fact store (57 seed rows)
│   │   └── wikipedia_retriever.py      # Wikipedia API wrapper (LRU-cached)
│   └── utils/
│       └── pipeline.py                 # Text pipeline orchestrator
│
├── image_evaluator/                    # Image pipeline (fully isolated module)
│   ├── __init__.py
│   ├── datatypes.py                    # Typed dataclasses for all step results
│   ├── metadata_checker.py             # EXIF / XMP / ICC metadata analysis
│   ├── pixel_forensics.py              # ELA · JPEG Ghost · noise · FFT · quant
│   ├── ai_artifact_classifier.py       # HuggingFace + FreqNet + 6 heuristics
│   ├── image_text_consistency.py       # CLIP cosine similarity + keyword fallback
│   ├── reverse_image_search.py         # Google Vision / Bing / SerpApi backends
│   ├── image_scoring.py                # Adaptive aggregation → ImageEvaluationReport
│   └── evaluate_image.py               # Public entry-point: evaluate_image()
│
├── data/                               # Local cache / DuckDB facts.db
├── requirements.txt
└── .env.example
```

---

## Configuration

All thresholds, scoring weights, and feature flags live in `config/` and are loaded once at startup via `config_loader.py`. No code changes are needed to tune the evaluation behaviour.

### `config/weights.yaml` — scoring weights

```yaml
text:
  scoring:
    accuracy:     0.30
    consistency:  0.20
    safety:       0.20
    bias:         0.10
    clarity:      0.10
    completeness: 0.10

image:
  scoring_weights:
    photo:        { metadata: 0.20, pixel: 0.25, ai: 0.35, consistency: 0.10, reverse: 0.10 }
    illustration: { metadata: 0.20, pixel: 0.35, ai: 0.20, consistency: 0.10, reverse: 0.15 }
    screenshot:   { metadata: 0.40, pixel: 0.15, ai: 0.10, consistency: 0.20, reverse: 0.15 }
```

### `config/thresholds.yaml` — grade cut-offs and signal limits

Key entries: `text.grade.*`, `image.grade.*`, `image.ela_p95_norm_cap`, `image.noise.*`, `image.fft_norm_cap`, `image.metadata_penalty.*` (includes `ai_dimensions: 15`), `image.ai_pixel_blend.*` (gate_low, gate_high, max_contribution; plus illustration overrides `illus_*` and photo conditional `photo_highcv_*`), `image.top_signals_ai_min_prob`.

### `config/features.yaml` — feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `debug` | `false` | Adds intermediate scores to `report.metadata["debug"]` |
| `text.llm_bias_check` | `true` | Enables LLM-powered bias checker |

---

## Scoring Weights

### Text Pipeline

| Dimension | Weight |
|-----------|--------|
| Accuracy (factual) | 30% |
| Consistency | 20% |
| Safety | 20% |
| Bias | 10% |
| Clarity | 10% |
| Completeness | 10% |

### Image Pipeline (photo — default)

| Component | Weight | Notes |
|-----------|--------|-------|
| Metadata score | 20% | EXIF penalties / valid-camera bonus / −15 power-of-two dimensions |
| Pixel forensics | 25% | ELA · JPEG Ghost · noise · FFT |
| AI artifact (1 − ai_prob) | 35% | HuggingFace + FreqNet + 6 heuristics; ai_prob ML-gated boosted by noise block CV |
| Image-text consistency | 10% | Skipped & redistributed if no caption |
| Reverse image search | 10% | Skipped & redistributed if no API key |

Weights are automatically adjusted for **illustration** and **screenshot** image types. The image type is detected from pixel-level signals: `flat_ratio > 0.30 OR unique_ratio < 0.10 → screenshot`; `noise_std > 2.5 AND flat_ratio < 0.25 → photo`; otherwise `illustration`. The `flat_ratio` cap is 0.25 (not 0.05) to avoid misclassifying photos with large smooth regions — snow, sky, water, fog — as illustrations.

#### ai_likelihood computation

`ai_likelihood` is not simply the raw ML classifier output. It applies a **type-aware, ML-gated pixel forensics boost** driven by noise block consistency (16-block spatial CV). The gate parameters vary by image type and corroborating evidence:

**Standard photo gate** (default):
```
gate      = clip((ai_prob − 0.10) / 0.20, 0, 1)   # 0 when ML confident real
pixel_sig = clip((block_cv − 0.15) / 0.40, 0, 1)  # noise inconsistency signal
ai_prob*  = ai_prob + gate × pixel_sig × 0.50 × (1 − ai_prob)
```

**Photo conditional gate** — activates when `no_exif AND (block_cv ≥ 0.45 OR ela_bad ≥ 20)`:
```
gate      = clip(ai_prob / 0.15, 0, 1)             # opens sooner; gate_low = 0
ai_prob*  = ai_prob + gate × pixel_sig × 0.60 × (1 − ai_prob)
```
Covers photorealistic AI images where the ML model under-detects but pixel forensics show spatially inconsistent noise (`block_cv ≥ 0.45`) or synthetic compression patterns (`ela_bad ≥ 20`).

**Illustration gate** — always applied for illustration image type:
```
gate      = clip(ai_prob / 0.15, 0, 1)             # gate_low = 0 (ML unreliable)
fft_sig   = clip((fft_ratio − 0.30) / 0.25, 0, 1) # FFT included (PNG/lossless)
pixel_sig = (fft_sig + consist_sig) / 2
ai_prob*  = ai_prob + gate × pixel_sig × 0.60 × (1 − ai_prob)
```
ML models trained on photorealistic deepfakes fail on illustration-style AI. FFT peaks are included for illustrations (typically PNG) because JPEG block-DCT compression artifacts are absent.

**FFT exclusion for photos:** FFT spectral peaks are excluded from `pixel_sig` in the photo gate — JPEG 8×8 block-DCT routinely raises FFT ratios in authentic camera photos, making it unreliable as an AI discriminator at the `ai_likelihood` level.

**Watermark floor:** When the AI watermark detector finds a confirmed watermark (metadata generator tag, PNG text chunk, or OCR-detected overlay), `ai_prob*` is clamped to a minimum of **0.85** regardless of the ML classifier or pixel forensics result. Watermark absence is intentionally neutral and never reduces `ai_likelihood`.
```
if watermark.has_watermark:
    ai_prob* = max(ai_prob*, 0.85)
```

#### Pixel forensics sub-scores

| Signal | Weight in pixel score |
|--------|-----------------------|
| ELA (Error Level Analysis) | 30% |
| JPEG Ghost | 20% |
| Noise (patch + 16-block composite) | 25% |
| FFT spectral peaks | 25% |

#### Reverse search similarity tiers

| Result | Score |
|--------|-------|
| Exact match (≥ 0.90) | 100 |
| High similarity (≥ 0.70) | 80 |
| Partial match (≥ 0.30) | 50 |
| No match | 0 |

---

## Supported Models (via Ollama)

| Display Name | Ollama Tag |
|---|---|
| Llama 3 (8B) | `llama3` |
| Llama 3 (70B) | `llama3:70b` |
| Mistral 7B | `mistral` |
| Phi-3 Mini | `phi3` |
| Phi-3 Medium | `phi3:medium` |
| Gemma 2 (2B) | `gemma2:2b` |
| Gemma 2 (9B) | `gemma2` |

---

## Optional API Keys (`.env`)

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM backend |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `GOOGLE_API_KEY` | Google Cloud Vision reverse image search |
| `BING_SEARCH_KEY` | Bing Visual Search reverse image search |
| `SERPAPI_KEY` | SerpApi reverse image search |
| `SKIP_HF_MODEL` | Set to `1` to disable HuggingFace AI image detector |

---

## License

MIT
