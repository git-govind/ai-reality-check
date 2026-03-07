# AI Reality Check

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
       │                                            │   EXIF · software · timestamps
       ▼                                            │   score: 100 − penalties + bonus
Evaluation Engine                                  ├── Pixel Forensics
  ├── Factual Checker (DuckDB + Wikipedia)          │   ELA · noise · FFT
  ├── Consistency Checker (heuristic + LLM)         │   pixel_score = 100−(ela×0.40
  ├── Bias & Safety Checker (rules + LLM)           │     +noise×0.30+fft×0.30)
  └── Clarity Scorer (completeness + clarity)       ├── AI Artifact Classifier
       │                                            │   6 heuristics + FreqNet
       ▼                                            │   (spectral roll-off, entropy,
Text Scoring Engine                                 │    mid-freq anomaly, peak prom)
  Accuracy / Safety / Bias / Clarity                │   + optional CLIP probe
  Confidence Grade A–F                              ├── Image-Text Consistency
       │                                            │   CLIP penalty scoring
       │                                            │   (or keyword heuristic fallback)
       │                                            └── Reverse Image Search
       │                                                Google Vision / Bing / SerpApi
       │                                                         │
       │                                                         ▼
       │                                            Image Scoring Engine
       │                                              authenticity_score =
       │                                                0.20 × metadata_score
       │                                              + 0.25 × pixel_score
       │                                              + 0.35 × (1 − ai_prob)
       │                                              + 0.10 × consistency_score
       │                                              + 0.10 × reverse_search_score
       │                                              Grade A–F · ai_likelihood
       │                                              editing_likelihood
       └──────────────────────────────────────────────────┘
                                    │
                                    ▼
                         Dashboard (pages/1_Dashboard.py)
                           📝 Text Evaluations tab
                           🖼 Image Evaluations tab
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
| Weighted confidence score | Text | ✅ |
| Side-by-side model comparison | Text | ✅ |
| Export evaluation as JSON | Text | ✅ |
| Stress Test Mode | Text | ✅ |
| EXIF metadata analysis | Image | ✅ |
| Error Level Analysis (ELA) | Image | ✅ |
| FFT spectral anomaly detection | Image | ✅ |
| FreqNet-inspired AI artifact classifier | Image | ✅ |
| CLIP image-text consistency check | Image | ✅ |
| Reverse image search (Google / Bing / SerpApi) | Image | ✅ |
| Authenticity grade A–F | Image | ✅ |
| AI likelihood & editing likelihood scores | Image | ✅ |
| Unified Dashboard (text + image history) | Both | ✅ |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- At least one model pulled, e.g. `ollama pull mistral`

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

## Project Structure

```
ai-reality-check/
├── app.py                              # Navigation router (st.navigation)
├── pages/
│   ├── 1_Dashboard.py                  # Tabbed dashboard: text + image history
│   ├── 2_Text_Evaluator.py             # Text evaluation UI  ("AI Reality Check")
│   └── 3_Image_Evaluator.py            # Image evaluation UI ("Image Evaluator")
├── src/
│   ├── llm/
│   │   └── response_generator.py       # LLM query interface (Ollama / OpenAI)
│   ├── evaluation/
│   │   ├── factual_checker.py          # DuckDB-first + Wikipedia fact check
│   │   ├── consistency_checker.py      # Heuristic + LLM consistency
│   │   ├── bias_safety_checker.py      # Rule + LLM bias/safety detection
│   │   └── clarity_scorer.py           # Completeness & readability
│   ├── scoring/
│   │   └── scoring_engine.py           # Weighted text scoring engine
│   ├── retrieval/
│   │   ├── duckdb_retriever.py         # Local DuckDB fact store (57 seed rows)
│   │   └── wikipedia_retriever.py      # Wikipedia API wrapper (LRU-cached)
│   └── utils/
│       └── pipeline.py                 # Orchestration helper
├── image_evaluator/                    # Image pipeline (fully isolated module)
│   ├── __init__.py
│   ├── datatypes.py                    # Typed dataclasses for all step results
│   ├── metadata_checker.py             # EXIF / XMP / ICC metadata analysis
│   ├── pixel_forensics.py              # ELA · noise residual · FFT · JPEG quant
│   ├── ai_artifact_classifier.py       # 6 heuristics + FreqNet + optional CLIP
│   ├── image_text_consistency.py       # CLIP penalty scoring + keyword fallback
│   ├── reverse_image_search.py         # Google Vision / Bing / SerpApi backends
│   ├── image_scoring.py                # Weighted aggregation → ImageEvaluationReport
│   └── evaluate_image.py               # Public entry-point: evaluate_image()
├── data/                               # Local cache / DuckDB facts.db
├── requirements.txt
└── .env.example
```

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

### Image Pipeline

| Component | Weight | Notes |
|-----------|--------|-------|
| Metadata score | 20% | EXIF penalties / valid-camera bonus |
| Pixel forensics | 25% | ELA × 0.40 + noise × 0.30 + FFT × 0.30 |
| AI artifact (1 − ai_prob) | 35% | FreqNet + 6 heuristics + optional CLIP |
| Image-text consistency | 10% | Skipped & redistributed if no caption |
| Reverse image search | 10% | Skipped & redistributed if no API key |

#### Reverse search similarity tiers
| Result | Score |
|--------|-------|
| Exact match | 100 |
| High similarity | 80 |
| Partial match | 50 |
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

## License

MIT
