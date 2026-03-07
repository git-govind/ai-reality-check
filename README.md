# AI Reality Check

An open-source tool that evaluates the **correctness, safety, and reliability** of AI model responses using a multi-layer evaluation engine.

---

## Architecture

```
User UI (Streamlit)
       │
       ▼
LLM Response Engine  ──►  Ollama (Llama 3 / Mistral / Phi-3)
       │
       ▼
Evaluation Engine
  ├── Factual Checker       (Wikipedia API cross-reference)
  ├── Consistency Checker   (LLM self-critique + heuristics)
  ├── Bias & Safety Checker (rule-based + LLM)
  └── Clarity Scorer        (completeness + readability)
       │
       ▼
Scoring Engine  →  Accuracy / Safety / Bias / Clarity / Confidence
       │
       ▼
Results Display + Dashboard
```

---

## Features

| Feature | Status |
|---------|--------|
| Prompt → AI Response | ✅ |
| Factual claim verification (Wikipedia) | ✅ |
| LLM self-critique (consistency) | ✅ |
| Bias & safety rule engine | ✅ |
| Completeness & clarity scoring | ✅ |
| Weighted confidence score | ✅ |
| Side-by-side model comparison | ✅ |
| Export evaluation as JSON | ✅ |
| Stress Test Mode | ✅ |
| Reliability Dashboard | ✅ |

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
# Edit .env if Ollama runs on a non-default port
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
├── app.py                         # Main Streamlit UI
├── pages/
│   └── 1_Dashboard.py             # Trend & reliability dashboard
├── src/
│   ├── llm/
│   │   └── response_generator.py  # LLM query interface (Ollama)
│   ├── evaluation/
│   │   ├── factual_checker.py     # Wikipedia-based fact verification
│   │   ├── consistency_checker.py # LLM + heuristic consistency
│   │   ├── bias_safety_checker.py # Rule + LLM bias/safety detection
│   │   └── clarity_scorer.py      # Completeness & readability
│   ├── scoring/
│   │   └── scoring_engine.py      # Aggregate weighted scoring
│   ├── retrieval/
│   │   └── wikipedia_retriever.py # Wikipedia API wrapper
│   └── utils/
│       └── pipeline.py            # Orchestration helper
├── data/                          # Local cache / ChromaDB
├── requirements.txt
└── .env.example
```

---

## Scoring Weights

| Dimension | Weight |
|-----------|--------|
| Accuracy (factual) | 30% |
| Consistency | 20% |
| Safety | 20% |
| Bias | 10% |
| Clarity | 10% |
| Completeness | 10% |

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
