"""
models/
-------
Global model registries for the AI Reality Check pipeline.

  llm_registry          – LLM client cache (Ollama / OpenAI / demo)
  image_model_registry  – HuggingFace image classifier + CLIP OOD detector
  embeddings_registry   – CLIP SentenceTransformer for image-text consistency

All registries follow the same contract:

  get_model(name: str) -> ModelEntry
      Return a cached entry, loading the model on the first call.
      Thread-safe: concurrent calls for the same name block until loading
      completes; only one load attempt is ever made per name.

  is_available(name: str) -> bool
      Return True when the named model loaded successfully.
"""
