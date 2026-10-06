# OmniRAG Studio

**OmniRAG Studio** is an enterprise-grade, interactive Advanced Retrieval-Augmented Generation (RAG) platform with runtime hyperparameter tuning and a real-time 4-tier observability suite.

---

## 🏗️ Modular Architecture

The monolithic engine in `prd.MD` has been refactored into a clean, decoupled production package:

| Module | File | Purpose |
| :--- | :--- | :--- |
| **Configuration & Presets** | [`config.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/config.py) | Dual domain presets (Technical Study Buddy vs. E-Commerce Support), model hyperparameter defaults, and pricing tables. |
| **Multi-Format Ingestion** | [`loaders.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/loaders.py) | Document loaders supporting **PDF** (`pypdf`), **DOCX** (`python-docx`), **TXT/MD**, and raw in-memory streams. |
| **Pluggable Embeddings** | [`embeddings.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/embeddings.py) | **TF-IDF Sparse** (lexical/zero-latency), **Sentence-Transformers** (dense 384d local neural), and **OpenAI** (`text-embedding-3`). |
| **Dynamic Indexer** | [`indexer.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/indexer.py) | Sliding-window chunker with boundary tracking, sub-millisecond dynamic re-indexing, and 2D PCA projection computation. |
| **Metric Retriever** | [`retriever.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/retriever.py) | Vector cosine ranking, threshold cutoff ($\tau$), and query sparsity/similarity distribution metrics. |
| **Grounded Generator** | [`generator.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/generator.py) | Prompt assembly with `[Source X]` citations, refusal safeguards, token economics tracking, and multi-LLM support (Anthropic, OpenAI, Offline). |
| **Unified Facade** | [`engine.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/engine.py) | Single `AdvancedDynamicRAG` orchestrator connecting indexing, retrieval, synthesis, and negative probe testing. |
| **Interactive Dashboard** | [`app.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/app.py) | Full-featured Streamlit UI with live tuning sliders, 2D PCA vector scatterplot, and 4-tier observability panel. |
| **Verification Suite** | [`test_engine.py`](file:///c:/Users/RAJASHEKAR%20JATOTH/RAG/test_engine.py) | 11 unit & integration tests validating all loaders, re-indexing speeds, retrieval, and prompt guardrails. |

---

## 🚀 Quickstart

### 1. Run Unit Tests
```bash
python test_engine.py
```

### 2. Launch Streamlit UI
```bash
streamlit run app.py
```

### 3. Programmatic Usage in Python

```python
from engine import AdvancedDynamicRAG

# Initialize with pre-configured domain preset
rag = AdvancedDynamicRAG.from_preset("study_buddy")

# Ask questions with top-K and relevance threshold
result = rag.ask(
    query="How does the operating system mitigate starvation in priority scheduling?",
    top_k=3,
    threshold=0.10
)

print("Answer:", result["answer"])
print("Latency:", result["latency_ms"], "ms")
print("Token Metrics:", result["token_metrics"])
print("Retrieval Metrics:", result["retrieval_metrics"])

# Dynamically re-chunk in memory on the fly
rag.reindex(max_words=60, overlap=12)

# Ingest external files (PDF, DOCX, TXT)
rag.load_file("my_lecture_notes.pdf")
```
