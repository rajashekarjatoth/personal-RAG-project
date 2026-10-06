# Product Requirement Document (PRD)

---

## 1. Document Overview & Executive Summary

* **Product Name**: **OmniRAG Studio** (Dynamic Multi-Domain Advanced RAG Engine & Observability Platform)
* **Status**: Draft / Proposed
* **Target Audience**: AI Engineers, ML Platform Developers, Enterprise Domain Specialists, and Technical Interviewers
* **Document Objective**: Define end-to-end functional, architectural, and analytical specifications for an enterprise-ready, interactive Advanced Retrieval-Augmented Generation (RAG) platform. The platform supports dynamic parameter reconfiguration at runtime (chunk size, overlap, top-K, similarity thresholds), dual-domain deployment, and real-time observability across token economics, dense vector spaces, and retrieval quality metrics.

---

## 2. Problem Statement & Strategic Value

### 2.1 The Problem

* Standard LLMs suffer from three core limitations: knowledge cutoffs, factual hallucinations, and lack of access to private enterprise data.


* Traditional RAG tutorials often present a static, "black-box" pipeline where chunking size, overlap ratios, and retrieval parameters are hardcoded. In production, chunking without overlap causes boundary cuts where facts are severed across chunks, while over-retrieval ($K > 8$) leads to context dilution, higher costs, and the **"Lost in the Middle"** phenomenon.


* Engineers and researchers lack an interactive observability environment to inspect vector similarity distributions, token costs, latency bottlenecks, and chunk boundary impacts in real time.

### 2.2 The Solution

* An **Advanced RAG Engine** that decouples the indexing and query layers, enabling dynamic, runtime adjustment of chunk parameters without server restarts.


* Complete visibility into internal RAG mechanics: token attribution, embedding vector geometries, cosine similarity distributions, and retrieval precision/recall benchmarks.


* Multi-persona and multi-domain handling within a single unified platform.



---

## 3. Selected Use Cases & Persona Architectures

As established in the core engine design, the underlying mathematical and retrieval pipeline remains consistent while the domain datasets, chunking profiles, and persona system prompts adapt dynamically. The platform implements the following two distinct use cases:

### 3.1 Use Case A: Technical Exam Study Buddy (Dense Academic & Conceptual Text)

* **Source Material**: Academic textbooks, lecture notes, algorithms, systems design principles (e.g., OS CPU scheduling, memory management, database locking mechanisms).


* **Data Nature**: High semantic density, interdependent logic, complex multi-hop terminology.


* **System Persona**:
> *"You are a patient, expert study partner helping a student revise for technical exams. Explain concepts rigorously yet simply, citing retrieved lecture notes using format `[Source X]`. Strictly ground your explanation in provided notes; if absent, state: 'I don't have that information in your revision notes.'"*
> 


* **Default Tuning Baseline**:
* Chunk Size: $50$ words (~$65$ tokens)


* Chunk Overlap: $12\text{--}15$ words ($20\text{--}25\%$)


* Top-$K$: $2\text{--}4$ chunks


* Reason: Dense academic text packs multiple concepts into single sentences; smaller chunks isolate atomic concepts without dilution.





### 3.2 Use Case B: Enterprise E-Commerce & Policy Support Agent (Structured Product & Policy Specs)

* **Source Material**: Product catalogs, return windows, warranty terms, shipping tiers, SKU dimension charts.


* **Data Nature**: Semi-structured numerical constraints, explicit conditional thresholds (days, dimensions, refund rules).


* **System Persona**:
> *"You are a polite, highly accurate customer support agent. Answer customer queries with strict adherence to policy rules and product specifications using format `[Source X]`. Never extrapolate or assume policies outside the documented terms. If uncertain or unlisted, say 'I don't have information on that policy.'"*
> 


* **Default Tuning Baseline**:
* Chunk Size: $80\text{--}100$ words (~$110\text{--}130$ tokens)


* Chunk Overlap: $15\text{--}20$ words ($15\text{--}20\%$)


* Top-$K$: $3\text{--}5$ chunks


* Reason: Policy clauses contain condition-action rules (e.g., "Returns accepted within 15 days IF tags intact") that must remain unified in one chunk to prevent hallucinated compliance.





---

## 4. System Architecture & Technical Pipeline

The architecture is divided into two primary loops: the **Dynamic Indexing/Re-Chunking Pipeline** (Phase A) and the **Real-Time Retrieval & Generation Pipeline** (Phase B).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FRONTEND: DYNAMIC CONTROL DASHBOARD                    │
│  [Use Case Switch]  [Chunk Size: 20-300w]  [Overlap: 0-50w]  [Top-K: 1-10]  │
└───────┬───────────────────────────────┬──────────────────────────────┬──────┘
        │ (1) Config Changes            │ (4) User Query               │ (7) Metrics Stream
        ▼                               ▼                              ▼
┌───────────────────────────┐   ┌──────────────────────────┐   ┌───────────────────────────┐
│     INDEXING PIPELINE     │   │    RETRIEVAL PIPELINE    │   │   OBSERVABILITY ENGINE    │
│                           │   │                          │   │                           │
│ 1. Dynamic Chunker        │   │ 1. Query Embedder        │   │ • Token Counter (Prompt/  │
│    (Sliding Window/Tokens)│   │ 2. Cosine Vector Search  │   │   Completion/Context)     │
│ 2. Dual Embedder Engine   │   │ 3. Score Threshold Filter│   │ • Vector Dimensionality & │
│    (TF-IDF / Neural 1536d)│   │ 4. Optional Re-ranker    │   │   2D PCA Projection       │
│ 3. In-Memory Vector Store │   │ 5. Grounded Prompt Inject│   │ • Cosine Sim Distributions│
└─────────────┬─────────────┘   └─────────────┬────────────┘   │ • Recall@K & Faithfulness │
              │                               │                └─────────────▲─────────────┘
              ▼                               ▼                              │
     [Vector Store / RAM] ─────────────► [LLM Inference] ────────────────────┘

```

### 4.1 Indexing Pipeline (Real-Time Reactive)

1. **Dynamic Chunking Engine**:
* Evaluates input raw text arrays or uploaded documents (`PDF`, `DOCX`, `TXT`, `Markdown`).


* Parameterized sliding-window algorithm:

$$\text{Step Size} = \text{Chunk Size} - \text{Overlap}$$



* Emits chunks tagged with metadata: `chunk_id`, `doc_id`, `start_char`, `end_char`, `token_count`.


2. **Pluggable Vectorizer & Embedder**:
* **Lightweight / Local Mode**: Scikit-Learn `TfidfVectorizer` (for zero-latency development and exact lexical scoring).


* **Neural / Production Mode**: Dense embeddings (e.g., Voyage AI, OpenAI `text-embedding-3-small` at 1536-d, or Hugging Face `all-MiniLM-L6-v2` at 384-d).




3. **In-Memory / Vector Storage**:
* In-memory cosine similarity matrix for sub-1,000 chunks.


* Optional FAISS / Chroma layer for large-scale passage scaling ($>10,000$ chunks).





### 4.2 Query & Generation Pipeline

1. **Query Processing**: Vectorizes the incoming question using the active embedder pipeline.


2. **Cosine Similarity Computation**:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\Vert{}\vec{q}\Vert{}_2 \Vert{}\vec{d}\Vert{}_2}$$



3. **Top-$K$ Retrieval & Dynamic Cutoff**:
* Extracts top-$K$ chunks based on similarity scores.


* Applies similarity cutoff threshold $\tau$ (drops chunks where score $< \tau$).




4. **Augmented Prompt Engineering**:
* Context is constructed by prefixing each chunk with clear provenance markers (`[Source 1]`, `[Source 2]`).


* Hard constraint enforcement: Instructs model to output explicit uncertainty statements when context is insufficient, preventing hallucinations.




5. **Generation**: Invokes target LLM (e.g., Claude 3.5 Sonnet / GPT-4o / open-source alternatives).



---

## 5. Dynamic Dashboard & Metric Observability Suite

The dashboard provides real-time control sliders on the left panel and a live four-tier metrics visualization panel on the right.

### 5.1 Real-Time Interactive Controls (Live Tuning Panel)

* **Use Case Selector**: Toggle between **Technical Study Buddy** and **E-Commerce Support Bot**. Switching updates documents, default hyperparameters, and persona prompts.


* **Chunk Size Slider**: Dynamic slider from $20$ to $500$ words (or $30$ to $700$ tokens). Re-chunks documents upon release.


* **Chunk Overlap Slider**: Dynamic slider from $0$ to $100$ words ($0\text{--}40\%$). Displays warning when overlap exceeds $50\%$.


* **Top-$K$ Retrieval Slider**: Dynamic slider from $K=1$ to $K=10$.


* **Relevance Threshold ($\tau$)**: Slider from $0.00$ to $1.00$ (filters out low-scoring irrelevant noise).


* **Embedding Model Selector**: Switch between `TF-IDF Sparse (Vocabulary Size)` and `Dense Neural (e.g., 384-d / 1536-d)`.



### 5.2 Real-Time Observability Metrics Display

| Metric Category | Displayed Parameters & Visualizations | Diagnostic Purpose |
| --- | --- | --- |
| **1. Token Economics** | • Total Prompt Tokens<br>

<br>• Retrieved Context Token Count<br>

<br>• Completion Tokens<br>

<br>• Estimated API Query Cost ($USD) | Tracks prompt overhead, identifies token waste, and monitors context window saturation.

 |
| **2. Vector & Embedding Dynamics** | • Vector Dimension size (e.g., $V=142$ for TF-IDF, $D=1536$ for Neural)

<br>

<br>• 2D PCA / UMAP Projection Scatterplot (visualizing query position vs. top-$K$ vs. background chunks)<br>

<br>• Sparsity / Density Ratio | Validates whether the query vector aligns with semantic clusters or falls into semantic voids.

 |
| **3. Retrieval Analytics** | • Ranked list of retrieved chunks with raw Cosine Scores

<br>

<br>• Similarity Distribution Bar Chart ($1.0$ down to $0.0$ threshold)<br>

<br>• Chunk ID & Document Origin mapping

 | Diagnoses **Retrieval Failure** vs. **Generation Failure**; identifies optimal threshold cutoffs.

 |
| **4. Quality & Faithfulness** | • Context-to-Answer Attribution Score (%)<br>

<br>• Source Citation Fidelity (`[Source X]` verification)

<br>

<br>• Fallback Detection ("I don't have that information" flag)

 | Verifies answer grounding and monitors hallucination reduction.

 |

---

## 6. Functional Specifications & Requirements

### 6.1 Parameter Adjustment Lifecycle

* **REQ-F1 (Instant Re-indexing)**: Changing the chunk size or overlap slider must re-index the active domain corpus in memory in under $500\text{ ms}$ for documents under $100{,}000$ words.
* **REQ-F2 (Dynamic Chunk Highlighting)**: Chunks must be rendered visually in a document inspector tab, highlighting chunk boundaries and overlapping zones in alternating colors to allow immediate inspection of boundary cuts.


* **REQ-F3 (Dual Persona Orchestration)**: Switching between the Study Buddy and Support Agent must automatically switch persona system prompts without overwriting custom prompt modifications.



### 6.2 Grounded Generation & Citation Engine

* **REQ-F4 (Source Indexing & Traceability)**: Every retrieved passage injected into the context window must be wrapped in `[Source N]` delimiters.


* **REQ-F5 (Strict Context Constraint)**: The system prompt must enforce zero extrapolation on ungrounded facts, requiring explicit admission of missing context.


* **REQ-F6 (Out-of-Domain Probe Test)**: A dedicated "Negative Test" button sends an out-of-scope query to verify whether the engine correctly states that the answer is not present.



---

## 7. Non-Functional & Performance Requirements

* **NFR-1 (End-to-End Latency)**: Retrieval latency must stay under $50\text{ ms}$ for in-memory sparse vectors and under $150\text{ ms}$ for neural embedding API calls. LLM time-to-first-token (TTFT) should be under $1.5\text{ s}$.
* **NFR-2 (Memory Overhead)**: The in-memory vector matrix for default datasets must consume less than $50\text{ MB}$ RAM, ensuring compatibility with lightweight execution environments.


* **NFR-3 (Extensibility)**: Modular architecture allowing easy replacement of the embedding provider or vector store.



---

## 8. Implementation Code Architecture

The following executable Python implementation powers the core engine behind this PRD, including real-time parameter re-indexing, metrics tracking, and dual-use-case configuration:

```python
"""
Advanced RAG Engine with Dynamic Tuning and Real-Time Observability Metrics
Compatible with Streamlit, Gradio, or FastAPI backends.
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

class AdvancedDynamicRAG:
    def __init__(self, raw_documents: List[str], persona: str):
        self.raw_documents = raw_documents
        self.persona = persona
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.chunk_vectors = None
        self.client = anthropic.Anthropic()
        
    def reindex(self, max_words: int = 80, overlap: int = 15) -> Dict[str, Any]:
        """Dynamically re-chunks and re-embeds source documents."""
        start_time = time.time()
        self.chunks = []
        
        for doc_id, doc in enumerate(self.raw_documents):
            words = doc.split()
            if len(words) <= max_words:
                self.chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": len(self.chunks),
                    "text": doc,
                    "word_count": len(words)
                })
            else:
                start = 0
                while start < len(words):
                    end = start + max_words
                    chunk_text = " ".join(words[start:end])
                    self.chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": len(self.chunks),
                        "text": chunk_text,
                        "word_count": len(words[start:end])
                    })
                    start += (max_words - overlap)
                    if start >= len(words):
                        break
                        
        # Re-compute Vector Embeddings
        self.vectorizer = TfidfVectorizer(stop_words="english")
        texts = [c["text"] for c in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(texts)
        indexing_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "total_chunks": len(self.chunks),
            "vocab_size": self.chunk_vectors.shape[1],
            "matrix_shape": self.chunk_vectors.shape,
            "indexing_time_ms": round(indexing_latency_ms, 2)
        }

    def retrieve_with_metrics(self, query: str, top_k: int = 3, threshold: float = 0.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieves top-K chunks and calculates vector retrieval metrics."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors)[0]
        
        # Sort indices by descending similarity
        ranked_indices = similarities.argsort()[::-1]
        
        retrieved = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                retrieved.append({
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "doc_id": self.chunks[idx]["doc_id"],
                    "text": self.chunks[idx]["text"],
                    "score": round(score, 4)
                })
                
        metrics = {
            "query_non_zero_components": int(query_vec.nnz),
            "max_similarity_score": round(float(np.max(similarities)), 4) if len(similarities) > 0 else 0.0,
            "mean_similarity_score": round(float(np.mean(similarities)), 4) if len(similarities) > 0 else 0.0,
            "chunks_above_threshold": len(retrieved),
            "vocabulary_dimension": self.chunk_vectors.shape[1]
        }
        return retrieved, metrics

    def ask(self, query: str, top_k: int = 3, threshold: float = 0.0) -> Dict[str, Any]:
        """Executes full RAG loop with prompt augmentation and token usage tracking."""
        start_time = time.time()
        retrieved, ret_metrics = self.retrieve_with_metrics(query, top_k=top_k, threshold=threshold)
        
        # Build Augmented Prompt
        context = "\n\n".join(
            f"[Source {i+1}]: {r['text']}" for i, r in enumerate(retrieved)
        )
        prompt = (
            f"You are {self.persona}. Answer using ONLY the context provided below.\n"
            f"If the answer is not contained in the context, explicitly say: 'I don't have that information.'\n"
            f"Always cite sources where relevant, e.g., [Source 1].\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        # LLM Invocation
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "answer": response.content[0].text,
            "retrieved_sources": retrieved,
            "retrieval_metrics": ret_metrics,
            "token_metrics": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            "latency_ms": round(total_latency_ms, 2)
        }

```

---

## 9. Failure Mode Analysis & Edge Case Handling

Following practical RAG failure mode guidelines:

| Failure Mode | Root Cause | Observable Metric | Mitigation Strategy |
| --- | --- | --- | --- |
| **Chunk Boundary Severing**<br> | Overlap set to $0$ words; sentence split across boundary.

 | Low similarity score on adjacent chunks ($<0.15$).

 | Increase overlap slider to $15\text{--}25\%$.

 |
| **Lost in the Middle**<br> | Over-retrieval ($K > 6$); model neglects middle context.

 | Total context tokens $> 2,000$, low citation of middle chunks.

 | Constrain $K$ to $2\text{--}4$ and sort chunks by score descending.

 |
| **Hallucination on Unknowns**<br> | Weak system prompt instructions or low relevance threshold.

 | Answer contains ungrounded facts without `[Source]` tags.

 | Tighten prompt ("say you don't have that info") and raise threshold $\tau \ge 0.20$.

 |
| **Sparse Vocabulary Mismatch**<br> | TF-IDF misses synonyms (e.g., "refund" vs "money back").

 | Similarity score $= 0.0$ despite relevant text in corpus.

 | Switch embedder toggle to Neural Dense Vectors.

 |

---

## 10. Success Criteria & KPIs

1. **Groundedness / Faithfulness**: $\ge 95\%$ of generated statements must directly map to an injected `[Source X]` reference.


2. **Negative Accuracy (Refusal Rate)**: $\ge 98\%$ correct refusal ("I don't have that information") on out-of-domain evaluation questions.


3. **Indexing Reconfiguration Speed**: Dynamic re-chunking and vector indexing completed in $<500\text{ ms}$ upon dashboard slider adjustments.
4. **Interview Demonstrability**: Allows an engineer to articulate the tradeoffs between chunk sizes, overlap percentages, vector sparsity, and top-$K$ selection on a live dashboard.