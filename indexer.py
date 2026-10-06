"""
Indexer module for OmniRAG Studio.
Handles dynamic sliding-window chunking, vector embedding generation,
indexing telemetry, and 2D dimensionality reduction (PCA) for observability.
"""

import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.decomposition import PCA

from embeddings import BaseEmbeddingProvider, get_embedding_provider
from loaders import Document


class DynamicChunker:
    """Configurable sliding-window text chunker with boundary tracking."""
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 80,
        overlap: int = 15,
        doc_id: int = 0,
        source_title: str = "Document"
    ) -> List[Dict[str, Any]]:
        """Splits a document text into overlapping word-level chunks with metadata."""
        words = text.split()
        if not words:
            return []
            
        chunks = []
        # Ensure overlap is strictly less than chunk_size to avoid infinite loops
        effective_overlap = min(overlap, max(0, chunk_size - 1))
        step_size = max(1, chunk_size - effective_overlap)
        
        if len(words) <= chunk_size:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": 0,
                "source_title": source_title,
                "text": text.strip(),
                "word_count": len(words),
                "char_length": len(text.strip()),
                "start_word_idx": 0,
                "end_word_idx": len(words),
                "is_overlap_boundary": False
            })
            return chunks

        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_slice = words[start:end]
            chunk_str = " ".join(chunk_slice)
            
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_idx,
                "source_title": source_title,
                "text": chunk_str,
                "word_count": len(chunk_slice),
                "char_length": len(chunk_str),
                "start_word_idx": start,
                "end_word_idx": end,
                "is_overlap_boundary": start > 0 and effective_overlap > 0
            })
            
            chunk_idx += 1
            start += step_size
            if end >= len(words):
                break
                
        return chunks


class VectorIndexer:
    """Manages chunk storage, vector embeddings, re-indexing profiling, and PCA projections."""
    
    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or get_embedding_provider("tfidf")
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None
        self.pca_coords_2d: Optional[np.ndarray] = None
        self.last_metrics: Dict[str, Any] = {}

    def set_embedding_provider(self, provider: BaseEmbeddingProvider) -> None:
        """Dynamically switches the active embedding provider."""
        self.embedding_provider = provider
        
    def index_documents(
        self,
        documents: List[Union[Document, str, Dict[str, str]]],
        chunk_size: int = 80,
        overlap: int = 15
    ) -> Dict[str, Any]:
        """
        Dynamically re-chunks and re-embeds source documents.
        Meets REQ-F1: Instant re-indexing in under 500ms.
        """
        start_time = time.time()
        self.chunks = []
        
        # 1. Normalize and chunk all input documents
        for doc_id, doc in enumerate(documents):
            if isinstance(doc, Document):
                text = doc.content
                title = doc.metadata.get("title", f"Document {doc_id + 1}")
            elif isinstance(doc, dict):
                text = doc.get("content", "")
                title = doc.get("title", f"Document {doc_id + 1}")
            else:
                text = doc
                title = f"Document {doc_id + 1}"
                
            doc_chunks = DynamicChunker.chunk_text(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id=doc_id,
                source_title=title
            )
            # Re-assign global sequential chunk_id
            for c in doc_chunks:
                c["global_chunk_id"] = len(self.chunks)
                self.chunks.append(c)

        if not self.chunks:
            self.embeddings = np.zeros((0, self.embedding_provider.dimension or 0))
            self.pca_coords_2d = np.zeros((0, 2))
            return {
                "total_chunks": 0,
                "vocab_size": 0,
                "matrix_shape": (0, 0),
                "indexing_time_ms": 0.0
            }

        # 2. Extract chunk texts and generate embeddings
        texts = [c["text"] for c in self.chunks]
        self.embedding_provider.fit(texts)
        self.embeddings = self.embedding_provider.embed_documents(texts)
        
        # 3. Fit 2D PCA for visual projection if we have at least 2 samples and >1 dimensions
        n_samples, n_features = self.embeddings.shape
        if n_samples >= 2 and n_features >= 2:
            try:
                self.pca = PCA(n_components=2)
                self.pca_coords_2d = self.pca.fit_transform(self.embeddings)
            except Exception:
                self.pca = None
                self.pca_coords_2d = np.zeros((n_samples, 2))
        else:
            self.pca = None
            self.pca_coords_2d = np.zeros((n_samples, 2))

        indexing_latency_ms = (time.time() - start_time) * 1000
        
        self.last_metrics = {
            "total_chunks": len(self.chunks),
            "vocab_size": self.embeddings.shape[1] if self.embeddings.ndim > 1 else 0,
            "matrix_shape": list(self.embeddings.shape),
            "indexing_time_ms": round(indexing_latency_ms, 2),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "step_size": max(1, chunk_size - min(overlap, max(0, chunk_size - 1))),
            "provider": self.embedding_provider.provider_name
        }
        return self.last_metrics

    def project_query_2d(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        """Projects a query vector onto the fitted 2D PCA space for scatterplot rendering."""
        if (
            self.pca is not None
            and query_vector is not None
            and self.embeddings is not None
            and query_vector.shape[1] == self.embeddings.shape[1]
        ):
            try:
                return self.pca.transform(query_vector)[0]
            except Exception:
                return None
        return None
