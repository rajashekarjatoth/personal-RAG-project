"""
Retriever module for OmniRAG Studio.
Handles vector similarity calculation, top-K ranking, score thresholding,
and vector observability analytics (sparsity, distribution metrics, 2D coordinates).
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from indexer import VectorIndexer
from embeddings import TFIDFEmbeddingProvider


class VectorRetriever:
    """Performs metric-rich retrieval over VectorIndexer chunks."""
    
    def __init__(self, indexer: VectorIndexer):
        self.indexer = indexer

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieves top-K chunks above threshold and calculates vector observability metrics.
        Complies with PRD Section 4.2 and Section 5.2.
        """
        start_time = time.time()
        
        if not self.indexer.chunks or self.indexer.embeddings is None or len(self.indexer.embeddings) == 0:
            return [], {
                "query_non_zero_components": 0,
                "max_similarity_score": 0.0,
                "mean_similarity_score": 0.0,
                "min_similarity_score": 0.0,
                "chunks_above_threshold": 0,
                "total_chunks": 0,
                "vocabulary_dimension": 0,
                "retrieval_latency_ms": 0.0,
                "all_scores": [],
                "query_2d_coord": None
            }

        # 1. Embed query
        query_vec = self.indexer.embedding_provider.embed_query(query)
        
        # 2. Compute Cosine Similarity
        # Handles both dense numpy arrays and sparse TF-IDF outputs uniformly
        sim_matrix = cosine_similarity(query_vec, self.indexer.embeddings)
        similarities = sim_matrix[0] if len(sim_matrix) > 0 else np.array([])
        
        # 3. Sort indices in descending order of similarity
        ranked_indices = similarities.argsort()[::-1]
        
        # 4. Project query to 2D PCA space for UI visualization
        query_2d = self.indexer.project_query_2d(query_vec)
        query_2d_coord = [round(float(query_2d[0]), 4), round(float(query_2d[1]), 4)] if query_2d is not None else None
        
        retrieved = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                chunk = self.indexer.chunks[idx].copy()
                chunk["score"] = round(score, 4)
                
                # Attach 2D coordinates if available
                if self.indexer.pca_coords_2d is not None and len(self.indexer.pca_coords_2d) > idx:
                    chunk["pca_coord"] = [
                        round(float(self.indexer.pca_coords_2d[idx][0]), 4),
                        round(float(self.indexer.pca_coords_2d[idx][1]), 4)
                    ]
                retrieved.append(chunk)

        # 5. Extract query sparsity
        if isinstance(self.indexer.embedding_provider, TFIDFEmbeddingProvider):
            sparsity_info = self.indexer.embedding_provider.get_query_sparsity(query)
            query_non_zero = sparsity_info["non_zero_terms"]
            matched_vocab = sparsity_info["matched_vocab"]
        else:
            query_non_zero = int(np.count_nonzero(query_vec))
            matched_vocab = []

        retrieval_latency_ms = (time.time() - start_time) * 1000
        
        all_scores = [round(float(s), 4) for s in similarities]
        
        metrics = {
            "query_non_zero_components": query_non_zero,
            "matched_terms": matched_vocab,
            "max_similarity_score": round(float(np.max(similarities)), 4) if len(similarities) > 0 else 0.0,
            "mean_similarity_score": round(float(np.mean(similarities)), 4) if len(similarities) > 0 else 0.0,
            "min_similarity_score": round(float(np.min(similarities)), 4) if len(similarities) > 0 else 0.0,
            "chunks_above_threshold": len(retrieved),
            "total_chunks": len(self.indexer.chunks),
            "vocabulary_dimension": int(self.indexer.embeddings.shape[1]),
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "all_scores": all_scores,
            "query_2d_coord": query_2d_coord
        }
        
        return retrieved, metrics
