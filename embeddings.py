"""
Embeddings module for OmniRAG Studio.
Provides pluggable embedding providers:
1. TF-IDF Sparse (scikit-learn) - Zero-latency lexical scoring & vocabulary profiling.
2. Dense Neural / Local (SentenceTransformers all-MiniLM-L6-v2) - 384-dimensional dense semantic vectors.
3. Dense Neural / Cloud (OpenAI text-embedding-3-small) - 1536-dimensional dense vectors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fits vocabulary/weights if required by the model (e.g. TF-IDF)."""
        pass
        
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a batch of text chunks. Returns (N, D) array."""
        pass
        
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generates embedding for a single query. Returns (1, D) array."""
        pass
        
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns human-readable name of the provider."""
        pass

    @property
    def is_sparse(self) -> bool:
        """Whether the representation is sparse (e.g., TF-IDF)."""
        return False


class TFIDFEmbeddingProvider(BaseEmbeddingProvider):
    """Sparse lexical TF-IDF vectorizer using scikit-learn."""
    
    def __init__(self, stop_words: str = "english"):
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self._fitted = False
        self._dim = 0
        
    @property
    def provider_name(self) -> str:
        return f"TF-IDF Sparse (Vocab: {self._dim} terms)"
        
    @property
    def dimension(self) -> int:
        return self._dim
        
    @property
    def is_sparse(self) -> bool:
        return True

    def fit(self, texts: List[str]) -> None:
        if not texts:
            self._dim = 0
            self._fitted = False
            return
        self.vectorizer.fit(texts)
        self._dim = len(self.vectorizer.vocabulary_)
        self._fitted = True
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        if not texts or self._dim == 0:
            return np.zeros((len(texts), 0))
        # Transform returns a scipy sparse matrix; convert to dense array for uniform similarity math
        sparse_mat = self.vectorizer.transform(texts)
        return sparse_mat.toarray()
        
    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted or self._dim == 0:
            return np.zeros((1, 0))
        sparse_vec = self.vectorizer.transform([query])
        return sparse_vec.toarray()

    def get_query_sparsity(self, query: str) -> Dict[str, Any]:
        """Observability helper: returns non-zero components in the query vector."""
        if not self._fitted or self._dim == 0:
            return {"non_zero_terms": 0, "matched_vocab": []}
        sparse_vec = self.vectorizer.transform([query])
        indices = sparse_vec.indices
        feature_names = self.vectorizer.get_feature_names_out()
        matched = [feature_names[i] for i in indices]
        return {
            "non_zero_terms": int(sparse_vec.nnz),
            "matched_vocab": matched
        }


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Dense neural embeddings using local HuggingFace / SentenceTransformers models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim = 384  # default for all-MiniLM-L6-v2
        
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_embedding_dimension()
            self._dim = dim if dim is not None else 384
            
    @property
    def provider_name(self) -> str:
        return f"SentenceTransformers ({self.model_name}, {self._dim}d)"
        
    @property
    def dimension(self) -> int:
        return self._dim

    def fit(self, texts: List[str]) -> None:
        # Pretrained neural embeddings do not require fitting on corpus
        self._load_model()
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self._dim))
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)
        
    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        assert self._model is not None
        emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Dense neural embeddings using OpenAI text-embedding-3 models."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None
        self._dim = 1536 if "small" in model_name or "ada" in model_name else 3072
        
    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client
        
    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model_name}, {self._dim}d)"
        
    @property
    def dimension(self) -> int:
        return self._dim

    def fit(self, texts: List[str]) -> None:
        pass
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim))
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model_name)
        vectors = [item.embedding for item in response.data]
        arr = np.array(vectors, dtype=np.float32)
        # Normalize for unit cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
        
    def embed_query(self, query: str) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(input=[query], model=self.model_name)
        vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def get_embedding_provider(
    provider_type: str = "tfidf",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> BaseEmbeddingProvider:
    """Factory creating the appropriate embedding provider."""
    ptype = provider_type.lower().replace("_", "-")
    
    if ptype in ("tfidf", "sparse"):
        return TFIDFEmbeddingProvider()
    elif ptype in ("sentence-transformers", "dense", "local", "minilm"):
        m_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbeddingProvider(model_name=m_name)
    elif ptype in ("openai", "text-embedding-3"):
        m_name = model_name or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(model_name=m_name, api_key=api_key)
    else:
        # Default fallback
        return TFIDFEmbeddingProvider()
