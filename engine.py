"""
OmniRAG Core Engine module.
Orchestrates Ingestion, Indexing, Retrieval, and Generation into a cohesive,
production-ready AdvancedDynamicRAG facade.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import time

from config import PRESETS, PersonaPreset, ModelConfig
from loaders import Document, DocumentLoader
from embeddings import get_embedding_provider, BaseEmbeddingProvider
from indexer import VectorIndexer
from retriever import VectorRetriever
from generator import GroundedGenerator


class AdvancedDynamicRAG:
    """
    Enterprise-ready Advanced Dynamic RAG engine with runtime parameter tuning
    and real-time 4-tier observability metrics.
    """
    
    def __init__(
        self,
        raw_documents: Optional[List[Union[str, Document, Dict[str, str]]]] = None,
        persona: Optional[str] = None,
        embedding_provider: str = "tfidf",
        llm_provider: str = "anthropic",
        api_key: Optional[str] = None
    ):
        self.raw_documents: List[Union[str, Document, Dict[str, str]]] = raw_documents or []
        self.persona: str = persona or "You are an intelligent AI assistant."
        
        # Initialize sub-modules
        self.embedding_provider = get_embedding_provider(embedding_provider, api_key=api_key)
        self.indexer = VectorIndexer(embedding_provider=self.embedding_provider)
        self.retriever = VectorRetriever(indexer=self.indexer)
        self.generator = GroundedGenerator(provider=llm_provider, api_key=api_key)
        
        self.active_preset_name: Optional[str] = None
        self.active_chunk_size: int = 80
        self.active_overlap: int = 15

        # If initial documents provided, run index
        if self.raw_documents:
            self.reindex(max_words=self.active_chunk_size, overlap=self.active_overlap)

    @classmethod
    def from_preset(
        cls,
        preset_key: str = "study_buddy",
        embedding_provider: str = "tfidf",
        llm_provider: str = "anthropic",
        api_key: Optional[str] = None
    ) -> "AdvancedDynamicRAG":
        """Factory method to instantiate the engine configured with a PRD domain preset."""
        preset = PRESETS.get(preset_key, PRESETS["study_buddy"])
        instance = cls(
            raw_documents=preset.sample_documents,
            persona=preset.persona_prompt,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            api_key=api_key
        )
        instance.active_preset_name = preset_key
        instance.active_chunk_size = preset.default_chunk_size
        instance.active_overlap = preset.default_overlap
        instance.reindex(max_words=preset.default_chunk_size, overlap=preset.default_overlap)
        return instance

    def load_preset(self, preset_key: str) -> Dict[str, Any]:
        """Switches domain preset (Technical Study Buddy vs E-Commerce Support)."""
        preset = PRESETS.get(preset_key, PRESETS["study_buddy"])
        self.raw_documents = preset.sample_documents
        self.persona = preset.persona_prompt
        self.active_preset_name = preset_key
        self.active_chunk_size = preset.default_chunk_size
        self.active_overlap = preset.default_overlap
        return self.reindex(max_words=preset.default_chunk_size, overlap=preset.default_overlap)

    def add_document(self, document: Union[Document, str, Dict[str, str]]) -> None:
        """Appends a document to the current corpus."""
        self.raw_documents.append(document)

    def load_file(self, file_source: Any, filename: Optional[str] = None) -> Document:
        """Loads and appends an external file (PDF, DOCX, TXT)."""
        doc = DocumentLoader.auto_load(file_source, filename=filename)
        self.add_document(doc)
        return doc

    def set_embedding_provider(
        self,
        provider_type: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Switches embedding backend and triggers re-indexing."""
        self.embedding_provider = get_embedding_provider(
            provider_type=provider_type,
            model_name=model_name,
            api_key=api_key
        )
        self.indexer.set_embedding_provider(self.embedding_provider)
        return self.reindex(max_words=self.active_chunk_size, overlap=self.active_overlap)

    def set_llm_provider(
        self,
        provider_type: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        """Switches generation model provider (Anthropic, OpenAI, or Offline)."""
        self.generator = GroundedGenerator(
            provider=provider_type,
            model=model_name,
            api_key=api_key
        )

    def reindex(self, max_words: int = 80, overlap: int = 15) -> Dict[str, Any]:
        """
        Dynamically re-chunks and re-embeds source documents.
        Maintains signature compatibility with PRD Section 8.
        """
        self.active_chunk_size = max_words
        self.active_overlap = overlap
        metrics = self.indexer.index_documents(
            documents=self.raw_documents,
            chunk_size=max_words,
            overlap=overlap
        )
        return metrics

    def retrieve_with_metrics(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieves top-K chunks and calculates vector retrieval metrics.
        Maintains signature compatibility with PRD Section 8.
        """
        return self.retriever.retrieve(query=query, top_k=top_k, threshold=threshold)

    def ask(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes full RAG loop: vector retrieval + prompt augmentation + LLM synthesis + telemetry.
        Maintains signature compatibility with PRD Section 8 while returning enhanced observability.
        """
        start_time = time.time()
        
        # 1. Retrieve evidence
        retrieved, ret_metrics = self.retrieve_with_metrics(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
        
        # 2. Synthesize grounded answer
        gen_result = self.generator.generate(
            persona=self.persona,
            retrieved_sources=retrieved,
            query=query
        )
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "answer": gen_result["answer"],
            "prompt": gen_result["prompt"],
            "retrieved_sources": retrieved,
            "retrieval_metrics": ret_metrics,
            "token_metrics": gen_result["token_metrics"],
            "quality_metrics": gen_result["quality_metrics"],
            "latency_ms": round(total_latency_ms, 2)
        }

    def run_negative_test(self, top_k: int = 3, threshold: float = 0.20) -> Dict[str, Any]:
        """
        REQ-F6: Out-of-Domain Probe Test.
        Sends an intentionally irrelevant question to test whether the system avoids hallucinations
        and emits the negative refusal constraint.
        """
        out_of_domain_query = "What is the average flight speed of an African swallow carrying a coconut?"
        return self.ask(query=out_of_domain_query, top_k=top_k, threshold=threshold)
