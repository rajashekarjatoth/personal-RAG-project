"""
Unit & Integration Test Suite for OmniRAG Studio.
Tests document loaders (PDF, DOCX, TXT), embedding providers, dynamic re-indexing,
vector retrieval, grounded generation, and failure mode edge cases.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PRESETS
from loaders import DocumentLoader, Document
from embeddings import TFIDFEmbeddingProvider, get_embedding_provider
from indexer import DynamicChunker, VectorIndexer
from retriever import VectorRetriever
from generator import PromptBuilder, GroundedGenerator
from engine import AdvancedDynamicRAG


class TestDocumentLoaders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_txt_loader(self):
        file_path = os.path.join(self.temp_dir.name, "test_notes.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Operating systems rely on CPU scheduling algorithms such as Round Robin and Priority Scheduling.")
            
        doc = DocumentLoader.auto_load(file_path)
        self.assertIn("Round Robin", doc.content)
        self.assertEqual(doc.metadata["file_type"], "txt")
        self.assertGreater(doc.word_count, 5)

    def test_docx_loader(self):
        import docx
        file_path = os.path.join(self.temp_dir.name, "test_policy.docx")
        doc_obj = docx.Document()
        doc_obj.add_paragraph("Return policy states that items must be returned within 30 calendar days.")
        doc_obj.add_paragraph("Electronics carry an accelerated 14-day return window.")
        doc_obj.save(file_path)

        doc = DocumentLoader.auto_load(file_path)
        self.assertIn("30 calendar days", doc.content)
        self.assertIn("14-day", doc.content)
        self.assertEqual(doc.metadata["file_type"], "docx")
        self.assertEqual(doc.metadata["total_paragraphs"], 2)

    def test_pdf_loader(self):
        import pypdf
        file_path = os.path.join(self.temp_dir.name, "test_doc.pdf")
        
        # Generate a minimal synthetic PDF using pypdf writer
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(file_path, "wb") as f:
            writer.write(f)
            
        doc = DocumentLoader.auto_load(file_path)
        self.assertEqual(doc.metadata["file_type"], "pdf")
        self.assertEqual(doc.metadata["total_pages"], 1)


class TestDynamicChunkingAndIndexing(unittest.TestCase):
    def test_sliding_window_chunker(self):
        sample_text = "word " * 100
        sample_text = sample_text.strip()
        
        chunks = DynamicChunker.chunk_text(sample_text, chunk_size=40, overlap=10)
        self.assertGreater(len(chunks), 1)
        # Step size = 40 - 10 = 30
        self.assertEqual(chunks[0]["start_word_idx"], 0)
        self.assertEqual(chunks[0]["end_word_idx"], 40)
        self.assertEqual(chunks[1]["start_word_idx"], 30)

    def test_reindexing_latency_and_metrics(self):
        """Meets REQ-F1: Dynamic re-indexing in under 500ms."""
        indexer = VectorIndexer(embedding_provider=TFIDFEmbeddingProvider())
        docs = [
            "CPU Scheduling decides which process in the ready queue is allocated CPU time.",
            "Virtual memory maps a process logical address space to physical RAM via page tables.",
            "Two-Phase Locking (2PL) guarantees serializability in relational databases."
        ]
        # Quick warmup for DLL loading
        indexer.index_documents(docs, chunk_size=15, overlap=2)
        
        # Test dynamic re-indexing latency
        metrics = indexer.index_documents(docs, chunk_size=10, overlap=3)
        self.assertIn("indexing_time_ms", metrics)
        self.assertLess(metrics["indexing_time_ms"], 500.0)
        self.assertGreater(metrics["total_chunks"], 0)
        self.assertGreater(metrics["vocab_size"], 0)
        self.assertIsNotNone(indexer.pca_coords_2d)


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.engine = AdvancedDynamicRAG.from_preset("study_buddy", embedding_provider="tfidf", llm_provider="offline")

    def test_similarity_ranking_and_diagnostics(self):
        query = "How does CPU scheduling prevent starvation?"
        retrieved, metrics = self.engine.retrieve_with_metrics(query, top_k=2, threshold=0.0)
        
        self.assertEqual(len(retrieved), 2)
        # Highest scoring chunk should contain CPU / scheduling / starvation / aging
        top_chunk = retrieved[0]
        self.assertIn("score", top_chunk)
        self.assertGreater(top_chunk["score"], 0.0)
        self.assertIn("query_non_zero_components", metrics)
        self.assertIn("max_similarity_score", metrics)
        self.assertGreater(metrics["max_similarity_score"], 0.0)

    def test_threshold_cutoff(self):
        query = "How does CPU scheduling prevent starvation?"
        # High threshold should filter out low similarity chunks
        retrieved, metrics = self.engine.retrieve_with_metrics(query, top_k=5, threshold=0.99)
        self.assertEqual(len(retrieved), 0)
        self.assertEqual(metrics["chunks_above_threshold"], 0)


class TestGroundedGeneratorAndPrompts(unittest.TestCase):
    def test_prompt_provenance_markers(self):
        """Meets REQ-F4: Provenance markers [Source N] in prompt."""
        fake_sources = [
            {"source_title": "Lecture 1", "text": "Aging gradually increments process priority."},
            {"source_title": "Lecture 2", "text": "Two-Phase locking avoids cascading aborts."}
        ]
        prompt = PromptBuilder.build_prompt("Test Persona", fake_sources, "What is aging?")
        self.assertIn("[Source 1] (Lecture 1):", prompt)
        self.assertIn("[Source 2] (Lecture 2):", prompt)
        self.assertIn("What is aging?", prompt)

    def test_out_of_domain_negative_refusal(self):
        """Meets REQ-F5 and REQ-F6: Out-of-Domain Probe and negative refusal."""
        generator = GroundedGenerator(provider="offline")
        res = generator.generate("Study Partner", [], "How many miles to Mars?")
        self.assertEqual(res["answer"], "I don't have that information.")
        self.assertTrue(res["quality_metrics"]["fallback_detected"])


class TestEndToEndEngine(unittest.TestCase):
    def test_preset_switching_and_ask_loop(self):
        engine = AdvancedDynamicRAG.from_preset("study_buddy", embedding_provider="tfidf", llm_provider="offline")
        res1 = engine.ask("What is CPU scheduling?", top_k=2)
        self.assertIn("answer", res1)
        self.assertIn("retrieved_sources", res1)
        self.assertIn("token_metrics", res1)
        self.assertIn("latency_ms", res1)
        
        # Switch to E-Commerce Support
        engine.load_preset("ecommerce")
        self.assertEqual(engine.active_preset_name, "ecommerce")
        res2 = engine.ask("What is the return window for laptops and electronics?", top_k=2)
        self.assertIn("answer", res2)
        self.assertGreater(len(res2["retrieved_sources"]), 0)

    def test_negative_test_helper(self):
        engine = AdvancedDynamicRAG.from_preset("ecommerce", embedding_provider="tfidf", llm_provider="offline")
        neg_res = engine.run_negative_test(threshold=0.30)
        self.assertTrue(neg_res["quality_metrics"]["fallback_detected"])


if __name__ == "__main__":
    unittest.main()
