"""
OmniRAG Studio - Streamlit Dashboard
Interactive Advanced RAG Engine with Dynamic Tuning, Multi-Domain Switcher,
and Real-Time 4-Tier Observability Metrics (Token Economics, 2D PCA Vectors,
Retrieval Distributions, Faithfulness).
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os

from config import PRESETS, PersonaPreset
from loaders import DocumentLoader
from embeddings import get_embedding_provider
from engine import AdvancedDynamicRAG

# Page Configuration
st.set_page_config(
    page_title="OmniRAG Studio | Dynamic RAG & Observability",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .chunk-card {
        background: #181b22;
        border-left: 4px solid #6366f1;
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .citation-tag {
        background: #312e81;
        color: #c7d2fe;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .score-badge {
        background: #064e3b;
        color: #a7f3d0;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if "engine" not in st.session_state:
    st.session_state.engine = AdvancedDynamicRAG.from_preset(
        preset_key="study_buddy",
        embedding_provider="tfidf",
        llm_provider="offline"
    )
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "current_preset_key" not in st.session_state:
    st.session_state.current_preset_key = "study_buddy"


engine: AdvancedDynamicRAG = st.session_state.engine

# ==========================================
# SIDEBAR: DYNAMIC CONTROL PANEL
# ==========================================
with st.sidebar:
    st.title("⚡ OmniRAG Controls")
    st.caption("Dynamic Runtime Tuning & Ingestion")

    # Domain / Persona Switcher
    preset_choice = st.selectbox(
        "Domain & Persona Preset",
        options=["study_buddy", "ecommerce"],
        format_func=lambda x: PRESETS[x].name,
        index=0 if st.session_state.current_preset_key == "study_buddy" else 1
    )

    if preset_choice != st.session_state.current_preset_key:
        st.session_state.current_preset_key = preset_choice
        engine.load_preset(preset_choice)
        st.session_state.last_response = None
        st.rerun()

    active_preset = PRESETS[preset_choice]
    st.info(f"**Focus:** {active_preset.description}")

    st.markdown("---")
    st.subheader("🛠️ Chunking & Indexing")
    
    chunk_size = st.slider(
        "Chunk Size (words)",
        min_value=20,
        max_value=300,
        value=engine.active_chunk_size,
        step=5,
        help="Target length of each atomic text passage."
    )

    overlap = st.slider(
        "Chunk Overlap (words)",
        min_value=0,
        max_value=min(80, chunk_size - 1),
        value=min(engine.active_overlap, chunk_size - 1),
        step=2,
        help="Sliding step overlap to prevent context boundary cuts."
    )

    if overlap >= chunk_size * 0.5:
        st.warning(f"⚠️ High overlap ({overlap}/{chunk_size} words = {int(overlap/chunk_size*100)}%). Redundant tokens may increase cost.")

    # Re-index if chunking sliders changed
    if chunk_size != engine.active_chunk_size or overlap != engine.active_overlap:
        with st.spinner("Re-indexing corpus in memory..."):
            idx_metrics = engine.reindex(max_words=chunk_size, overlap=overlap)
            st.toast(f"Re-indexed {idx_metrics['total_chunks']} chunks in {idx_metrics['indexing_time_ms']}ms", icon="⚡")

    st.markdown("---")
    st.subheader("🔍 Retrieval Hyperparameters")

    top_k = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=8,
        value=active_preset.default_top_k,
        help="Number of most similar chunks injected into prompt."
    )

    threshold = st.slider(
        "Relevance Cutoff Threshold (τ)",
        min_value=0.00,
        max_value=0.90,
        value=active_preset.default_threshold,
        step=0.05,
        help="Minimum cosine similarity required for passage inclusion."
    )

    st.markdown("---")
    st.subheader("🧠 Models & Backends")

    embed_choice = st.selectbox(
        "Embedding Provider",
        options=["tfidf", "sentence-transformers", "openai"],
        format_func=lambda x: {
            "tfidf": "TF-IDF (Sparse, Lexical)",
            "sentence-transformers": "MiniLM-L6-v2 (Dense 384d, Local)",
            "openai": "OpenAI text-embedding-3 (Dense 1536d)"
        }.get(x, x),
        index=0
    )

    llm_choice = st.selectbox(
        "LLM Generator",
        options=["offline", "anthropic", "openai"],
        format_func=lambda x: {
            "offline": "Offline Grounded Engine (Instant, Zero API Cost)",
            "anthropic": "Anthropic Claude 3.5 Sonnet",
            "openai": "OpenAI GPT-4o"
        }.get(x, x),
        index=0
    )

    api_key_input = None
    if llm_choice in ("anthropic", "openai") or embed_choice == "openai":
        api_key_input = st.text_input(
            f"{'Anthropic' if llm_choice == 'anthropic' else 'OpenAI'} API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY" if llm_choice == "anthropic" else "OPENAI_API_KEY", "")
        )

    # Apply engine provider switches if modified
    if embed_choice != getattr(engine.embedding_provider, "provider_name", ""):
        if st.button("Apply Backend Providers"):
            engine.set_embedding_provider(embed_choice, api_key=api_key_input)
            engine.set_llm_provider(llm_choice, api_key=api_key_input)
            st.rerun()

    st.markdown("---")
    st.subheader("📂 Custom Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Ingest Uploaded Documents"):
            for uf in uploaded_files:
                doc = DocumentLoader.auto_load(uf, filename=uf.name)
                engine.add_document(doc)
            engine.reindex(max_words=chunk_size, overlap=overlap)
            st.success(f"Ingested {len(uploaded_files)} files! Total corpus re-indexed.")
            st.rerun()


# ==========================================
# MAIN INTERFACE: QUERY & DASHBOARD
# ==========================================
st.title("⚡ OmniRAG Studio")
st.markdown(f"**Active Workspace:** `{active_preset.name}` | **Persona:** *{active_preset.persona_prompt}*")

# Query Input Box
col_q1, col_q2 = st.columns([5, 1])
with col_q1:
    default_q = (
        "How does the operating system mitigate starvation in priority scheduling?"
        if preset_choice == "study_buddy"
        else "What is the return window for electronics and laptops?"
    )
    query_text = st.text_input("Enter Question / Prompt:", value=default_q)

with col_q2:
    st.write("")
    st.write("")
    ask_button = st.button("Ask Query 🚀", use_container_width=True, type="primary")

# Dedicated Negative Test Probe (REQ-F6)
col_sub1, col_sub2 = st.columns([4, 2])
with col_sub2:
    if st.button("🧪 Run Negative Probe Test (REQ-F6)", use_container_width=True):
        with st.spinner("Executing out-of-domain probe..."):
            neg_res = engine.run_negative_test(top_k=top_k, threshold=threshold)
            st.session_state.last_response = neg_res

if ask_button and query_text:
    with st.spinner("Executing RAG retrieval and synthesis..."):
        res = engine.ask(query_text, top_k=top_k, threshold=threshold)
        st.session_state.last_response = res

# ==========================================
# OBSERVABILITY PANELS & RESULTS
# ==========================================
res = st.session_state.last_response

if res:
    st.markdown("---")
    
    # 1. Answer Card
    ans_col1, ans_col2 = st.columns([3, 1])
    with ans_col1:
        st.subheader("💡 Generated Grounded Response")
        st.markdown(f"> {res['answer']}")
    with ans_col2:
        st.subheader("⚡ Latency Profile")
        st.metric("Total Roundtrip", f"{res['latency_ms']} ms")
        ret_ms = res['retrieval_metrics'].get('retrieval_latency_ms', 0)
        st.caption(f"Vector Retrieval: **{ret_ms} ms**")

    st.markdown("---")
    st.subheader("📊 4-Tier Real-Time Observability Suite")

    tab_tokens, tab_vectors, tab_retrieval, tab_quality, tab_inspector = st.tabs([
        "1. Token Economics",
        "2. Vector Dynamics & 2D PCA",
        "3. Retrieval Analytics",
        "4. Quality & Grounding",
        "5. Chunk Inspector"
    ])

    # TIER 1: TOKEN ECONOMICS
    with tab_tokens:
        tok = res["token_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prompt Tokens", tok["prompt_tokens"])
        c2.metric("Retrieved Context Tokens", tok["context_tokens"])
        c3.metric("Completion Tokens", tok["completion_tokens"])
        c4.metric("Estimated Cost", f"${tok['estimated_cost_usd']:.6f}")
        
        # Token distribution chart
        tok_df = pd.DataFrame({
            "Token Category": ["Prompt Overhead", "Retrieved Context", "Output Completion"],
            "Count": [
                max(0, tok["prompt_tokens"] - tok["context_tokens"]),
                tok["context_tokens"],
                tok["completion_tokens"]
            ]
        })
        st.bar_chart(tok_df.set_index("Token Category"))

    # TIER 2: VECTOR DYNAMICS & 2D PCA PROJECTION
    with tab_vectors:
        rm = res["retrieval_metrics"]
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Embedding Dimension", rm["vocabulary_dimension"])
        vc2.metric("Query Active Components", rm["query_non_zero_components"])
        vc3.metric("Chunks Above Threshold", f"{rm['chunks_above_threshold']} / {rm['total_chunks']}")

        if rm.get("matched_terms"):
            st.write(f"**Matched Vocabulary Terms:** `{', '.join(rm['matched_terms'])}`")

        # 2D PCA Scatterplot
        st.write("##### 2D PCA Embedding Space Projection")
        if engine.indexer.pca_coords_2d is not None and len(engine.indexer.pca_coords_2d) > 0:
            coords = engine.indexer.pca_coords_2d
            df_points = []
            
            # All background chunks
            for i, c in enumerate(engine.indexer.chunks):
                df_points.append({
                    "PCA-1": coords[i, 0],
                    "PCA-2": coords[i, 1],
                    "Category": "Background Chunk",
                    "Label": f"Chunk {c['chunk_id']}: {c['text'][:40]}..."
                })
            
            # Mark retrieved chunks
            ret_ids = {r["global_chunk_id"] for r in res["retrieved_sources"]}
            for pt in df_points:
                # If matched
                for r in res["retrieved_sources"]:
                    if r["text"] in pt["Label"]:
                        pt["Category"] = f"Retrieved (Score: {r['score']})"
                        
            # Query point
            if rm.get("query_2d_coord"):
                df_points.append({
                    "PCA-1": rm["query_2d_coord"][0],
                    "PCA-2": rm["query_2d_coord"][1],
                    "Category": "Query Vector",
                    "Label": "User Question"
                })

            chart_data = pd.DataFrame(df_points)
            st.scatter_chart(chart_data, x="PCA-1", y="PCA-2", color="Category")
        else:
            st.info("At least 2 chunks required for 2D PCA projection.")

    # TIER 3: RETRIEVAL ANALYTICS
    with tab_retrieval:
        rm = res["retrieval_metrics"]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Max Similarity Score", rm["max_similarity_score"])
        rc2.metric("Mean Similarity Score", rm["mean_similarity_score"])
        rc3.metric("Min Similarity Score", rm["min_similarity_score"])

        st.write("##### Similarity Scores Distribution")
        if rm["all_scores"]:
            scores_df = pd.DataFrame({
                "Chunk Index": [f"Chunk {i}" for i in range(len(rm["all_scores"]))],
                "Cosine Similarity": rm["all_scores"]
            })
            st.bar_chart(scores_df.set_index("Chunk Index"))

        st.write("##### Retrieved Chunks Injected into Context Window:")
        if res["retrieved_sources"]:
            for i, src in enumerate(res["retrieved_sources"]):
                st.markdown(f"""
                <div class="chunk-card">
                    <span class="citation-tag">[Source {i+1}]</span> 
                    <strong>{src.get('source_title', 'Document')}</strong> — 
                    <span class="score-badge">Cosine: {src['score']}</span>
                    <p style="margin-top: 8px; color: #d1d5db;">{src['text']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No chunks passed the relevance threshold τ. Zero context was injected into the prompt.")

    # TIER 4: QUALITY & FAITHFULNESS
    with tab_quality:
        qm = res["quality_metrics"]
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Grounded Faithfulness", f"{qm['grounded_faithfulness_score']}%")
        qc2.metric("Citations Detected", len(qm["citations_detected"]))
        qc3.metric("Fallback Triggered", "YES (Refusal)" if qm["fallback_detected"] else "NO")

        if qm["fallback_detected"]:
            st.info("🛡️ **Zero-Hallucination Safe Mode Activated**: The query was determined to be out-of-scope or under-threshold, and the system correctly responded with 'I don't have that information.'")
        elif qm["has_citations"]:
            st.success(f"✅ **Grounding Verified**: Response successfully attributed evidence to source tags: `{qm['citations_detected']}`")

        with st.expander("Inspect Raw Prompt Injected to LLM"):
            st.code(res["prompt"], language="markdown")

    # TIER 5: CHUNK INSPECTOR (REQ-F2)
    with tab_inspector:
        st.write(f"##### Corpus Chunk Inspector ({len(engine.indexer.chunks)} total chunks)")
        st.caption("Inspect boundaries, word counts, and overlap steps.")
        for c in engine.indexer.chunks:
            boundary_color = "#3730a3" if c.get("is_overlap_boundary") else "#1e293b"
            st.markdown(f"""
            <div style="background: {boundary_color}; border-radius: 6px; padding: 10px; margin-bottom: 8px; border: 1px solid #4338ca;">
                <strong>Chunk {c['chunk_id']}</strong> ({c['word_count']} words, words [{c['start_word_idx']}..{c['end_word_idx']}]) — <em>{c.get('source_title', '')}</em>
                <p style="color: #cbd5e1; margin-top: 4px;">{c['text']}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Enter a query above or click **Ask Query 🚀** or **🧪 Run Negative Probe Test** to view the live RAG response and observability metrics.")
