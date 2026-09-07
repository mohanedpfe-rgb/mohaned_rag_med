from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_project.app.resilient_rag import ResilientRAGSystem
from rag_project.configuration.settings import Settings


def setup_app() -> ResilientRAGSystem:
    settings = Settings.from_env()
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = ResilientRAGSystem(settings)
    return st.session_state.rag_system


def run_ingestion(system: ResilientRAGSystem, source_dir: str) -> None:
    with st.spinner("Indexing PDFs. Follow the Developer Console for every pipeline stage..."):
        st.session_state.ingestion_status = system.ingest_directory(source_dir)


def apply_settings(system: ResilientRAGSystem) -> None:
    updates: dict[str, object] = {
        "ollama_base_url": st.session_state.ollama_host,
        "embedding_model": st.session_state.embedding_model,
        "generation_model": st.session_state.generation_model,
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
        "lexical_mode": st.session_state.lexical_mode,
        "vector_weight": st.session_state.vector_weight,
        "neighbor_expansion": st.session_state.neighbor_expansion,
    }
    success, warnings = system.apply_settings_in_place(updates)
    for warning in warnings:
        st.warning(warning)
    st.session_state.settings_message = (
        "Settings applied to the live runtime."
        if success and not warnings
        else "Settings applied with diagnostics shown above."
    )


def main() -> None:
    st.set_page_config(page_title="BookRAG Studio", page_icon="📚", layout="wide")
    st.title("BookRAG Studio")
    st.caption("Medical document RAG with resilient retrieval and a separate developer control room.")
    system = setup_app()
    with st.sidebar:
        st.subheader("Configuration")
        defaults = {
            "ollama_host": system.settings.ollama_base_url,
            "embedding_model": system.settings.embedding_model,
            "generation_model": system.settings.generation_model,
            "chunk_size": system.settings.chunk_size,
            "chunk_overlap": system.settings.chunk_overlap,
            "top_k": system.settings.top_k,
            "temperature": system.settings.temperature,
            "lexical_mode": system.settings.lexical_mode,
            "vector_weight": system.settings.vector_weight,
            "neighbor_expansion": system.settings.neighbor_expansion,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        with st.form("configuration_form"):
            st.text_input("Ollama host", key="ollama_host")
            st.text_input("Embedding model", key="embedding_model")
            st.text_input("Generation model", key="generation_model")
            st.number_input("Chunk size", min_value=200, max_value=2000, step=50, key="chunk_size")
            st.number_input("Chunk overlap", min_value=20, max_value=300, step=10, key="chunk_overlap")
            st.number_input("Top-k retrieval", min_value=1, max_value=20, key="top_k")
            st.slider("Generation temperature", min_value=0.0, max_value=1.0, step=0.1, key="temperature")
            st.selectbox("Retrieval mode", ["hybrid", "lexical", "vector"], key="lexical_mode")
            st.slider("Vector search weight", min_value=0.0, max_value=1.0, step=0.1, key="vector_weight")
            st.checkbox("Expand neighboring chunks", key="neighbor_expansion")
            st.form_submit_button("Apply settings", on_click=apply_settings, args=(system,))
        if st.session_state.get("settings_message"):
            st.success(st.session_state.settings_message)

        uploads = st.file_uploader("Add PDF files", type=["pdf"], accept_multiple_files=True)
        if uploads:
            for upload in uploads:
                safe_name = Path(upload.name).name
                target = system.settings.incoming_dir / safe_name
                target.write_bytes(upload.getvalue())
            st.success(f"Saved {len(uploads)} PDF file(s) to the ingestion queue.")
        st.button("Ingest folder", on_click=run_ingestion, args=(system, str(system.settings.incoming_dir)))
        st.page_link("http://localhost:8501/?page=dev", label="Open Developer Console", icon="🛠️")

    tabs = st.tabs(["Chat", "Ingestion", "Settings", "Diagnostics"])
    with tabs[0]:
        st.subheader("Ask a question")
        question = st.text_area("Question", placeholder="What does the document say about methodology?")
        if st.button("Search and answer"):
            if not question.strip():
                st.warning("Enter a question before searching.")
            else:
                with st.spinner("Searching indexed documents..."):
                    result = system.answer(question)
                st.markdown("### Answer")
                st.write(result.get("answer", ""))
                if result.get("degraded_mode"):
                    st.info(f"Degraded mode: retrieval={result.get('retrieval_mode', 'unknown')}")
                confidence = result.get("confidence")
                if confidence:
                    st.metric(
                        "Evidence confidence",
                        str(confidence.get("level", "unknown")).upper(),
                        f"top={float(confidence.get('top_score', 0.0)):.3f}, margin={float(confidence.get('margin', 0.0)):.3f}",
                    )
                st.markdown("### Citations")
                for citation in result.get("citations", []):
                    st.write(citation)

    with tabs[1]:
        st.subheader("Ingestion progress")
        for item in st.session_state.get("ingestion_status", []):
            st.write(item)
            if item.get("timings_ms"):
                st.json(item["timings_ms"])
        if not st.session_state.get("ingestion_status"):
            st.info("No ingestion has run yet. Use the Developer Console for detailed live diagnostics.")

    with tabs[2]:
        st.subheader("System settings")
        st.json({
            "embedding_model": system.settings.embedding_model,
            "generation_model": system.settings.generation_model,
            "incoming_dir": str(system.settings.incoming_dir),
            "vector_db_dir": str(system.settings.vector_db_dir),
            "ingestion_lease_seconds": system.settings.ingestion_lease_seconds,
        })

    with tabs[3]:
        st.subheader("Diagnostics")
        compatibility = system.vector_store.compatibility_report(system.embedding_service.identity)
        st.metric("Indexed chunks", system.vector_store.count())
        st.json({
            "index_compatibility": compatibility["status"],
            "index_message": compatibility["message"],
            "embedding_dimension": system.embedding_service.dimension,
            "embedding_error": system.embedding_startup_error,
            "reranker_error": getattr(system.reranker, "error", None),
            "last_component_error": getattr(system, "_last_component_error", None),
        })


if __name__ == "__main__":
    main()
