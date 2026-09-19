"""
Debug the evidence compilation process.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from rag_project.storage.vector_store import VectorStore
    from rag_project.embeddings.embedding_service import EmbeddingService
    from rag_project.retrieval.hybrid_retriever import HybridRetriever
    from rag_project.configuration.settings import Settings
    from rag_project.intelligence.med_evidence_pro import EvidenceCompiler, RouteMetadata, _sentences
    
    print("Debugging evidence compilation...")
    settings = Settings.from_env()
    
    # Initialize components
    vector_store = VectorStore(settings.vector_db_dir)
    embedding_service = EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        test_mode=settings.embedding_test_mode,
    )
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        lexical_mode=settings.lexical_mode,
        vector_weight=settings.vector_weight,
    )
    
    # Test retrieval with diabetes query
    query = "What is diabetes?"
    hits = retriever.retrieve(query, top_k=10)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(hits)} hits\n")
    
    # Create a simple route metadata
    route = RouteMetadata(
        intent="factual",
        complexity=0.5,
        entities=("diabetes",),
        numeric_sensitivity=False,
        temporal_sensitivity=False,
        conditional_context=(),
        is_follow_up=False,
        confidence_threshold=0.75,
        template_type=None,
        retrieval_timeout_ms=1200,
        needs_multi_hop=False,
        query_variants=()
    )
    
    # Compile evidence
    compiler = EvidenceCompiler()
    compiled = compiler.compile(query, hits, route, {})
    
    print(f"Compiled {compiled.get('claim_count')} claims:")
    for i, claim in enumerate(compiled.get('claims', [])[:5], 1):
        try:
            text_clean = claim.text.encode('ascii', 'ignore').decode('ascii')
            print(f"  {i}. {text_clean}")
        except Exception as e:
            print(f"  {i}. [Error: {e}]")
    
    # Debug: show raw hit content
    print(f"\nFirst hit content:")
    if hits:
        try:
            hit_text_clean = hits[0].text.encode('ascii', 'ignore').decode('ascii')
            print(hit_text_clean[:500])
        except Exception as e:
            print(f"[Error: {e}]")
    
    # Debug: test sentence extraction
    print(f"\nTesting sentence extraction on first hit:")
    if hits:
        sentences = _sentences(hits[0].text)
        print(f"Extracted {len(sentences)} sentences:")
        for i, sent in enumerate(sentences[:5], 1):
            try:
                sent_clean = sent.encode('ascii', 'ignore').decode('ascii')
                print(f"  {i}. {sent_clean}")
            except Exception as e:
                print(f"  {i}. [Error: {e}]")
    
    print(f"\nCompressed answer:")
    try:
        compressed_clean = compiled.get('compressed').encode('ascii', 'ignore').decode('ascii')
        print(compressed_clean)
    except Exception as e:
        print(f"[Error: {e}]")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()