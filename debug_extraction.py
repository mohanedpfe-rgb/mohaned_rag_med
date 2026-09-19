"""
Debug the sentence extraction process.
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
    from rag_project.intelligence.med_evidence_pro import _sentences
    
    print("Debugging sentence extraction...")
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
    query = "diabetes"
    hits = retriever.retrieve(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(hits)} hits\n")
    
    for i, hit in enumerate(hits, 1):
        print(f"Hit {i}:")
        print(f"  Score: {hit.score:.3f}")
        print(f"  Document ID: {hit.doc_id}")
        try:
            text_clean = hit.text.encode('ascii', 'ignore').decode('ascii')
            print(f"  Text preview: {text_clean[:200]}...")
        except Exception as e:
            print(f"  Text preview: [Error: {e}]")
            
        print(f"\n  Extracted sentences:")
        sentences = _sentences(hit.text)
        for j, sentence in enumerate(sentences[:5], 1):
            try:
                sentence_clean = sentence.encode('ascii', 'ignore').decode('ascii')
                print(f"    {j}. {sentence_clean}")
            except Exception as e:
                print(f"    {j}. [Error: {e}]")
        print()
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()