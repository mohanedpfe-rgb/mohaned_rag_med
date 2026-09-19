"""
Diagnostic script to identify why the RAG system is failing to answer questions from PDFs.
This script tests each component of the pipeline individually.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_vector_store():
    """Test if vector store has documents and can retrieve them."""
    print("=" * 60)
    print("TEST 1: Vector Store Status")
    print("=" * 60)
    
    try:
        from rag_project.storage.vector_store import VectorStore
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        vector_store = VectorStore(settings.vector_db_dir)
        
        # Check document count
        total_count = vector_store.count()
        lexical_count = vector_store.lexical_count()
        
        print(f"Total documents in vector store: {total_count}")
        print(f"Documents in lexical index: {lexical_count}")
        
        if total_count == 0:
            print("[X] ISSUE: No documents found in vector store!")
            print("   -> PDFs may not have been ingested properly")
            return False
        
        # Try to get some sample documents
        try:
            sample_docs = vector_store.get_documents()
            doc_ids = sample_docs.get('ids', [])
            print(f"Sample document IDs: {doc_ids[:3] if doc_ids else 'None'}")
            
            if doc_ids:
                print(f"[OK] Vector store contains {total_count} documents")
                return True
            else:
                print("[X] ISSUE: Vector store reports documents but returns empty list")
                return False
        except Exception as e:
            print(f"[X] ERROR retrieving documents: {e}")
            return False

    except Exception as e:
        print(f"[X] ERROR accessing vector store: {e}")
        return False

def test_embedding_service():
    """Test if embedding service is working."""
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Service")
    print("=" * 60)
    
    try:
        from rag_project.embeddings.embedding_service import EmbeddingService
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        
        try:
            embedding_service = EmbeddingService(
                base_url=settings.ollama_base_url,
                model=settings.embedding_model,
                test_mode=settings.embedding_test_mode,
            )
            
            # Test embedding a simple query
            test_query = "diabetes treatment"
            embedding = embedding_service.embed_query(test_query)
            
            if embedding and len(embedding) > 0:
                print(f"[OK] Embedding service working (dimension: {len(embedding)})")
                print(f"   Model: {settings.embedding_model}")
                print(f"   Test query: '{test_query}'")
                return True
            else:
                print("[X] ISSUE: Embedding service returned empty vector")
                return False

        except Exception as e:
            print(f"[X] ERROR with embedding service: {e}")
            print(f"   -> Ollama may not be running or model not available")
            return False

    except Exception as e:
        print(f"[X] ERROR initializing embedding service: {e}")
        return False

def test_retrieval():
    """Test if retrieval system can find documents."""
    print("\n" + "=" * 60)
    print("TEST 3: Retrieval System")
    print("=" * 60)
    
    try:
        from rag_project.storage.vector_store import VectorStore
        from rag_project.embeddings.embedding_service import EmbeddingService
        from rag_project.retrieval.hybrid_retriever import HybridRetriever
        from rag_project.configuration.settings import Settings
        
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
        
        # Test retrieval with a simple medical query
        test_queries = [
            "diabetes",
            "treatment",
            "diabetes treatment",
            "metformin",
        ]
        
        for query in test_queries:
            try:
                hits = retriever.retrieve(query, top_k=3)
                print(f"Query: '{query}' -> Found {len(hits)} hits")
                
                if hits:
                    print(f"  [OK] Top hit score: {hits[0].score:.3f}")
                    print(f"  [OK] Top hit text preview: {hits[0].text[:100]}...")
                    return True
            except Exception as e:
                print(f"  [X] Error with query '{query}': {e}")

        print("[X] ISSUE: No documents retrieved for any test query")
        print("   -> Vector search may not be working properly")
        return False

    except Exception as e:
        print(f"[X] ERROR with retrieval system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_answer_pipeline():
    """Test the answer pipeline directly."""
    print("\n" + "=" * 60)
    print("TEST 4: Answer Pipeline")
    print("=" * 60)
    
    try:
        from rag_project.application import MedEvidenceProductionRAGSystem
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        system = MedEvidenceProductionRAGSystem(settings)
        
        # Test with a simple question
        test_question = "What is diabetes?"
        
        print(f"Testing question: '{test_question}'")
        
        try:
            result = system.answer(test_question)
            
            print(f"Status: {result.get('status')}")
            print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
            print(f"Hits found: {len(result.get('hits', []))}")
            print(f"Generation path: {result.get('generation_path')}")
            
            if result.get('status') in ['SUCCESS', 'SUCCESS_WITH_WARNINGS']:
                print("[OK] Answer pipeline working")
                return True
            else:
                print(f"[X] ISSUE: Answer pipeline returned status: {result.get('status')}")
                print(f"   Error details: {result.get('error', 'No error info')}")
                return False

        except Exception as e:
            print(f"[X] ERROR executing answer pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"[X] ERROR initializing answer system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_ingestion():
    """Test if PDF ingestion is working."""
    print("\n" + "=" * 60)
    print("TEST 5: PDF Ingestion Status")
    print("=" * 60)
    
    try:
        from rag_project.ingestion.state_store import IngestionStateStore
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        state_store = IngestionStateStore(settings.ingestion_db_path)
        
        # Get all documents
        documents = state_store.get_all_documents()
        
        print(f"Total documents in state store: {len(documents)}")
        
        if not documents:
            print("[X] ISSUE: No documents found in state store")
            print("   -> No PDFs have been ingested")
            return False
        
        # Check document statuses
        ready_count = sum(1 for doc in documents if doc.get('status') == 'READY')
        failed_count = sum(1 for doc in documents if doc.get('status') == 'FAILED')
        running_count = sum(1 for doc in documents if doc.get('status') == 'RUNNING')
        
        print(f"READY documents: {ready_count}")
        print(f"FAILED documents: {failed_count}")
        print(f"RUNNING documents: {running_count}")
        
        if ready_count > 0:
            print("[OK] PDF ingestion has processed documents successfully")

            # Show sample document info
            sample_doc = documents[0]
            print(f"\nSample document:")
            print(f"  File: {sample_doc.get('file_name')}")
            print(f"  Status: {sample_doc.get('status')}")
            print(f"  Chunks: {sample_doc.get('chunk_count', 'N/A')}")
            print(f"  Pages: {sample_doc.get('total_pages', 'N/A')}")

            return True
        else:
            print("[X] ISSUE: No documents are in READY state")
            print("   -> PDFs may have failed during ingestion")
            return False

    except Exception as e:
        print(f"[X] ERROR accessing state store: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("RAG SYSTEM DIAGNOSTIC TOOL")
    print("=" * 60)
    print("This tool will test each component of your RAG system")
    print("to identify why it's failing to answer questions from PDFs.\n")
    
    results = {
        "Vector Store": test_vector_store(),
        "Embedding Service": test_embedding_service(),
        "Retrieval System": test_retrieval(),
        "Answer Pipeline": test_answer_pipeline(),
        "PDF Ingestion": test_pdf_ingestion(),
    }
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "[OK] PASS" if passed else "[X] FAIL"
        print(f"{status}: {component}")

    failed_components = [comp for comp, passed in results.items() if not passed]

    if failed_components:
        print(f"\n[X] ISSUES FOUND IN: {', '.join(failed_components)}")
        print("\nRECOMMENDED ACTIONS:")

        if "Vector Store" in failed_components:
            print("  1. Check if PDFs have been ingested (use the UI to upload PDFs)")
            print("  2. Verify vector store directory exists and is accessible")

        if "Embedding Service" in failed_components:
            print("  1. Ensure Ollama is running: ollama serve")
            print("  2. Check if embedding model is available: ollama list")
            print(f"  3. Pull the required model: ollama pull {Settings.from_env().embedding_model}")

        if "Retrieval System" in failed_components:
            print("  1. Check vector store and embedding service are working")
            print("  2. Verify documents have been properly indexed")

        if "Answer Pipeline" in failed_components:
            print("  1. Check if retrieval system is returning results")
            print("  2. Review answer pipeline logs for specific errors")

        if "PDF Ingestion" in failed_components:
            print("  1. Upload PDFs through the UI")
            print("  2. Check ingestion logs for errors")
            print("  3. Verify PDF files are not corrupted")
    else:
        print("\n[OK] ALL COMPONENTS PASSED")
        print("The system should be working. If you're still experiencing issues,")
        print("please provide a specific question that's failing for further analysis.")

if __name__ == "__main__":
    main()