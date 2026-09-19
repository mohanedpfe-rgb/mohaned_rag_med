"""
Test script to analyze answer quality and identify specific issues.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_specific_questions():
    """Test the system with specific medical questions."""
    print("=" * 60)
    print("TESTING SPECIFIC MEDICAL QUESTIONS")
    print("=" * 60)
    
    try:
        from rag_project.application import MedEvidenceProductionRAGSystem
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        system = MedEvidenceProductionRAGSystem(settings)
        
        test_questions = [
            "What is diabetes?",
            "What are the symptoms of diabetes?",
            "How is diabetes treated?",
            "What is metformin?",
            "What causes diabetes?",
        ]
        
        for question in test_questions:
            print(f"\n{'=' * 60}")
            print(f"Question: {question}")
            print('=' * 60)
            
            try:
                result = system.answer(question)
                
                print(f"Status: {result.get('status')}")
                print(f"Generation Path: {result.get('generation_path')}")
                answer = result.get('answer', 'No answer')
                # Clean answer for display
                answer_clean = answer.encode('ascii', 'ignore').decode('ascii')
                print(f"\nAnswer: {answer_clean}")
                print(f"\nHits found: {len(result.get('hits', []))}")
                
                # Show top hit details
                hits = result.get('hits', [])
                if hits:
                    try:
                        hit_text = hits[0].text if hasattr(hits[0], 'text') else str(hits[0])
                        hit_text_clean = hit_text.encode('ascii', 'ignore').decode('ascii')[:200]
                        print(f"\nTop hit text: {hit_text_clean}...")
                        print(f"Top hit score: {hits[0].score if hasattr(hits[0], 'score') else 'N/A'}")
                    except Exception as e:
                        print(f"\nError displaying hit: {e}")
                
                # Check for warnings/errors
                if result.get('errors'):
                    print(f"\nErrors: {result.get('errors')}")
                if result.get('warnings'):
                    print(f"\nWarnings: {result.get('warnings')}")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"ERROR initializing system: {e}")
        import traceback
        traceback.print_exc()

def test_raw_retrieval():
    """Test raw retrieval to see what content is actually being retrieved."""
    print("\n" + "=" * 60)
    print("TESTING RAW RETRIEVAL")
    print("=" * 60)
    
    try:
        from rag_project.storage.vector_store import VectorStore
        from rag_project.embeddings.embedding_service import EmbeddingService
        from rag_project.retrieval.hybrid_retriever import HybridRetriever
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        
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
        
        # Test with diabetes query
        query = "diabetes"
        hits = retriever.retrieve(query, top_k=5)
        
        print(f"Query: '{query}'")
        print(f"Retrieved {len(hits)} hits\n")
        
        for i, hit in enumerate(hits, 1):
            print(f"Hit {i}:")
            print(f"  Score: {hit.score:.3f}")
            print(f"  Document ID: {hit.doc_id}")
            try:
                text_clean = hit.text.encode('ascii', 'ignore').decode('ascii')[:150]
                print(f"  Text preview: {text_clean}...")
            except Exception as e:
                print(f"  Text preview: [Encoding error: {e}]")
            print(f"  Metadata: {hit.metadata}")
            print()
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_encoding_issues():
    """Test for encoding issues in the stored content."""
    print("\n" + "=" * 60)
    print("TESTING ENCODING ISSUES")
    print("=" * 60)
    
    try:
        from rag_project.storage.vector_store import VectorStore
        from rag_project.configuration.settings import Settings
        
        settings = Settings.from_env()
        vector_store = VectorStore(settings.vector_db_dir)
        
        # Get sample documents
        docs = vector_store.get_documents()
        
        print(f"Total documents: {len(docs.get('ids', []))}")
        
        # Check first few documents for encoding issues
        for i in range(min(3, len(docs.get('ids', [])))):
            doc_id = docs.get('ids', [])[i]
            text = docs.get('documents', [])[i]
            metadata = docs.get('metadatas', [])[i]
            
            print(f"\nDocument {i+1}: {doc_id[:20]}...")
            print(f"File: {metadata.get('file_name', 'Unknown')}")
            try:
                text_clean = text.encode('ascii', 'ignore').decode('ascii')[:200]
                print(f"Text preview: {text_clean}")
            except Exception as e:
                print(f"Text preview: [Encoding error: {e}]")
            
            # Check for encoding issues
            try:
                text.encode('utf-8').decode('utf-8')
                print("Encoding: OK")
            except UnicodeError as e:
                print(f"Encoding Issue: {e}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_questions()
    test_raw_retrieval()
    test_encoding_issues()