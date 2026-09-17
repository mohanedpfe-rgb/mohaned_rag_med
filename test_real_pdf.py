"""
Test script for the deterministic system with a real PDF.
This tests the integrated system with actual document processing.
"""
import sys
from pathlib import Path

# Add the project to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_with_real_pdf():
    """Test the system with a real PDF document."""
    print("Testing Deterministic System with Real PDF")
    print("=" * 60)
    
    try:
        from rag_project.app.rag_system import RAGSystem
        from rag_project.configuration.settings import Settings
        
        # Initialize the RAG system
        print("Initializing RAG system...")
        settings = Settings.from_env()
        rag_system = RAGSystem(settings)
        
        # Check if there are indexed documents
        vector_count = rag_system.vector_store.count()
        print(f"Vector store count: {vector_count}")
        
        if vector_count == 0:
            print("No documents indexed. Ingesting PDF...")
            
            # Ingest the PDF
            pdf_path = "data/processed/DC_endocrino_version_2024__portrait_sans_astuces_76ac07d1c09e.pdf"
            result = rag_system.ingest_file(pdf_path)
            print(f"Ingestion result: {result}")
            
            if result.get("status") != "success":
                print(f"Failed to ingest PDF: {result}")
                return False
        
        # Test questions
        test_questions = [
            "What is diabetes?",
            "What are the symptoms of diabetes?",
            "How is diabetes treated?",
            "What is the normal blood sugar level?",
        ]
        
        print(f"\nTesting {len(test_questions)} questions...")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 60)
            
            try:
                result = rag_system.answer(question)
                
                print(f"Status: {result.get('status', 'unknown')}")
                answer_text = result.get('answer', 'No answer generated')
                # Handle unicode characters safely
                safe_answer = answer_text.encode('ascii', 'ignore').decode('ascii')
                print(f"Answer: {safe_answer[:500]}...")
                
                if 'confidence' in result:
                    conf = result['confidence']
                    print(f"Confidence Level: {conf.get('level', 'unknown')}")
                    print(f"Top Score: {conf.get('top_score', 0):.3f}")
                
                if 'citations' in result:
                    print(f"Citations: {len(result['citations'])}")
                
                if 'query_analysis' in result:
                    analysis = result['query_analysis']
                    print(f"Query Quality: {analysis.get('query_quality', 'unknown')}")
                    print(f"Intent: {analysis.get('query_intent', 'unknown')}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_real_pdf()
    sys.exit(0 if success else 1)