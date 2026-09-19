"""
Check what content is actually indexed in the vector database.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from rag_project.application import MedEvidenceProductionRAGSystem
    from rag_project.configuration.settings import Settings
    
    print("Initializing system...")
    settings = Settings.from_env()
    system = MedEvidenceProductionRAGSystem(settings)
    
    print("Checking indexed documents...")
    try:
        # Get all documents from the vector store
        raw_docs = system.vector_store.get_documents()
        ids = raw_docs.get("ids", [])
        documents = raw_docs.get("documents", [])
        metadatas = raw_docs.get("metadatas", [])
        
        print(f"Total indexed chunks: {len(ids)}")
        
        # Sample some documents to see what content is indexed
        print("\nSample indexed content:")
        for i in range(min(5, len(documents))):
            doc_text = documents[i]
            metadata = metadatas[i] if i < len(metadatas) else {}
            print(f"\n--- Document {i+1} ---")
            print(f"File: {metadata.get('file_name', 'unknown')}")
            print(f"Content preview: {doc_text[:200]}")
            
        # Search for diabetes-specific content
        print("\n\nSearching for diabetes content...")
        diabetes_hits = system.retriever.retrieve("diabetes", top_k=5)
        print(f"Found {len(diabetes_hits)} hits for 'diabetes'")
        for i, hit in enumerate(diabetes_hits[:3]):
            print(f"\n--- Hit {i+1} ---")
            print(f"Score: {hit.score}")
            print(f"Content: {hit.text[:200]}")
            
    except Exception as e:
        print(f"Error checking indexed content: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"SYSTEM ERROR: {e}")
    import traceback
    traceback.print_exc()