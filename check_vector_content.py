"""
Check the actual content stored in the vector store.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from rag_project.storage.vector_store import VectorStore
    from rag_project.configuration.settings import Settings
    
    print("Checking vector store content...")
    settings = Settings.from_env()
    vector_store = VectorStore(settings.vector_db_dir)
    
    # Get sample documents
    docs = vector_store.get_documents()
    
    print(f"Total documents: {len(docs.get('ids', []))}")
    
    # Check first few documents in detail
    for i in range(min(5, len(docs.get('ids', [])))):
        doc_id = docs.get('ids', [])[i]
        text = docs.get('documents', [])[i]
        metadata = docs.get('metadatas', [])[i]
        
        print(f"\n{'=' * 60}")
        print(f"Document {i+1}: {doc_id[:30]}...")
        print(f"File: {metadata.get('file_name', 'Unknown')}")
        print(f"Page: {metadata.get('page_numbers', 'Unknown')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'Unknown')}")
        print(f"Index State: {metadata.get('index_state', 'Unknown')}")
        print(f"\nFull text ({len(text)} chars):")
        print(text)
        print(f"\n{'=' * 60}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()