"""
Search for diabetes-specific content in the vector store.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from rag_project.storage.vector_store import VectorStore
    from rag_project.configuration.settings import Settings
    
    print("Searching for diabetes content...")
    settings = Settings.from_env()
    vector_store = VectorStore(settings.vector_db_dir)
    
    # Get all documents
    docs = vector_store.get_documents()
    
    print(f"Total documents: {len(docs.get('ids', []))}")
    
    # Search for diabetes-related content
    diabetes_keywords = ['diabetes', 'diabete', 'diab', 'sugar', 'glucose', 'insulin']
    
    diabetes_docs = []
    for i in range(len(docs.get('ids', []))):
        text = docs.get('documents', [])[i].lower()
        metadata = docs.get('metadatas', [])[i]
        
        if any(keyword in text for keyword in diabetes_keywords):
            diabetes_docs.append({
                'index': i,
                'doc_id': docs.get('ids', [])[i],
                'text': docs.get('documents', [])[i],
                'metadata': metadata
            })
    
    print(f"\nFound {len(diabetes_docs)} documents containing diabetes-related terms")
    
    # Show the most relevant diabetes documents
    for i, doc in enumerate(diabetes_docs[:10], 1):
        print(f"\n{'=' * 60}")
        print(f"Diabetes Document {i}:")
        print(f"File: {doc['metadata'].get('file_name', 'Unknown')}")
        print(f"Page: {doc['metadata'].get('page_numbers', 'Unknown')}")
        print(f"\nText content:")
        try:
            # Try to encode as ASCII to avoid display issues
            text_clean = doc['text'].encode('ascii', 'ignore').decode('ascii')
            print(text_clean)
        except Exception as e:
            print(f"[Error displaying text: {e}]")
            print(f"Text length: {len(doc['text'])} characters")
        print(f"\n{'=' * 60}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()