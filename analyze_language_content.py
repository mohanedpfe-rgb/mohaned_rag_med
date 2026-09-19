"""
Analyze the language distribution of indexed content.
"""
import sys
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from rag_project.application import MedEvidenceProductionRAGSystem
    from rag_project.configuration.settings import Settings
    
    print("Initializing system...")
    settings = Settings.from_env()
    system = MedEvidenceProductionRAGSystem(settings)
    
    print("Analyzing indexed content language distribution...")
    try:
        # Get all documents from the vector store
        raw_docs = system.vector_store.get_documents()
        ids = raw_docs.get("ids", [])
        documents = raw_docs.get("documents", [])
        metadatas = raw_docs.get("metadatas", [])
        
        print(f"Total indexed chunks: {len(ids)}")
        
        # Analyze language distribution
        french_count = 0
        english_count = 0
        other_count = 0
        
        french_keywords = ['diabète', 'le', 'la', 'les', 'et', 'est', 'sont', 'pour', 'avec', 'dans']
        english_keywords = ['diabetes', 'the', 'and', 'is', 'are', 'for', 'with', 'in', 'of']
        
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            french_score = sum(1 for word in french_keywords if word in doc_lower)
            english_score = sum(1 for word in english_keywords if word in doc_lower)
            
            if french_score > english_score:
                french_count += 1
            elif english_score > french_score:
                english_count += 1
            else:
                other_count += 1
        
        print(f"\nLanguage distribution:")
        print(f"French content: {french_count} ({french_count/len(documents)*100:.1f}%)")
        print(f"English content: {english_count} ({english_count/len(documents)*100:.1f}%)")
        print(f"Other/unclear: {other_count} ({other_count/len(documents)*100:.1f}%)")
        
        # Check file names
        file_names = set()
        for metadata in metadatas:
            file_name = metadata.get('file_name', 'unknown')
            file_names.add(file_name)
        
        print(f"\nIndexed files ({len(file_names)} unique):")
        for file_name in sorted(file_names):
            print(f"  - {file_name}")
            
        # Check for diabetes content in different languages
        print("\n\nSearching for diabetes content in different languages:")
        
        # English search
        english_hits = system.retriever.retrieve("diabetes", top_k=3)
        print(f"English 'diabetes' search: {len(english_hits)} hits")
        if english_hits:
            print(f"  First hit: {english_hits[0].text[:100]}")
        
        # French search
        french_hits = system.retriever.retrieve("diabète", top_k=3)
        print(f"French 'diabète' search: {len(french_hits)} hits")
        if french_hits:
            print(f"  First hit: {french_hits[0].text[:100]}")
            
    except Exception as e:
        print(f"Error analyzing content: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"SYSTEM ERROR: {e}")
    import traceback
    traceback.print_exc()