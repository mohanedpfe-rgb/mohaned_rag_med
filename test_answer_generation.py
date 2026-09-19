"""
Test to debug answer generation issues.
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
    
    print("Testing question: 'What is diabetes?'")
    result = system.answer("What is diabetes?")
    
    print(f"Status: {result.get('status')}")
    print(f"Generation Path: {result.get('generation_path')}")
    
    answer = result.get('answer', 'No answer')
    try:
        print(f"Answer: {answer[:300]}")
    except UnicodeEncodeError:
        print(f"Answer: {answer.encode('ascii', 'ignore').decode('ascii')[:300]}")
    
    print(f"\nHits found: {len(result.get('hits', []))}")
    
    # Debug verification details
    verification = result.get('verification', {})
    print(f"\nVerification details:")
    print(f"  Allow: {verification.get('allow')}")
    print(f"  Supported ratio: {verification.get('supported_ratio')}")
    print(f"  Claim count: {verification.get('claim_count')}")
    print(f"  Blocked claims: {verification.get('blocked_claims')}")
    
    # Check evidence compilation
    evidence = result.get('evidence', {})
    print(f"\nEvidence details:")
    print(f"  Claim count: {evidence.get('claim_count')}")
    print(f"  Compressed context: {evidence.get('compressed_context', '')[:200]}")
    
    # Check individual hits
    hits = result.get('hits', [])
    if hits:
        try:
            hit_text = hits[0].text if hasattr(hits[0], 'text') else str(hits[0])
            print(f"\nFirst hit text: {hit_text[:200]}")
        except Exception as e:
            print(f"\nError reading hit text: {e}")
    
    # Test with a more specific question
    print("\n\nTesting with specific question: 'What is the mechanism of diabetes?'")
    try:
        result2 = system.answer("What is the mechanism of diabetes?")
        print(f"Status: {result2.get('status')}")
        answer2 = result2.get('answer', 'No answer')
        try:
            print(f"Answer: {answer2[:200]}")
        except UnicodeEncodeError:
            print(f"Answer: {answer2.encode('ascii', 'ignore').decode('ascii')[:200]}")
    except Exception as e:
        print(f"Error in second test: {e}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()