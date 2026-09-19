"""
Simple test to check answer generation.
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
    print(f"Answer (raw): {repr(answer[:100])}")
    
    # Try to clean the answer
    try:
        answer_clean = answer.encode('ascii', 'ignore').decode('ascii')
        print(f"Answer (cleaned): {answer_clean[:200]}")
    except Exception as e:
        print(f"Error cleaning answer: {e}")
    
    print(f"Hits found: {len(result.get('hits', []))}")
    
    # Debug verification details
    verification = result.get('verification', {})
    print(f"\nVerification details:")
    print(f"  Allow: {verification.get('allow')}")
    print(f"  Supported ratio: {verification.get('supported_ratio')}")
    print(f"  Claim count: {verification.get('claim_count')}")
    print(f"  Blocked claims: {verification.get('blocked_claims')}")
    print(f"  Reason: {verification.get('reason')}")
    
    # Check evidence compilation
    evidence = result.get('evidence', {})
    print(f"\nEvidence details:")
    print(f"  Claim count: {evidence.get('claim_count')}")
    print(f"  Compressed context length: {len(evidence.get('compressed_context', ''))}")
    
    # Check if there's a final answer verification
    final_answer = verification.get('final_answer', {})
    print(f"\nFinal answer verification:")
    print(f"  Allow: {final_answer.get('allow')}")
    print(f"  Checked: {final_answer.get('checked')}")
    
    # Check retrieval details
    retrieval = result.get('retrieval', {})
    print(f"\nRetrieval details:")
    print(f"  Tier: {retrieval.get('tier')}")
    print(f"  Cache hit: {retrieval.get('cache_hit')}")
    print(f"  Early exit: {retrieval.get('early_exit')}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()