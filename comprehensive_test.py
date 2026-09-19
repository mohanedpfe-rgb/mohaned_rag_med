"""
Comprehensive test to verify system functionality after fixes.
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
    
    test_questions = [
        "What is diabetes?",
        "What is the mechanism of diabetes?", 
        "What is metformin?",
        "What are the symptoms of diabetes?",
        "How is diabetes treated?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Testing: {question}")
        print('='*60)
        try:
            result = system.answer(question)
            print(f"Status: {result.get('status')}")
            print(f"Generation Path: {result.get('generation_path')}")
            
            answer = result.get('answer', 'No answer')
            try:
                print(f"Answer: {answer[:200]}")
            except UnicodeEncodeError:
                print(f"Answer: {answer.encode('ascii', 'ignore').decode('ascii')[:200]}")
            
            print(f"Hits found: {len(result.get('hits', []))}")
            
            verification = result.get('verification', {})
            print(f"Verification - Allow: {verification.get('allow')}, Supported Ratio: {verification.get('supported_ratio'):.2f}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST COMPLETED")
    print('='*60)
    
except Exception as e:
    print(f"SYSTEM ERROR: {e}")
    import traceback
    traceback.print_exc()