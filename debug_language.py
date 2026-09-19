"""
Debug language detection in the answer generation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from rag_project.application import MedEvidenceProductionRAGSystem
    from rag_project.configuration.settings import Settings
    from rag_project.intelligence.med_evidence_pro import AnswerCascade
    
    print("Testing language detection...")
    
    # Test the language detection function directly
    cascade = AnswerCascade(None)
    
    test_texts = [
        "Le diabète est une maladie métabolique chronique caractérisée par une hyperglycémie.",
        "Diabetes is a chronic metabolic disease characterized by hyperglycemia.",
        "o Dpression et tats dagitation o Intolrance au glucose voire diabte.",
    ]
    
    for text in test_texts:
        detected_lang = cascade._detect_content_language(text)
        print(f"Text: {text[:50]}...")
        print(f"Detected language: {detected_lang}")
        print()
    
    print("\nNow testing with full system...")
    settings = Settings.from_env()
    system = MedEvidenceProductionRAGSystem(settings)
    
    result = system.answer("What is diabetes?")
    print(f"Status: {result.get('status')}")
    print(f"Generation Path: {result.get('generation_path')}")
    
    # Check generation meta for language info
    gen_meta = result.get('generation_meta', {})
    print(f"Generation meta: {gen_meta}")
    
    # Check if language mismatch is detected
    if gen_meta.get('language_mismatch'):
        print(f"Language mismatch detected: Content={gen_meta.get('content_language')}, Query={gen_meta.get('query_language')}")
    
    answer = result.get('answer', 'No answer')
    try:
        print(f"Answer: {answer[:400]}")
    except UnicodeEncodeError:
        print(f"Answer: {answer.encode('ascii', 'ignore').decode('ascii')[:400]}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()