import sys
sys.path.insert(0, ".")
from rag_project.configuration.settings import Settings
from rag_project.app.rag_system import RAGSystem

settings = Settings.from_env()
print("Settings loaded. Embedding model:", settings.embedding_model)
print("Building RAGSystem...")
try:
    rag = RAGSystem(settings)
    print("RAGSystem built OK")
    print("  embedding_startup_error:", rag.embedding_startup_error)
    print("  index_compatibility:", rag.index_compatibility)
    print("  RAGSystem.top_k:", rag.settings.top_k)
except Exception as e:
    print("ERROR:", type(e).__name__, e)
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\nTest answer call:")
try:
    out = rag.answer("What is diabetes?")
    print("  answer[:400]:", (out.get("answer", "") or "")[:400])
    print("  hits count:", len(out.get("hits", []) or []))
    print("  citations count:", len(out.get("citations", []) or []))
except Exception as e:
    print("  ERROR during answer:", type(e).__name__, e)
    import traceback; traceback.print_exc()
