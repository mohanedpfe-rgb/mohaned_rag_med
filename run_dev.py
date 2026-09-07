import rag_project.app.dev_ui as dev_ui
from rag_project.app.resilient_rag import ResilientRAGSystem

# dev_ui resolves RAGSystem from its module globals at call time, so replacing
# the symbol here makes the entire developer console observe the resilient runtime.
dev_ui.RAGSystem = ResilientRAGSystem


if __name__ == "__main__":
    dev_ui.main()
