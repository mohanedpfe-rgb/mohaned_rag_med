from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings
from rag_project.app.studio_ui import main

# Keep application startup explicit and composition-root driven.
create_rag_system(Settings.from_env())

if __name__ == "__main__":
    main()
