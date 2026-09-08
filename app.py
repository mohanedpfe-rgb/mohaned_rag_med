from rag_project.runtime import install as install_runtime

install_runtime()

from rag_project.app.dev_ui import main

if __name__ == "__main__":
    main()
