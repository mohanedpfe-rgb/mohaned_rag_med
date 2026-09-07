"""Compatibility entry point for the single Streamlit interface."""

from rag_project.app.dev_ui import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
