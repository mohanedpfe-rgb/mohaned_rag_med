"""Canonical Streamlit entry point for the production BookRAG interface.

This compatibility module deliberately delegates to the same application root as
``app.py`` so the production Ask flow (including Phase-5 intelligence visibility)
remains the only normal user-facing entry point.
"""

from app import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
