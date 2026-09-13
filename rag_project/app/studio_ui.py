"""Compatibility entry point for the production studio UI."""
from .bookrag_ui import main

def render() -> None:
    main()

__all__ = ["main"]
