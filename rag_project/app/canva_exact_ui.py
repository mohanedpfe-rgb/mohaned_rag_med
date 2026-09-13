"""Compatibility entry point for the exact-answer UI."""
from .bookrag_ui import main

def render() -> None:
    main()

__all__ = ["main"]
