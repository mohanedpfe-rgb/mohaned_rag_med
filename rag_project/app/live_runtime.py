"""Compatibility entry point for live runtime views."""
from .bookrag_ui import main

def render() -> None:
    main()

__all__ = ["main"]
