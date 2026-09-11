"""Structured local knowledge sources used by the MedEvidence Pro runtime."""
from .medical_kb import connect, initialize, counts

__all__ = ["connect", "initialize", "counts"]
