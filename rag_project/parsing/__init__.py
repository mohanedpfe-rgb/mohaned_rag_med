"""Parsing package with universal PDF extraction binding."""

from rag_project.parsing import pdf_extractor as _pdf_extractor_module
from rag_project.parsing.universal_pdf_extractor import UniversalPDFExtractor

_pdf_extractor_module.PDFExtractor = UniversalPDFExtractor
