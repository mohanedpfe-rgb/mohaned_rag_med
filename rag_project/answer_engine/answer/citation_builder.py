"""Citation builder for deterministic medical answers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class Citation:
    """A citation reference."""
    id: str
    number: int
    source_id: str
    document_id: str
    document_version: str
    page: int
    section: str
    snippet: str
    text: str
    confidence: float
    relevance: float


@dataclass(frozen=True)
class CitationGroup:
    """A group of citations for the same source."""
    source_id: str
    citations: List[Citation]
    page_numbers: List[int]
    section: str


# ── Helper: extract a field from either a dict or a NormalizedEvidence/EvidenceItem ──

def _ev_get(evidence: Any, key: str, default: Any = "") -> Any:
    """Get a field from evidence regardless of whether it is a dict or
    a NormalizedEvidence dataclass (which wraps an EvidenceItem)."""
    if isinstance(evidence, dict):
        return evidence.get(key, default)

    # NormalizedEvidence → delegate to evidence_item
    if hasattr(evidence, "evidence_item"):
        item = evidence.evidence_item
        loc = getattr(item, "source_location", None)
        mapping = {
            "id":               getattr(item, "id", default),
            "text":             getattr(item, "text", default),
            "confidence":       getattr(item, "semantic_relevance", default),
            "relevance":        getattr(item, "query_intent_match", default),
            "document_id":      getattr(loc, "document_id", default) if loc else default,
            "document_version": getattr(loc, "document_version", default) if loc else default,
            "page":             getattr(loc, "page", default) if loc else default,
            "section":          getattr(loc, "section", default) if loc else default,
        }
        return mapping.get(key, default)

    # Arbitrary object: try attribute, then default
    return getattr(evidence, key, default)


def build_citation(
    source_id: str,
    document_id: str,
    document_version: str,
    page: int,
    section: str,
    snippet: str,
    text: str,
    confidence: float = 0.0,
    relevance: float = 0.0,
    number: int = 0,
) -> Citation:
    """Build a citation object."""
    return Citation(
        id=source_id,
        number=number,
        source_id=source_id,
        document_id=document_id,
        document_version=document_version,
        page=page,
        section=section,
        snippet=snippet,
        text=text,
        confidence=confidence,
        relevance=relevance,
    )


def build_citations(
    evidence_list: List[Any],
    claim_map: Dict[str, List[str]],  # claim_id -> evidence_ids
) -> List[Citation]:
    """Build citations from evidence list.

    Accepts both plain dicts and NormalizedEvidence dataclass instances.
    """
    citations = []

    for i, evidence in enumerate(evidence_list, 1):
        source_id   = _ev_get(evidence, "id",               f"src_{i}")
        document_id = _ev_get(evidence, "document_id",      "")
        doc_version = _ev_get(evidence, "document_version", "")
        page        = _ev_get(evidence, "page",              0)
        section     = _ev_get(evidence, "section",           "")
        text        = _ev_get(evidence, "text",              "")
        confidence  = _ev_get(evidence, "confidence",        0.0)
        relevance   = _ev_get(evidence, "relevance",         0.0)

        citation = build_citation(
            source_id=str(source_id),
            document_id=str(document_id),
            document_version=str(doc_version),
            page=int(page) if page else 0,
            section=str(section),
            snippet=str(text)[:200] if text else "",
            text=str(text),
            confidence=float(confidence) if confidence else 0.0,
            relevance=float(relevance) if relevance else 0.0,
            number=i,
        )

        citations.append(citation)

    return citations


def group_citations_by_source(
    citations: List[Citation],
) -> List[CitationGroup]:
    """Group citations by their source."""
    groups: Dict[str, CitationGroup] = {}

    for citation in citations:
        source_id = citation.source_id

        if source_id not in groups:
            groups[source_id] = CitationGroup(
                source_id=source_id,
                citations=[citation],
                page_numbers=[citation.page],
                section=citation.section,
            )
        else:
            group = groups[source_id]
            if citation.page not in group.page_numbers:
                group.page_numbers.append(citation.page)
            group.citations.append(citation)

    return list(groups.values())


def format_citation_list(
    citations: List[Citation],
    style: str = "numeric",
) -> str:
    """Format a list of citations."""
    if not citations:
        return ""

    if style == "numeric":
        return _format_numeric_citations(citations)
    elif style == "author_date":
        return _format_author_date_citations(citations)
    elif style == "harvard":
        return _format_harvard_citations(citations)
    else:
        return _format_numeric_citations(citations)


def _format_numeric_citations(citations: List[Citation]) -> str:
    return " ".join(f"[{c.number}]" for c in citations if c.number > 0)


def _format_author_date_citations(citations: List[Citation]) -> str:
    formatted = []
    for c in citations:
        author = _extract_author(c.document_id)
        year = _extract_year(c.document_version)
        if author and year:
            formatted.append(f"({author}, {year})")
    return " ".join(formatted) if formatted else ""


def _format_harvard_citations(citations: List[Citation]) -> str:
    formatted = []
    for c in citations:
        author = _extract_author(c.document_id)
        year = _extract_year(c.document_version)
        page = c.page
        parts = [author] if author else []
        if year:
            parts.append(f"({year})")
        if page:
            parts.append(f"p.{page}")
        if parts:
            formatted.append(" ".join(parts))
    return "; ".join(formatted) if formatted else ""


def _extract_author(document_id: str) -> Optional[str]:
    patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"author[_-]([a-z]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, document_id)
        if match:
            return match.group(1)
    return None


def _extract_year(version: str) -> Optional[str]:
    match = re.search(r"(\d{4})", version)
    if match:
        return match.group(1)
    return None


def build_inline_citation(
    claim_id: str,
    citations: List[Citation],
    claim_map: Dict[str, List[str]],
) -> str:
    evidence_ids = claim_map.get(claim_id, [])
    supporting_citations = [c for c in citations if c.source_id in evidence_ids]
    if not supporting_citations:
        return ""
    return " ".join(f"[{c.number}]" for c in supporting_citations if c.number > 0)


def build_citation_map(
    citations: List[Citation],
) -> Dict[str, List[Citation]]:
    result: Dict[str, List[Citation]] = {}
    for citation in citations:
        source_id = citation.source_id
        if source_id not in result:
            result[source_id] = []
        result[source_id].append(citation)
    return result


def format_citation_section(
    citations: List[Citation],
    style: str = "numeric",
) -> str:
    if not citations:
        return ""
    lines = ["## References"]
    for citation in citations:
        line = _format_single_citation(citation, style)
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def _format_single_citation(citation: Citation, style: str) -> Optional[str]:
    author = _extract_author(citation.document_id)
    year = _extract_year(citation.document_version)
    title = citation.section or "Medical Document"
    if style == "numeric":
        return f"[{citation.number}] {author or 'Author'}. {title}. {year or 'Year'}. Page {citation.page}."
    elif style == "author_date":
        return f"{author or 'Author'} ({year or 'Year'}). {title}. Page {citation.page}."
    elif style == "harvard":
        return f"{author or 'Author'} {year or 'Year'}. {title}. Page {citation.page}."
    else:
        return f"[{citation.number}] {author or 'Author'}. {title}. Page {citation.page}."


def build_evidence_citations(
    evidence: List[Any],
    claims: List[Dict[str, Any]],
    claim_evidence_map: Dict[str, List[str]],
) -> tuple:
    citations = build_citations(evidence, claim_evidence_map)
    citation_map = build_citation_map(citations)
    claim_citation_map: Dict[str, List[Citation]] = {}
    for claim in claims:
        claim_id = claim.get("id", "")
        evidence_ids = claim_evidence_map.get(claim_id, [])
        claim_citation_map[claim_id] = [
            c for c in citations if c.source_id in evidence_ids
        ]
    return citations, claim_citation_map


def format_citations_for_claim(
    claim_id: str,
    claim_citation_map: Dict[str, List[Citation]],
) -> str:
    citations = claim_citation_map.get(claim_id, [])
    if not citations:
        return ""
    return " ".join(f"[{c.number}]" for c in citations if c.number > 0)
