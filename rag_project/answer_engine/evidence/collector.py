"""Evidence collector for deterministic medical answers."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from rag_project.retrieval.hybrid_retriever import RetrievalHit


@dataclass
class CollectedEvidence:
    """Collected evidence with grouping."""
    hits: List[RetrievalHit]
    documents: Dict[str, List[RetrievalHit]]
    pages: Dict[str, List[RetrievalHit]]
    sections: Dict[str, List[RetrievalHit]]
    tables: List[RetrievalHit]
    figures: List[RetrievalHit]
    text_chunks: List[RetrievalHit]
    confidence: float = 0.0


class EvidenceCollector:
    """Collects and organizes evidence from retrieval results."""
    
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
    
    def collect(self, hits: List[RetrievalHit]) -> CollectedEvidence:
        """Collect and organize evidence from retrieval hits."""
        hits = hits[:self.top_k]
        
        documents: Dict[str, List[RetrievalHit]] = defaultdict(list)
        pages: Dict[str, List[RetrievalHit]] = defaultdict(list)
        sections: Dict[str, List[RetrievalHit]] = defaultdict(list)
        tables: List[RetrievalHit] = []
        figures: List[RetrievalHit] = []
        text_chunks: List[RetrievalHit] = []
        
        for hit in hits:
            doc_id = str(hit.metadata.get("document_id", hit.doc_id))
            documents[doc_id].append(hit)
            
            page_numbers = hit.metadata.get("page_numbers", [])
            for page in page_numbers:
                pages[str(page)].append(hit)
            
            section = hit.metadata.get("section", "")
            if section:
                sections[section].append(hit)
            
            source_type = hit.metadata.get("source_type", "text")
            if source_type == "table":
                tables.append(hit)
            elif source_type == "figure":
                figures.append(hit)
            else:
                text_chunks.append(hit)
        
        confidence = self._estimate_confidence(hits)
        
        return CollectedEvidence(
            hits=hits,
            documents=dict(documents),
            pages=dict(pages),
            sections=dict(sections),
            tables=tables,
            figures=figures,
            text_chunks=text_chunks,
            confidence=confidence,
        )
    
    def _estimate_confidence(self, hits: List[RetrievalHit]) -> float:
        """Estimate confidence in collected evidence."""
        if not hits:
            return 0.0
        
        # Base confidence on scores
        avg_score = sum(h.score for h in hits) / len(hits)
        
        # Bonus for document diversity
        doc_count = len(set(str(h.metadata.get("document_id", h.doc_id)) for h in hits))
        diversity_bonus = min(0.15, (doc_count - 1) * 0.05)
        
        # Bonus for table/figure inclusion
        table_count = sum(1 for h in hits if h.metadata.get("source_type") == "table")
        figure_count = sum(1 for h in hits if h.metadata.get("source_type") == "figure")
        structured_bonus = min(0.1, (table_count + figure_count) * 0.05)
        
        return min(1.0, avg_score + diversity_bonus + structured_bonus)
    
    def filter_by_relevance(self, hits: List[RetrievalHit], min_score: float = 0.4) -> List[RetrievalHit]:
        """Filter hits by minimum relevance score."""
        return [h for h in hits if h.score >= min_score]
    
    def select_top_documents(self, collected: CollectedEvidence, n: int = 3) -> List[RetrievalHit]:
        """Select top evidence from n most relevant documents."""
        doc_scores: Dict[str, float] = {}
        
        for doc_id, doc_hits in collected.documents.items():
            doc_scores[doc_id] = max(h.score for h in doc_hits)
        
        top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        top_doc_ids = {doc_id for doc_id, _ in top_docs}
        
        return [h for h in collected.hits if str(h.metadata.get("document_id", h.doc_id)) in top_doc_ids]


def collect_evidence(hits: List[RetrievalHit], top_k: int = 20) -> CollectedEvidence:
    """Convenience function to collect evidence."""
    collector = EvidenceCollector(top_k)
    return collector.collect(hits)
