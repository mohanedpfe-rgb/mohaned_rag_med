from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from rag_project.retrieval.hybrid_retriever import RetrievalHit


_EVIDENCE_WRAPPER_RE = re.compile(r"<evidence id=\"S(\d+)\">(.*?)</evidence>", re.DOTALL)
NeighborResolver = Callable[[str, int, int], list[RetrievalHit] | None]


def _safe_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []


class ContextBuilder:
    def __init__(
        self,
        token_budget: int = 5000,
        max_per_document: int = 4,
        *,
        neighbor_expansion: bool = True,
        neighbor_resolver: NeighborResolver | None = None,
    ):
        self.token_budget = max(100, token_budget)
        self.max_per_document = max(1, max_per_document)
        self.neighbor_expansion = bool(neighbor_expansion)
        self.neighbor_resolver = neighbor_resolver

    def build(self, hits: Iterable[RetrievalHit] | None) -> tuple[str, list[RetrievalHit]]:
        selected: list[RetrievalHit] = []
        seen: set[str] = set()
        document_counts: dict[str, int] = {}
        used_tokens = 0
        ordered = [item for item in _safe_list(hits) if item is not None]
        if self.neighbor_expansion and self.neighbor_resolver:
            ordered = self._expand_neighbors(ordered)
        for index, hit in enumerate(ordered):
            if hit is None:
                continue
            metadata = hit.metadata or {}
            chunk_id = str(metadata.get("chunk_id", f"{hit.doc_id}:{index}"))
            document_id = str(metadata.get("document_id", hit.doc_id))
            if chunk_id in seen or document_counts.get(document_id, 0) >= self.max_per_document:
                continue
            estimated_tokens = max(1, len(str(hit.text or "").split()) * 4 // 3)
            if selected and used_tokens + estimated_tokens > self.token_budget:
                break
            seen.add(chunk_id)
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            used_tokens += estimated_tokens
            selected.append(hit)
        context = "\n\n".join(
            (
                f"<evidence id=\"S{index + 1}\" chunk_id=\"{str((hit.metadata or {}).get('chunk_id') or f'{hit.doc_id}:{index}') }\">"
                f"[{(hit.metadata or {}).get('file_name', 'unknown')} pages "
                f"{(hit.metadata or {}).get('page_numbers', [])} chunk_id={str((hit.metadata or {}).get('chunk_id') or f'{hit.doc_id}:{index}')}]\n{hit.text}\n</evidence>"
            )
            for index, hit in enumerate(selected)
        )
        return context, selected

    def _expand_neighbors(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits or not self.neighbor_resolver:
            return hits
        expanded: list[RetrievalHit] = list(hits)
        existing_ids = {
            str((hit.metadata or {}).get("chunk_id") or hit.doc_id)
            for hit in hits
            if hit is not None
        }
        for hit in hits:
            if hit is None:
                continue
            metadata = hit.metadata or {}
            document_id = str(metadata.get("document_id") or hit.doc_id)
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None:
                continue
            try:
                index = int(chunk_index)
            except (TypeError, ValueError):
                continue
            try:
                neighbors = _safe_list(self.neighbor_resolver(document_id, index, 1))
            except Exception:
                continue
            for neighbor in neighbors:
                if neighbor is None:
                    continue
                neighbor_id = str((getattr(neighbor, "metadata", {}) or {}).get("chunk_id") or getattr(neighbor, "doc_id", ""))
                if neighbor_id not in existing_ids:
                    existing_ids.add(neighbor_id)
                    expanded.append(neighbor)
        return expanded
