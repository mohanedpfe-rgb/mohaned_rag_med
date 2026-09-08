from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence


_NUM = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
_MEAS = re.compile(
    r"(?P<value>[-+]?\d+(?:[\.,]\d+)?(?:\s*[-–]\s*\d+(?:[\.,]\d+)?)?)\s*"
    r"(?P<unit>mcg|µg|ug|mg|g|kg|ml|l|mmhg|cmh2o|%|bpm|°c|c|mm|cm|m|hz|khz|m/s|h|min|s|day|days|week|weeks|month|months|year|years)\b",
    re.I,
)
_WORD = re.compile(r"[\wÀ-ÿ][\wÀ-ÿ'/-]{1,}", re.U)
_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


@dataclass(frozen=True)
class TableCell:
    row: int
    column: int
    value: str
    numeric_value: float | None = None
    unit: str = ""


@dataclass(frozen=True)
class TableEvidence:
    table_id: str
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    cells: tuple[TableCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FigureEvidence:
    figure_id: str
    caption: str
    references: tuple[str, ...]
    nearby_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutedDocument:
    document_id: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceDecision:
    allow: bool
    reason: str
    score: float
    contradictions: tuple[str, ...] = ()
    blocked_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class HopEvidence:
    hop: int
    hit: Any
    edge_reason: str


def _float(value: str) -> float | None:
    try:
        return float(value.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _unit(unit: str) -> str:
    aliases = {"µg": "ug", "mcg": "ug", "°c": "c", "liter": "l", "litre": "l"}
    return aliases.get(unit.casefold(), unit.casefold())


def _measurement(value: str, unit: str) -> tuple[float | tuple[float, float] | None, str]:
    value = value.replace(",", ".").replace("–", "-").strip()
    if "-" in value:
        pieces = [p.strip() for p in value.split("-", 1)]
        return (_float(pieces[0]), _float(pieces[1])), _unit(unit)
    return _float(value), _unit(unit)


_UNIT_SCALE = {
    "ug": ("mass", 1e-6),
    "mg": ("mass", 1e-3),
    "g": ("mass", 1.0),
    "kg": ("mass", 1000.0),
    "ml": ("volume", 1.0),
    "l": ("volume", 1000.0),
    "mmhg": ("pressure", 1.0),
    "cmh2o": ("pressure", 0.735559),
    "%": ("percent", 1.0),
    "bpm": ("rate", 1.0),
    "c": ("temperature", 1.0),
    "mm": ("length", 1.0),
    "cm": ("length", 10.0),
    "m": ("length", 1000.0),
    "hz": ("frequency", 1.0),
    "khz": ("frequency", 1000.0),
    "m/s": ("velocity", 1.0),
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    "day": ("time", 86400.0),
    "days": ("time", 86400.0),
    "week": ("time", 604800.0),
    "weeks": ("time", 604800.0),
    "month": ("time", 2592000.0),
    "months": ("time", 2592000.0),
    "year": ("time", 31536000.0),
    "years": ("time", 31536000.0),
}


def normalize_numeric_measurements(text: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for match in _MEAS.finditer(text or ""):
        raw, unit = match.group("value"), _unit(match.group("unit"))
        value, unit = _measurement(raw, unit)
        result.append({"raw": f"{raw} {unit}", "value": value, "unit": unit})
    return result


def convert_measurement(value: float, unit: str, target_unit: str) -> float | None:
    unit, target_unit = _unit(unit), _unit(target_unit)
    src, dst = _UNIT_SCALE.get(unit), _UNIT_SCALE.get(target_unit)
    if not src or not dst or src[0] != dst[0]:
        return None
    return value * src[1] / dst[1]


def measurements_compatible(a: dict[str, Any], b: dict[str, Any], tolerance: float = 1e-6) -> bool:
    av, bv = a.get("value"), b.get("value")
    au, bu = a.get("unit", ""), b.get("unit", "")
    if isinstance(av, tuple) or isinstance(bv, tuple):
        return a.get("raw") == b.get("raw")
    if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
        return False
    converted = convert_measurement(float(av), au, bu)
    if converted is None:
        return au == bu and math.isclose(float(av), float(bv), rel_tol=0.0, abs_tol=tolerance)
    return math.isclose(converted, float(bv), rel_tol=0.0, abs_tol=tolerance)


def parse_markdown_table(text: str, table_index: int = 1) -> TableEvidence | None:
    lines = [line.strip() for line in (text or "").splitlines()]
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("|") and line.count("|") >= 2:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r"[:\- ]+", cell or "-") for cell in cells):
                continue
            rows.append(cells)
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    padded = [tuple((row + [""] * width)[:width]) for row in rows]
    headers = padded[0]
    body = tuple(padded[1:])
    parsed_cells: list[TableCell] = []
    for r, row in enumerate(body, start=1):
        for c, value in enumerate(row):
            match = _MEAS.search(value)
            num = _float(match.group("value")) if match and "-" not in match.group("value") else None
            unit = _unit(match.group("unit")) if match else ""
            parsed_cells.append(TableCell(r, c, value, num, unit))
    return TableEvidence(f"table-{table_index}", "", headers, body, tuple(parsed_cells))


def parse_table_evidence(text: str) -> tuple[TableEvidence, ...]:
    tables: list[TableEvidence] = []
    parsed = parse_markdown_table(text)
    if parsed:
        tables.append(parsed)
    # Lightweight pipe/tab table recovery for extracted PDFs.
    if not tables:
        groups: list[list[str]] = []
        current: list[str] = []
        for line in (text or "").splitlines():
            if "\t" in line or line.count("|") >= 2:
                current.append(line)
            elif current:
                groups.append(current); current = []
        if current:
            groups.append(current)
        for index, group in enumerate(groups, start=1):
            pseudo = "\n".join("|" + "|".join(re.split(r"\t+|\s{2,}", ln.strip())) + "|" for ln in group)
            parsed = parse_markdown_table(pseudo, index)
            if parsed:
                tables.append(parsed)
    return tuple(tables)


def parse_figure_evidence(text: str) -> tuple[FigureEvidence, ...]:
    results: list[FigureEvidence] = []
    lines = (text or "").splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"\b(fig(?:ure)?\.?\s*\d+[A-Za-z]?)\s*[:.-]?\s*(.*)", line, re.I)
        if not match:
            continue
        figure_id = match.group(1).replace(".", "").strip()
        caption = match.group(2).strip()
        window = " ".join(lines[max(0, index - 1):min(len(lines), index + 3)])
        refs = tuple(sorted(set(re.findall(r"\b(?:fig(?:ure)?\.?\s*\d+[A-Za-z]?)\b", window, re.I))))
        results.append(FigureEvidence(figure_id, caption, refs, window))
    return tuple(results)


def extract_evidence_structures(text: str) -> dict[str, Any]:
    tables = parse_table_evidence(text)
    figures = parse_figure_evidence(text)
    return {
        "tables": tables,
        "figures": figures,
        "table_ids": [t.table_id for t in tables],
        "figure_ids": [f.figure_id for f in figures],
    }


def document_router(question: str, hits: Sequence[Any]) -> tuple[RoutedDocument, ...]:
    q = {token.casefold() for token in _WORD.findall(question or "") if len(token) >= 3}
    scores: dict[str, tuple[float, list[str]]] = {}
    for hit in hits:
        meta = getattr(hit, "metadata", {}) or {}
        doc = str(meta.get("document_id") or getattr(hit, "doc_id", "unknown"))
        text = str(getattr(hit, "text", ""))
        tokens = {token.casefold() for token in _WORD.findall(text) if len(token) >= 3}
        overlap = len(q & tokens) / max(len(q), 1)
        page_quality = float(meta.get("page_quality", 1.0) or 1.0)
        prior = float(meta.get("document_quality", 1.0) or 1.0)
        score = 0.65 * overlap + 0.20 * page_quality + 0.15 * prior
        reasons = ["query-overlap"] if overlap else ["metadata-quality"]
        previous = scores.get(doc)
        if previous is None or score > previous[0]:
            scores[doc] = (score, reasons)
    return tuple(sorted((RoutedDocument(doc, round(score, 4), tuple(reasons)) for doc, (score, reasons) in scores.items()), key=lambda x: x.score, reverse=True))


def _hit_key(hit: Any) -> str:
    meta = getattr(hit, "metadata", {}) or {}
    return str(meta.get("chunk_id") or f"{getattr(hit, 'doc_id', '')}:{getattr(hit, 'text', '')[:60]}")


def build_parent_child_context(hits: Sequence[Any], all_hits: Sequence[Any], max_per_parent: int = 2) -> list[Any]:
    by_parent: dict[str, list[Any]] = {}
    for hit in all_hits:
        meta = getattr(hit, "metadata", {}) or {}
        parent = str(meta.get("parent_id") or meta.get("section_id") or meta.get("document_id") or getattr(hit, "doc_id", ""))
        by_parent.setdefault(parent, []).append(hit)
    result: list[Any] = list(hits)
    seen = {_hit_key(h) for h in result}
    for hit in hits:
        meta = getattr(hit, "metadata", {}) or {}
        parent = str(meta.get("parent_id") or meta.get("section_id") or meta.get("document_id") or getattr(hit, "doc_id", ""))
        siblings = sorted(by_parent.get(parent, []), key=lambda h: getattr(h, "score", 0.0), reverse=True)
        for sibling in siblings[:max_per_parent]:
            key = _hit_key(sibling)
            if key not in seen:
                seen.add(key); result.append(sibling)
    return result


def expand_neighbors(hits: Sequence[Any], all_hits: Sequence[Any], radius: int = 1) -> list[Any]:
    result = list(hits)
    seen = {_hit_key(h) for h in result}
    by_doc_page: dict[str, list[Any]] = {}
    for hit in all_hits:
        meta = getattr(hit, "metadata", {}) or {}
        doc = str(meta.get("document_id") or getattr(hit, "doc_id", ""))
        pages = meta.get("page_numbers") or []
        page = int(pages[0]) if pages and str(pages[0]).isdigit() else None
        if page is not None:
            by_doc_page.setdefault(doc, []).append(hit)
    for hit in hits:
        meta = getattr(hit, "metadata", {}) or {}
        doc = str(meta.get("document_id") or getattr(hit, "doc_id", ""))
        pages = meta.get("page_numbers") or []
        if not pages or not str(pages[0]).isdigit():
            continue
        page = int(pages[0])
        for candidate in by_doc_page.get(doc, []):
            cmeta = getattr(candidate, "metadata", {}) or {}
            cpages = cmeta.get("page_numbers") or []
            if cpages and str(cpages[0]).isdigit() and abs(int(cpages[0]) - page) <= radius:
                key = _hit_key(candidate)
                if key not in seen:
                    seen.add(key); result.append(candidate)
    return result


def multi_hop_expand(question: str, hits: Sequence[Any], all_hits: Sequence[Any], max_hops: int = 2) -> list[HopEvidence]:
    current = list(hits)
    seen = {_hit_key(h) for h in current}
    result = [HopEvidence(1, h, "direct retrieval") for h in current]
    for hop in range(2, max_hops + 1):
        expanded = expand_neighbors(build_parent_child_context(current, all_hits), all_hits, radius=1)
        new_items = []
        for h in expanded:
            key = _hit_key(h)
            if key not in seen:
                seen.add(key)
                new_items.append(h)
                result.append(HopEvidence(hop, h, "linked parent/neighbor evidence"))
        current = new_items
        if not current:
            break
    return result


def sentence_compress(text: str, query: str, max_chars: int) -> tuple[str, dict[str, Any]]:
    sentences = [s.strip() for s in _SENT.split(text or "") if s.strip()]
    q = {token.casefold() for token in _WORD.findall(query or "") if len(token) >= 3}
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        tokens = {token.casefold() for token in _WORD.findall(sentence) if len(token) >= 3}
        overlap = len(q & tokens) / max(len(q), 1)
        numeric_bonus = 0.15 if _MEAS.search(sentence) else 0.0
        heading_bonus = 0.10 if re.match(r"^\d+(?:\.\d+)*\s+[A-ZÀ-Ý]", sentence) else 0.0
        score = overlap + numeric_bonus + heading_bonus
        scored.append((score, sentence))
    selected: list[str] = []
    used = 0
    for score, sentence in sorted(scored, key=lambda x: x[0], reverse=True):
        if used + len(sentence) + 1 > max_chars:
            continue
        # Deduplicate near-identical sentence fragments.
        if any(sentence.casefold() in other.casefold() or other.casefold() in sentence.casefold() for other in selected):
            continue
        selected.append(sentence); used += len(sentence) + 1
    if not selected and text:
        selected = [text[:max_chars]]
    return " ".join(reversed(selected)), {"original_sentences": len(sentences), "selected_sentences": len(selected), "compressed": len(selected) < len(sentences)}


def contradiction_groups(claims: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    groups: dict[str, dict[str, Any]] = {}
    for claim in claims:
        text = str(getattr(claim, "claim", claim))
        normalized = re.sub(r"\[[A-Z]\d*\]", "", text.casefold())
        normalized = re.sub(r"\b\d+(?:[\.,]\d+)?\s*[a-zµ°%]+\b", "<measurement>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        groups.setdefault(normalized[:220], {"subject": normalized[:220], "claims": []})["claims"].append(claim)
    out = []
    neg_words = re.compile(r"\b(no|not|without|never|contraindicated|avoid|cannot|does not|doesn't|none|not associated|unrelated)\b", re.I)
    for group in groups.values():
        claims_group = group["claims"]
        polarities = {bool(neg_words.search(str(getattr(c, "claim", c)))) for c in claims_group}
        if len(polarities) > 1:
            group["contradicted"] = True
        elif len(claims_group) > 1:
            # Numeric conflict under a shared normalized subject.
            values = {tuple(sorted((m["raw"] for m in normalize_numeric_measurements(str(getattr(c, "claim", c)))))) for c in claims_group}
            group["contradicted"] = len(values) > 1 and any(values)
        else:
            group["contradicted"] = False
        out.append(group)
    return tuple(out)


def resolve_conflicts(claims: Sequence[Any], evidence_quality: dict[str, float] | None = None) -> dict[str, Any]:
    evidence_quality = evidence_quality or {}
    groups = contradiction_groups(claims)
    unresolved: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    for group in groups:
        if not group["contradicted"]:
            continue
        candidates = group["claims"]
        ranked = sorted(candidates, key=lambda c: (float(getattr(c, "support", 0.0)), max((evidence_quality.get(s, 0.0) for s in getattr(c, "sources", ())), default=0.0)), reverse=True)
        if len(ranked) >= 2:
            top, second = ranked[0], ranked[1]
            top_score = float(getattr(top, "support", 0.0))
            second_score = float(getattr(second, "support", 0.0))
            quality = max((evidence_quality.get(s, 0.0) for s in getattr(top, "sources", ())), default=0.0)
            if top_score - second_score >= 0.15 and quality >= 0.65:
                resolutions.append({"subject": group["subject"], "selected": getattr(top, "claim", str(top)), "reason": "stronger evidence margin and source quality"})
            else:
                unresolved.append(group)
    return {"resolved": resolutions, "unresolved": unresolved, "has_unresolved": bool(unresolved)}


def grounded_evidence_gate(claims: Sequence[Any], *, min_ratio: float = 0.75, allow_partial: bool = True) -> EvidenceDecision:
    if not claims:
        return EvidenceDecision(False, "no claims", 0.0)
    safe_status = {"SUPPORTED", "PARTIAL"} if allow_partial else {"SUPPORTED"}
    supported = [c for c in claims if getattr(c, "status", "") in safe_status and not getattr(c, "contradiction", False)]
    blocked = [c for c in claims if c not in supported]
    ratio = len(supported) / len(claims)
    contradictions = tuple(getattr(c, "claim", "") for c in blocked if getattr(c, "status", "") == "CONTRADICTED")
    blocked_claims = tuple(getattr(c, "claim", "") for c in blocked)
    return EvidenceDecision(
        allow=ratio >= min_ratio and not contradictions,
        reason="grounded" if ratio >= min_ratio and not contradictions else "insufficiently grounded",
        score=round(ratio, 4), contradictions=contradictions, blocked_claims=blocked_claims,
    )


def adversarial_probe(text: str) -> dict[str, Any]:
    patterns = {
        "prompt_injection": r"(?i)(ignore|disregard|override|system prompt|developer mode|jailbreak)",
        "role_hijack": r"(?i)^\s*(system|assistant|developer|user)\s*[:：-]",
        "fake_citation": r"(?i)\[(?:source|ref|citation)\s*[:#]?\s*\d+\]",
    }
    matches = {name: bool(re.search(pattern, text or "", re.M)) for name, pattern in patterns.items()}
    return {"flagged": any(matches.values()), "signals": matches}


def abstention_ladder(*, query_ok: bool, retrieval_ok: bool, evidence_score: float, grounding_ok: bool, contradiction: bool, generation_ok: bool) -> str:
    if not query_ok:
        return "QUERY_CLARIFICATION"
    if not retrieval_ok:
        return "RETRIEVAL_ABSTAIN"
    if evidence_score < 0.30:
        return "EVIDENCE_ABSTAIN"
    if contradiction:
        return "CONTRADICTION_ABSTAIN"
    if not generation_ok:
        return "GENERATION_ABSTAIN"
    if not grounding_ok:
        return "GROUNDING_ABSTAIN"
    return "ANSWER"


def build_query_trace(*, question: str, route: Sequence[RoutedDocument], hits: Sequence[Any], claims: Sequence[Any], timings: dict[str, float], decisions: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_length": len(question or ""),
        "documents": [asdict(r) for r in route],
        "retrieved_count": len(hits),
        "claims": [getattr(c, "to_dict", lambda: str(c))() for c in claims],
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "decisions": decisions,
    }


def full_reasoning_pass(question: str, hits: Sequence[Any], all_hits: Sequence[Any], max_context_chars: int = 10000) -> dict[str, Any]:
    route = document_router(question, hits)
    structures = extract_evidence_structures("\n\n".join(str(getattr(h, "text", "")) for h in hits))
    parent_child = build_parent_child_context(hits, all_hits)
    neighbors = expand_neighbors(parent_child, all_hits)
    hops = multi_hop_expand(question, neighbors, all_hits, max_hops=2)
    compressed_texts: list[str] = []
    compression: list[dict[str, Any]] = []
    for hit in hops:
        compressed, stats = sentence_compress(str(getattr(hit.hit, "text", "")), question, max(600, max_context_chars // max(len(hops), 1)))
        compressed_texts.append(compressed)
        compression.append({"hit": _hit_key(hit.hit), **stats})
    context = "\n\n".join(compressed_texts)[:max_context_chars]
    return {
        "route": route,
        "structures": structures,
        "parent_child_hits": parent_child,
        "neighbor_hits": neighbors,
        "hop_evidence": hops,
        "compressed_context": context,
        "compression": compression,
        "numeric_evidence": normalize_numeric_measurements(context),
    }
