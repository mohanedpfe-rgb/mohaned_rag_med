from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PageQuality:
    page_number: int
    char_count: int
    word_count: int
    image_count: int
    text_density: float
    alpha_ratio: float
    weird_ratio: float
    repeated_ratio: float
    quality: float
    route: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentProfile:
    page_count: int
    text_pages: int
    scan_pages: int
    table_pages: int
    image_pages: int
    mixed_pages: int
    language: str
    risk: str
    routes: dict[str, int]


_NUMBER_RE = re.compile(
    r"(?<!\w)([-+]?\d+(?:[\.,]\d+)?(?:\s*[–-]\s*\d+(?:[\.,]\d+)?)?)\s*"
    r"(mg|g|kg|mcg|µg|ug|ml|l|mmhg|cmh2o|%|bpm|°c|c|mm|cm|m)(?:\b|$)",
    re.I,
)
_ENTITY_PATTERNS = {
    "drug": re.compile(r"\b(?:acetaminophen|paracetamol|ibuprofen|metformin|aspirin|amoxicillin|warfarin)\b", re.I),
    "dose": re.compile(r"\b\d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|µg|ml|mL)\b", re.I),
    "percent": re.compile(r"\b\d+(?:[\.,]\d+)?\s*%\b"),
    "date": re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b"),
    "section": re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S.+$", re.M),
}


def _safe_ratio(a: int | float, b: int | float) -> float:
    return float(a) / max(float(b), 1.0)


def normalize_numbers(text: str) -> str:
    """Canonicalize common numeric/medical units without destroying original text."""
    def repl(match: re.Match[str]) -> str:
        number = match.group(1).replace(",", ".").replace("–", "-").replace(" ", "")
        unit = match.group(2).lower().replace("µg", "ug")
        aliases = {"milligram": "mg", "milliliters": "ml", "milliliter": "ml", "litre": "l", "liter": "l"}
        unit = aliases.get(unit, unit)
        return f"{number} {unit}"
    return _NUMBER_RE.sub(repl, text or "")


def extract_entities(text: str) -> dict[str, list[str]]:
    text = text or ""
    result: dict[str, list[str]] = {}
    for name, pattern in _ENTITY_PATTERNS.items():
        values = []
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value not in values:
                values.append(value)
        result[name] = values[:50]
    return result


def enrich_text(text: str) -> dict[str, Any]:
    normalized = normalize_numbers(text or "")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    headings = [line for line in lines if _ENTITY_PATTERNS["section"].match(line)][:50]
    entities = extract_entities(normalized)
    keywords = []
    for token in re.findall(r"[\wÀ-ÿ]{3,}", normalized.casefold()):
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= 40:
            break
    return {
        "normalized_text": normalized,
        "entities": entities,
        "headings": headings,
        "keywords": keywords,
        "number_forms": sorted(set(m.group(0) for m in _NUMBER_RE.finditer(text or "")))[:100],
    }


def score_page_quality(text: str, *, page_number: int = 1, image_count: int = 0, page_area: float | None = None) -> PageQuality:
    raw = text or ""
    chars = len(raw)
    words = len(re.findall(r"\S+", raw))
    alpha = sum(ch.isalpha() for ch in raw)
    weird = sum(1 for ch in raw if ord(ch) < 9 or (0xD800 <= ord(ch) <= 0xDFFF))
    repeated = sum(1 for a, b, c in zip(raw, raw[1:], raw[2:]) if a == b == c)
    density = _safe_ratio(chars, page_area or 50000.0)
    alpha_ratio = _safe_ratio(alpha, chars)
    weird_ratio = _safe_ratio(weird, chars)
    repeated_ratio = _safe_ratio(repeated, max(chars - 2, 1))
    warnings: list[str] = []
    if chars < 30 and image_count > 0:
        warnings.append("likely_scanned")
    if alpha_ratio < 0.2 and chars > 100:
        warnings.append("low_alpha_ratio")
    if weird_ratio > 0.01:
        warnings.append("encoding_noise")
    if repeated_ratio > 0.10:
        warnings.append("repeated_garbage")
    quality = 0.0
    quality += min(chars / 1200.0, 1.0) * 0.45
    quality += min(alpha_ratio / 0.7, 1.0) * 0.25
    quality += (1.0 - min(weird_ratio * 20.0, 1.0)) * 0.15
    quality += (1.0 - min(repeated_ratio * 8.0, 1.0)) * 0.15
    quality = max(0.0, min(1.0, quality))
    if image_count and chars < 120:
        route = "ocr"
    elif quality < 0.35:
        route = "alternate_extractor"
    elif image_count and quality < 0.60:
        route = "ocr_verify"
    else:
        route = "native"
    return PageQuality(page_number, chars, words, image_count, density, alpha_ratio, weird_ratio, repeated_ratio, round(quality, 4), route, tuple(warnings))


def classify_document_pages(pages: Iterable[dict[str, Any]]) -> DocumentProfile:
    records = list(pages)
    counts = {"native": 0, "ocr": 0, "ocr_verify": 0, "alternate_extractor": 0}
    text_pages = scan_pages = table_pages = image_pages = mixed_pages = 0
    for item in records:
        text = str(item.get("text") or "")
        images = int(item.get("image_count") or 0)
        q = score_page_quality(text, page_number=int(item.get("page_number") or 1), image_count=images)
        counts[q.route] = counts.get(q.route, 0) + 1
        if q.char_count >= 100:
            text_pages += 1
        if q.route == "ocr":
            scan_pages += 1
        if images:
            image_pages += 1
        if item.get("table") or item.get("tables"):
            table_pages += 1
        if images and q.char_count >= 100:
            mixed_pages += 1
    page_count = len(records)
    scan_ratio = _safe_ratio(scan_pages, page_count)
    risk = "low" if scan_ratio < 0.15 else "medium" if scan_ratio < 0.50 else "high"
    language = "unknown"
    joined = " ".join(str(x.get("text") or "") for x in records[:8]).lower()
    if any(ch in joined for ch in "ةيىؤإأ" ):
        language = "ar"
    elif any(word in joined for word in ("bonjour", "avec", "dans", "patient")):
        language = "fr"
    elif joined.strip():
        language = "en"
    return DocumentProfile(page_count, text_pages, scan_pages, table_pages, image_pages, mixed_pages, language, risk, counts)


def build_manifest(*, file_path: str | Path, profile: DocumentProfile, pages: list[dict[str, Any]], chunks: int = 0, embeddings: int = 0, warnings: list[str] | None = None) -> dict[str, Any]:
    path = Path(file_path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_hash = digest.hexdigest()
    except OSError:
        file_hash = "unavailable"
    page_records = [asdict(score_page_quality(str(p.get("text") or ""), page_number=int(p.get("page_number") or i + 1), image_count=int(p.get("image_count") or 0))) for i, p in enumerate(pages)]
    failed_pages = [p["page_number"] for p in page_records if p["quality"] < 0.35]
    return {
        "file_name": path.name,
        "file_hash": file_hash,
        "page_count": profile.page_count,
        "extracted_pages": profile.text_pages,
        "scan_pages": profile.scan_pages,
        "failed_quality_pages": failed_pages,
        "chunks": int(chunks),
        "embeddings": int(embeddings),
        "routes": profile.routes,
        "language": profile.language,
        "risk": profile.risk,
        "warnings": list(warnings or []),
        "coverage_percent": round(100.0 * _safe_ratio(profile.text_pages, profile.page_count), 2),
    }
