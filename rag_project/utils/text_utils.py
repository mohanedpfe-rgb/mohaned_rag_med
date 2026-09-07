from __future__ import annotations

import re
import unicodedata
from typing import List

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "can", "do", "for", "how", "i",
    "in", "is", "it", "of", "on", "or", "the", "to", "what", "when", "where",
    "which", "who", "why", "with", "about", "does", "this", "that",
    "un", "une", "des", "du", "de", "et", "est", "les", "le", "la", "dans",
    "pour", "comment", "quel", "quelle", "quels", "quelles", "sur",
}

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_NORMALIZATION = str.maketrans({
    "\u0622": "\u0627",
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0671": "\u0627",
    "\u0649": "\u064a",
    "\u06cc": "\u064a",
    "\u06c0": "\u0647",
    "\u06c1": "\u0647",
    "\u06d5": "\u0647",
})


def normalize_arabic(value: str) -> str:
    """Normalize common Arabic presentation variants without transliteration."""
    return ARABIC_DIACRITICS.sub("", value or "").translate(ARABIC_NORMALIZATION)


def detect_language(value: str) -> str:
    """Return a conservative script-based language hint for routing and metadata."""
    text = value or ""
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    latin = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    if arabic > latin and arabic:
        return "ar"
    if latin:
        lowered = text.casefold()
        if re.search(r"\b(le|la|les|des|une|dans|pour|avec|est)\b", lowered):
            return "fr"
        return "en"
    return "unknown"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "").replace("\xa0", " ")
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def tokenize(value: str) -> List[str]:
    normalized = normalize_arabic(unicodedata.normalize("NFKC", value or "")).casefold()
    return re.findall(r"[\w]+(?:[/.-][\w]+)*", normalized, flags=re.UNICODE)


def meaningful_tokens(value: str) -> List[str]:
    return [token for token in tokenize(value) if token not in STOPWORDS and len(token) > 1]


def keyword_overlap_score(query: str, document: str) -> float:
    q_tokens = set(meaningful_tokens(query))
    d_tokens = set(meaningful_tokens(document))
    if not q_tokens:
        return 0.0
    overlap = q_tokens & d_tokens
    return len(overlap) / max(len(q_tokens), 1)


def keyword_proximity_score(query: str, document: str) -> float:
    terms = set(meaningful_tokens(query))
    tokens = meaningful_tokens(document)
    positions = [index for index, token in enumerate(tokens) if token in terms]
    if not terms or not positions:
        return 0.0
    coverage = len(set(tokens[index] for index in positions)) / len(terms)
    span = max(positions) - min(positions) + 1
    return coverage / (1.0 + min(span, 100) / 20.0)


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p and p.strip()]


def extract_page_number(page_text: str) -> int | None:
    if not page_text:
        return None
    text = page_text.strip()
    for pattern in (
        r"(?im)^\s*(?:page|p\.)\s*[:#-]?\s*(\d{1,4})\s*$",
        r"(?im)^\s*(?:page|p\.)\s*[:#-]?\s*(\d{1,4})\b",
        r"(?im)^\s*(\d{1,4})\s*(?:-|–|—)?\s*(?:of|/|\\)\s*\d{1,4}",
    ):
        match = re.search(pattern, text[:500])
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None
