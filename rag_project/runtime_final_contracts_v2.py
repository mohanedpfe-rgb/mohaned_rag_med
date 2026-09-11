from __future__ import annotations

import re

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.retrieval.query_rewriter import QueryRewriter

    def rewrite(question: str, history=None, llm=None) -> str:
        cleaned = re.sub(r"\s+", " ", str(question or "")).strip()
        if not cleaned:
            return ""
        explicit_followup = bool(
            re.search(
                r"\b(what about|how about|and the|and this|and that|this|that|it|they|them)\b",
                cleaned,
                re.I,
            )
            or re.match(r"^(et|and|also|then|و|ثم)\b", cleaned, re.I | re.UNICODE)
        )
        if not explicit_followup or not history:
            return cleaned
        recent = list(history[-3:])
        anchor_q = next((str(q).strip() for q, _ in reversed(recent) if str(q or "").strip()), "")
        anchor_a = next((str(a).strip() for _, a in reversed(recent) if str(a or "").strip()), "")
        terms: list[str] = []
        seen: set[str] = set()
        for term in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9-]{4,}", anchor_a):
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
            if len(terms) >= 6:
                break
        context = " ".join(terms)
        rewritten = " ".join(part for part in (anchor_q, context, cleaned) if part).strip()[:3500]
        return f"Follow-up: {rewritten}" if rewritten else cleaned

    QueryRewriter.rewrite = staticmethod(rewrite)
    _INSTALLED = True

__all__ = ["install"]
