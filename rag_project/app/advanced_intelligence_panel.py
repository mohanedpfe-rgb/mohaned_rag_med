from __future__ import annotations

from typing import Any

import streamlit as st

from rag_project.intelligence.document_intelligence import build_document_map, section_coverage


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _show_rows(title: str, rows: list[dict[str, Any]], limit: int = 8) -> None:
    if not rows:
        return
    st.markdown(f"**{title}**")
    for row in rows[:limit]:
        st.write(row)


def render_advanced_intelligence_panel(result: dict[str, Any]) -> None:
    """Expose the new document-aware retrieval contract without changing the answer UI."""
    if not isinstance(result, dict):
        return
    trace = result.get("query_trace") or {}
    routing = trace.get("routing") or {}
    retrieval = trace.get("retrieval") or {}
    quality = result.get("retrieval_quality") or {}
    coverage = retrieval.get("coverage") or {}
    route = routing.get("kind", "—")

    with st.expander("Advanced retrieval intelligence", expanded=False):
        st.write(
            f"**Route:** `{route}` · scope=`{routing.get('scope', '—')}` · "
            f"confidence={_pct(routing.get('confidence', 0))}"
        )
        flags = []
        for key in ("summary", "comparison", "multi_hop", "exact_lookup", "needs_table", "needs_numeric", "needs_figure"):
            if routing.get(key):
                flags.append(key)
        st.write(f"**Specialized strategy:** {', '.join(flags) if flags else 'standard factual retrieval'}")
        st.write(
            f"**Candidates:** {retrieval.get('candidates', quality.get('candidate_count', '—'))} · "
            f"final hits={retrieval.get('final_hits', quality.get('final_hits', '—'))} · "
            f"self-corrections={retrieval.get('self_corrections', quality.get('self_corrections', 0))}"
        )
        _show_rows("Retrieval strategy", [{"step": item} for item in (retrieval.get("strategy") or [])], 16)
        _show_rows("Query variants", [{"query": item} for item in (retrieval.get("queries") or [])], 16)

        st.write(
            f"**Evidence coverage:** {_pct(coverage.get('overall', quality.get('evidence_coverage', 0)))} · "
            f"entity coverage={_pct(coverage.get('entity_coverage', quality.get('entity_coverage', 0)))}"
        )
        slots = coverage.get("slots") or {}
        if slots:
            for slot, value in slots.items():
                st.write(f"`{slot}` · {_pct(value)}")

        hits = list(result.get("hits") or ())
        structure = build_document_map(hits)
        sections = section_coverage(hits)
        st.write(
            f"**Document structure in evidence:** {structure.get('document_count', 0)} document(s), "
            f"{structure.get('node_count', 0)} structural node(s), "
            f"{sections.get('distinct_sections', 0)} distinct section(s), "
            f"{sections.get('distinct_pages', 0)} page(s)"
        )
        if structure.get("documents"):
            _show_rows("Documents represented", structure["documents"], 8)

        contradiction = result.get("contradiction_report") or retrieval.get("contradiction") or {}
        if contradiction.get("has_contradiction"):
            st.warning(f"Source disagreement detected: {contradiction.get('count', len(contradiction.get('conflicts') or []))} conflict(s).")
            _show_rows("Detected conflicts", contradiction.get("conflicts") or [], 8)
        else:
            st.success("No obvious numeric/source disagreement was detected among selected evidence.")

        plan = result.get("answer_plan") or {}
        if plan:
            st.markdown("**Answer plan**")
            st.json(plan)

        st.caption(
            "This panel is diagnostic only. The canonical answer remains the certified "
            "document-aware evidence-first result."
        )


__all__ = ["render_advanced_intelligence_panel"]
