from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_critical_runtime_modules_importable() -> None:
    modules = [
        "rag_project.application",
        "rag_project.app.production_rag",
        "rag_project.intelligence.production_contract",
        "rag_project.intelligence.top_level_pipeline",
        "rag_project.intelligence.god_mode_100",
        "rag_project.intelligence.evidence_guard",
        "rag_project.intelligence.evidence_entailment",
        "rag_project.intelligence.entity_coverage",
        "rag_project.intelligence.confidence_calibration",
        "rag_project.app.intelligence_panel",
    ]
    for name in modules:
        __import__(name)


def test_canonical_pipeline_authority_is_explicit(repo_root: Path) -> None:
    source = (repo_root / "rag_project" / "application.py").read_text(encoding="utf-8")
    assert "ANSWER_PIPELINE_AUTHORITY" in source
    assert "rag_project.intelligence.top_level_pipeline.complete_phases" in source


def test_runtime_contract_exposes_production_answer_path(repo_root: Path) -> None:
    source = (repo_root / "rag_project" / "application.py").read_text(encoding="utf-8")
    assert "PRODUCTION_RUNTIME_CONTRACT" in source
    assert "enhanced_god_answer" in source


def test_settings_parse_boolean_values() -> None:
    from rag_project.configuration.settings import _as_bool

    assert _as_bool(True) is True
    assert _as_bool(False) is False
    assert _as_bool("true") is True
    assert _as_bool("YES") is True
    assert _as_bool("0") is False
    assert _as_bool("off") is False


def test_settings_parse_numeric_fallbacks() -> None:
    from rag_project.configuration.settings import _as_float, _as_int

    assert _as_int("12", 3) == 12
    assert _as_int("bad", 3) == 3
    assert _as_float("1.5", 3.0) == 1.5
    assert _as_float("bad", 3.0) == 3.0


def test_settings_normalize_core_ranges() -> None:
    from rag_project.configuration.settings import Settings

    settings = Settings(chunk_size=1, chunk_overlap=999999, max_workers=0, max_memory_target=1, ollama_concurrency=0)
    assert settings.chunk_size >= 200
    assert 0 <= settings.chunk_overlap < settings.chunk_size
    assert settings.max_workers >= 1
    assert settings.max_memory_target >= 256
    assert settings.ollama_concurrency >= 1
    assert settings.ingestion_lease_seconds >= 30
    assert 2 <= settings.retrieval_candidate_multiplier <= 12
    assert 0 <= settings.evidence_min_confidence <= 1
    assert 1 <= settings.max_query_variants <= 12


def test_text_pipeline_normalizes_unicode_whitespace_and_arabic() -> None:
    from rag_project.utils.text_utils import clean_text, normalize_arabic, normalize_whitespace

    assert normalize_whitespace("  alpha\n\n beta  ") == "alpha beta"
    assert clean_text("a\u00a0b\r\n\r\n\r\nc") == "a b\n\nc"
    assert normalize_arabic("إيمان آية ى") == "ايمان اية ي"


@pytest.mark.parametrize(
    ("text", "language"),
    [("diabetes treatment", "en"), ("le traitement", "fr"), ("علاج السكري", "ar"), ("12345", "unknown"), ("", "unknown")],
)
def test_language_detection_contract(text: str, language: str) -> None:
    from rag_project.utils.text_utils import detect_language

    assert detect_language(text) == language


@pytest.mark.parametrize(
    "text",
    [
        "Page 5",
        "p. 10",
        "page 42",
        "Page 7 of 100",
        "pp. 12-13",
    ],
)
def test_page_number_extraction_is_available(text: str) -> None:
    from rag_project.utils.text_utils import extract_page_number

    assert extract_page_number(text) is not None


def test_keyword_overlap_is_bounded() -> None:
    from rag_project.utils.text_utils import keyword_overlap_score

    for a, b in [("", "x"), ("diabetes treatment", "diabetes treatment"), ("alpha", "omega")]:
        value = keyword_overlap_score(a, b)
        assert 0 <= value <= 1


def test_metadata_filter_preserves_ready_indexes() -> None:
    from rag_project.retrieval.metadata_filter import MetadataFilter

    filt = MetadataFilter()
    assert filt.apply([], filters={}) == []
    assert filt._valid_index_state("READY")


def test_conversation_memory_is_bounded() -> None:
    from rag_project.retrieval.query_rewriter import ConversationMemory

    memory = ConversationMemory(max_turns=2)
    memory.add("q1", "a1")
    memory.add("q2", "a2")
    memory.add("q3", "a3")
    assert len(memory.history) == 2
    assert memory.history[0][0] == "q2"


def test_query_rewriter_fallback_preserves_question() -> None:
    from rag_project.retrieval.query_rewriter import QueryRewriter

    rewriter = QueryRewriter()
    assert rewriter.rewrite("What is diabetes?", history=[]) .strip()


def test_claim_splitting_removes_markers_but_keeps_claims() -> None:
    from rag_project.intelligence.evidence_guard import split_claims

    claims = split_claims("- Diabetes is chronic. [S1]\n- Treatment varies.")
    assert len(claims) == 2
    assert all("[S" not in claim for claim in claims)


def test_final_answer_verification_fails_closed_on_unsupported_claim() -> None:
    from rag_project.intelligence.final_answer_contract import verify_final_answer

    result = verify_final_answer(
        "The moon is blue.",
        evidence=["Diabetes is chronic."],
        evidence_ids=["S1"],
    )
    assert result["allow"] is False or result.get("verified") is False


def test_confidence_calibration_is_bounded_and_monotonic_in_basic_case() -> None:
    from rag_project.intelligence.confidence_calibration import calibrate_confidence

    low = calibrate_confidence(0.2, 0.2, 0.2, 0.2)
    high = calibrate_confidence(0.9, 0.9, 0.9, 0.9)
    assert 0 <= low.score <= 1
    assert 0 <= high.score <= 1
    assert high.score >= low.score


def test_critical_source_files_exist_and_are_nonempty(repo_root: Path) -> None:
    paths = [
        repo_root / "app.py",
        repo_root / "rag_project" / "application.py",
        repo_root / "rag_project" / "app" / "production_rag.py",
        repo_root / "rag_project" / "intelligence" / "god_mode_100.py",
        repo_root / "rag_project" / "intelligence" / "top_level_pipeline.py",
    ]
    assert all(path.is_file() and path.read_text(encoding="utf-8").strip() for path in paths)


def test_authority_constant_is_not_duplicated_with_conflicting_values(repo_root: Path) -> None:
    sources = []
    for name in ("application.py", "app/production_rag.py", "intelligence/god_mode_100.py"):
        sources.append((repo_root / "rag_project" / name).read_text(encoding="utf-8"))
    values = set()
    for source in sources:
        match = re.search(r"(?:ANSWER_PIPELINE_AUTHORITY|PIPELINE_AUTHORITY)\s*=\s*[\"']([^\"']+)", source)
        if match:
            values.add(match.group(1))
    assert values == {"rag_project.intelligence.top_level_pipeline.complete_phases"}
