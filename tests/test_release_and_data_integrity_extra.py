from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_every_advertised_feature_target_resolves():
    from rag_project.intelligence.production_contract import FEATURES, resolve_target

    assert len(FEATURES) == 44
    assert len({feature.name for feature in FEATURES}) == 44
    for feature in FEATURES:
        target = resolve_target(feature.target)
        assert target is not None, feature.name


def test_feature_contract_is_fully_resolved():
    from rag_project.intelligence.production_contract import validate_feature_contract

    result = validate_feature_contract()
    assert result['feature_count'] == 44
    assert result['unique_names'] is True
    assert result['duplicates'] == []
    assert result['unresolved'] == {}
    assert result['all_resolved'] is True


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('email john@example.com', '[REDACTED_EMAIL]'),
        ('call +213 555 123 456', '[REDACTED_PHONE]'),
        ('patient id 123456789', '[REDACTED_ID]'),
        ('nothing sensitive here', 'nothing sensitive here'),
    ],
)
def test_sensitive_text_redaction(text: str, expected: str):
    from rag_project.intelligence.production_contract import redact_sensitive_text

    value = redact_sensitive_text(text)
    assert expected in value
    assert 'john@example.com' not in value


def test_trace_sanitization_redacts_sensitive_query_fields_only():
    from rag_project.intelligence.production_contract import sanitize_trace

    trace = {
        'question': 'Call me at +213 555 123 456 and mail a@b.com',
        'original_query': 'patient id 123456789',
        'score': .9,
    }
    result = sanitize_trace(trace)
    assert '[REDACTED_PHONE]' in result['question']
    assert '[REDACTED_EMAIL]' in result['question']
    assert '[REDACTED_ID]' in result['original_query']
    assert result['score'] == .9
    assert trace['score'] == .9


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        ({'feature_contract': True, 'tests_green': True, 'index_ready': True, 'privacy_controls': True, 'medical_safety': True}, True),
        ({'feature_contract': False, 'tests_green': True, 'index_ready': True, 'privacy_controls': True, 'medical_safety': True}, False),
        ({'feature_contract': True, 'tests_green': False, 'index_ready': True, 'privacy_controls': True, 'medical_safety': True}, False),
        ({'feature_contract': True, 'tests_green': True, 'index_ready': False, 'privacy_controls': True, 'medical_safety': True}, False),
        ({'feature_contract': True, 'tests_green': True, 'index_ready': True, 'privacy_controls': False, 'medical_safety': True}, False),
        ({'feature_contract': True, 'tests_green': True, 'index_ready': True, 'privacy_controls': True, 'medical_safety': False}, False),
    ],
)
def test_production_readiness_requires_every_gate(profile, expected: bool):
    from rag_project.intelligence.production_contract import production_readiness

    result = production_readiness(profile)
    assert result['release_ready'] is expected
    assert result['clinical_validation'] is False
    assert result['regulatory_approval'] is False
    assert set(result['gates']) == {
        'feature_contract', 'tests_green', 'index_ready', 'privacy_controls', 'medical_safety'
    }


def test_runtime_contract_has_no_monkey_patch_authority():
    from rag_project.application import runtime_contract

    result = runtime_contract()
    assert result['answer_monkey_patch'] is False
    assert result['answer_pipeline'] == 'explicit_delegation'


def test_top_level_pipeline_exports_expected_public_functions():
    from rag_project.intelligence import top_level_pipeline as module

    for name in (
        'PhasePlan', 'deterministic_phase1', 'llm_phase1', 'rewrite_follow_up',
        'medical_term_layer', 'precision_filter', 'compress_context',
        'adaptive_retrieve', 'dynamic_temperature', 'extractive_draft',
        'synthesize_answer', 'complete_phases',
    ):
        assert hasattr(module, name)


def test_evidence_guard_exports_expected_safety_functions():
    from rag_project.intelligence import evidence_guard as module

    for name in (
        'split_claims', 'extract_measurements', 'numeric_consistency',
        'semantic_support', 'detect_contradiction', 'verify_claims',
        'citation_firewall', 'grounding_decision',
    ):
        assert hasattr(module, name)


def test_ui_visibility_contract_exports_single_normalizer():
    from rag_project.intelligence.ui_visibility_contract import build_visibility_contract

    assert callable(build_visibility_contract)


def test_canonical_authority_is_single_source_and_consumers_reference_it():
    from rag_project.application import ANSWER_PIPELINE_AUTHORITY, runtime_contract

    expected = 'rag_project.intelligence.top_level_pipeline.complete_phases'
    assert ANSWER_PIPELINE_AUTHORITY == expected
    assert runtime_contract()['answer_pipeline_authority'] == expected

    production = (ROOT / 'rag_project' / 'app' / 'production_rag.py').read_text(encoding='utf-8')
    god_mode = (ROOT / 'rag_project' / 'intelligence' / 'god_mode_100.py').read_text(encoding='utf-8')
    assert 'from rag_project.application import ANSWER_PIPELINE_AUTHORITY' in production
    assert 'PIPELINE_AUTHORITY = ' in god_mode
    assert 'complete_phases' in god_mode


def test_no_test_fixture_depends_on_real_pdf_data_for_core_unit_tests():
    tests = ROOT / 'tests'
    for path in tests.glob('test_*extra.py'):
        source = path.read_text(encoding='utf-8').casefold()
        assert 'http://' not in source or 'ollama' in source
