import re
import pytest
from tests.high_level.helpers import assert_citations_valid

@pytest.mark.high_level

def test_successful_answer__has_valid_citations(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_citations_valid(result)

@pytest.mark.high_level

def test_answer__does_not_emit_malformed_source_markers(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    answer = str(result.get("answer") or "")
    assert not re.search(r"\[(?:source|citation)\s*:\s*\]", answer, flags=re.I)
