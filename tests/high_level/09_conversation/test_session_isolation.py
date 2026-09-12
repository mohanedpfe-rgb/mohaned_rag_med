from __future__ import annotations

import pytest

from rag_project.application import create_rag_system


@pytest.mark.high_level
def test_conversation__separate_production_system_instances_have_isolated_histories(settings_i5):
    first = create_rag_system(settings_i5)
    second = create_rag_system(settings_i5)

    assert first.conversation_memory is not second.conversation_memory
    first.conversation_memory.history.append(("private session marker", {"status": "SUCCESS"}))

    assert list(second.conversation_memory.history) == []
    assert "private session marker" not in str(second.conversation_memory.prompt_context()).casefold()
