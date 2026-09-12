from __future__ import annotations

import pytest

from rag_project.generation.llm_client import MAX_PROMPT_WORDS, OllamaLLMClient


@pytest.mark.high_level
def test_generation__ollama_payload_truncates_user_context_to_shared_word_budget():
    client = object.__new__(OllamaLLMClient)
    prompt = " ".join(f"token{i}" for i in range(MAX_PROMPT_WORDS + 100))

    bounded = client._bound_prompt(prompt)
    assert len(bounded.split()) <= MAX_PROMPT_WORDS + 6
    assert "[TRUNCATED_CONTEXT:" in bounded


@pytest.mark.high_level
def test_generation__bounded_payload_preserves_short_prompt_exactly():
    client = object.__new__(OllamaLLMClient)
    prompt = "Question: What is diabetes? Evidence: diabetes is a chronic metabolic disorder."

    assert client._bound_prompt(prompt) == prompt
