from __future__ import annotations

import re
from typing import Any


class QueryRewriter:
    @staticmethod
    def rewrite(question: str, history: list[tuple[str, str]] | None = None, llm: Any = None) -> str:
        cleaned = question.strip()
        if not cleaned:
            return cleaned
            
        if llm:
            history_context = ""
            if history:
                history_context = "Conversation history:\n" + "\n".join(
                    f"User: {q}\nSystem: {a}" for q, a in history[-3:]
                ) + "\n\n"
            
            prompt = (
                "You are an expert search query rewriter for a RAG system.\n"
                f"{history_context}"
                "Your task is to rewrite the user's latest question into a clearer, standalone search query "
                "that includes any missing context or resolved pronouns from the history. "
                "Output ONLY the rewritten query, nothing else.\n\n"
                f"User question: {cleaned}"
            )
            try:
                # Use a fast low-temperature generation
                rewritten = llm.generate(prompt, temperature=0.0).strip()
                # Basic cleanup if the LLM adds quotes or prefix
                rewritten = re.sub(r'^(Query|Rewritten|Search):?\s*', '', rewritten, flags=re.IGNORECASE)
                rewritten = rewritten.strip('"\'')
                if rewritten:
                    return rewritten
            except Exception:
                pass # fallback below
                
        # Basic fallback
        if history and (
            len(cleaned.split()) <= 8
            or re.search(r"\b(it|this|that|they|them|those|these|what about)\b", cleaned, re.I)
        ):
            previous_question = history[-1][0]
            return f"{previous_question} Follow-up question: {cleaned}"
        return cleaned


class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: list[tuple[str, str]] = []

    def add(self, question: str, answer: str) -> None:
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def prompt_context(self) -> str:
        if not self.history:
            return ""
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history)
