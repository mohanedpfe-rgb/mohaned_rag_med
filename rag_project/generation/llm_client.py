from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import requests

from rag_project.generation.latency_budget import exhausted, remaining
from rag_project.security import validate_ollama_url

try:
    from rag_project.utils.logger import build_logger
    _logger = build_logger("llm_client")
except Exception:
    _logger = logging.getLogger("llm_client")
    if not _logger.handlers:
        _logger.addHandler(logging.StreamHandler())
        _logger.setLevel(logging.INFO)


MAX_PROMPT_WORDS = 600


class OllamaLLMClient:
    """Bounded Ollama client with request-scoped deadline and context budgets."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 180.0, max_output_tokens: int = 512, circuit_threshold: int = 2, circuit_open_seconds: float = 15.0):
        self.base_url = validate_ollama_url(base_url)
        self.model = str(model).strip()
        if not self.model or len(self.model) > 200 or any(ord(ch) < 32 for ch in self.model):
            raise ValueError("Invalid Ollama model identifier.")
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.request_timeout_seconds = min(self.timeout_seconds, 30.0)
        self.max_output_tokens = max(32, min(int(max_output_tokens), 4096))
        self.circuit_threshold = max(1, int(circuit_threshold))
        self.circuit_open_seconds = max(1.0, float(circuit_open_seconds))
        self.retries_encountered = 0
        self.last_metrics = None
        self.last_error = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    @staticmethod
    def _bound_prompt(prompt: str, max_words: int = MAX_PROMPT_WORDS) -> str:
        words = str(prompt or "").split()
        if len(words) <= max_words:
            return str(prompt or "")
        bounded = " ".join(words[:max_words]).rstrip()
        return f"{bounded}\n[TRUNCATED_CONTEXT: prompt budget {max_words} words]"

    def _circuit_is_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def health_check(self, timeout_seconds: float = 2.0) -> bool:
        if self._circuit_is_open():
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(1.0, max(0.5, min(float(timeout_seconds), 5.0))), allow_redirects=False)
            response.raise_for_status()
            return not (300 <= response.status_code < 400)
        except requests.RequestException:
            self.last_error = "Ollama health check failed."
            return False

    def _record_failure(self, exc: BaseException) -> None:
        self.last_error = type(exc).__name__
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_open_seconds

    def _record_success(self) -> None:
        self.last_error = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _effective_timeout(self, minimum: float = 1.0) -> float:
        budget = remaining(self.request_timeout_seconds)
        if budget is None:
            return self.request_timeout_seconds
        if budget < minimum:
            raise RuntimeError("Generation latency budget exhausted.")
        return max(1.0, min(self.request_timeout_seconds, budget))

    def _payload(self, prompt: str, system_prompt: str | None, temperature: float, output_format: Any = None, num_predict: int | None = None) -> dict[str, Any]:
        bounded_prompt = self._bound_prompt(prompt)
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": max(0.0, min(float(temperature), 1.0)), "num_predict": max(16, min(int(num_predict or self.max_output_tokens), 4096))},
            "messages": [{"role": "user", "content": bounded_prompt}],
        }
        if output_format is not None:
            payload["format"] = output_format
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": str(system_prompt)})
        return payload

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self._effective_timeout()
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=(2.0, timeout), allow_redirects=False)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Ollama returned a non-object JSON response.")
            return data
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise RuntimeError("Generation service request failed.") from exc

    def _generate_raw(self, prompt: str, system_prompt: str | None, temperature: float, output_format: Any = None, num_predict: int | None = None) -> dict[str, Any]:
        payload = self._payload(prompt, system_prompt, temperature, output_format, num_predict)
        last_error: Exception | None = None
        for attempt in range(2):
            if exhausted():
                raise RuntimeError("Generation latency budget exhausted.")
            try:
                return self._post_chat(payload)
            except RuntimeError as exc:
                last_error = exc
                if attempt == 0:
                    budget = remaining()
                    if budget is None or budget >= 4.0:
                        self.retries_encountered += 1
                        time.sleep(min(0.25, max(0.0, (budget or 0.25) / 20.0)))
                        continue
                raise
        raise last_error or RuntimeError("Generation failed.")

    def _content(self, data: dict[str, Any]) -> str:
        self.last_metrics = {k: data.get(k) for k in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration", "total_duration") if k in data}
        message = data.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            self._record_success()
            return message["content"].strip()[:20000]
        error = RuntimeError("Ollama returned an invalid chat response.")
        self._record_failure(error)
        raise error

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        if self._circuit_is_open():
            raise RuntimeError("Ollama circuit breaker is open; generation was skipped.")
        try:
            data = self._generate_raw(prompt, system_prompt, temperature)
        except Exception as exc:
            self._record_failure(exc)
            raise
        return self._content(data)

    def generate_json(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 180) -> str:
        if self._circuit_is_open():
            raise RuntimeError("Ollama circuit breaker is open; generation was skipped.")
        try:
            data = self._generate_raw(prompt, system_prompt, temperature, output_format="json", num_predict=max_tokens)
        except Exception as exc:
            self._record_failure(exc)
            raise
        raw = self._content(data)
        try:
            json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._record_failure(RuntimeError("Invalid JSON"))
            raise RuntimeError("Ollama returned invalid JSON.") from exc
        return raw

    def generate_stream(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> Iterator[str]:
        if self._circuit_is_open():
            raise RuntimeError("Ollama circuit breaker is open; generation was skipped.")
        if exhausted():
            raise RuntimeError("Generation latency budget exhausted.")
        payload = self._payload(prompt, system_prompt, temperature, num_predict=self.max_output_tokens)
        payload["stream"] = True
        response = None
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=(2.0, self._effective_timeout()), allow_redirects=False, stream=True)
            response.raise_for_status()
            emitted = 0
            metrics: dict[str, Any] = {}
            for raw_line in response.iter_lines(decode_unicode=True):
                if exhausted():
                    raise RuntimeError("Generation latency budget exhausted during streaming.")
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                if not isinstance(data, dict):
                    continue
                message = data.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    token = message["content"]
                    if token:
                        emitted += len(token)
                        yield token
                    if emitted > 20000:
                        raise RuntimeError("Ollama streaming response exceeded the output limit.")
                for key in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration", "total_duration"):
                    if key in data:
                        metrics[key] = data[key]
                if data.get("done"):
                    self.last_metrics = metrics or None
                    self._record_success()
                    return
            raise RuntimeError("streaming response ended before the terminal done event")
        except Exception as exc:
            self._record_failure(exc)
            raise RuntimeError("Ollama streaming generation failed.") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
