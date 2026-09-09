from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, stop_after_delay, wait_random_exponential

from rag_project.security import validate_ollama_url

try:
    from rag_project.utils.logger import build_logger

    _logger = build_logger("llm_client")
except Exception:
    _logger = logging.getLogger("llm_client")
    if not _logger.handlers:
        _logger.addHandler(logging.StreamHandler())
        _logger.setLevel(logging.INFO)


def _before_sleep_cb(retry_state):
    instance = retry_state.args[0]
    instance.retries_encountered += 1
    _logger.warning("Retry attempt %d failed", retry_state.attempt_number)


class OllamaLLMClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 180.0, max_output_tokens: int = 512, circuit_threshold: int = 2, circuit_open_seconds: float = 15.0):
        self.base_url = validate_ollama_url(base_url)
        self.model = str(model).strip()
        if not self.model or len(self.model) > 200 or any(ord(ch) < 32 for ch in self.model):
            raise ValueError("Invalid Ollama model identifier.")
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.request_timeout_seconds = min(self.timeout_seconds, 60.0)
        self.max_output_tokens = max(32, min(int(max_output_tokens), 4096))
        self.circuit_threshold = max(1, int(circuit_threshold))
        self.circuit_open_seconds = max(1.0, float(circuit_open_seconds))
        self.retries_encountered = 0
        self.last_metrics: dict[str, float] | None = None
        self.last_error: str | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _circuit_is_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def health_check(self, timeout_seconds: float = 2.0) -> bool:
        if self._circuit_is_open():
            return False
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(1.0, max(0.5, min(float(timeout_seconds), 5.0))),
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                self.last_error = "Ollama health endpoint returned an unexpected redirect."
                return False
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            self.last_error = "Ollama health check failed."
            return False

    def _record_failure(self, exc: Exception) -> None:
        self.last_error = type(exc).__name__
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_open_seconds

    def _record_success(self) -> None:
        self.last_error = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    @retry(stop=stop_after_attempt(2) | stop_after_delay(65), wait=wait_random_exponential(min=0.5, max=4), retry=retry_if_exception_type((requests.RequestException, RuntimeError)), before_sleep=_before_sleep_cb, reraise=True)
    def _generate_raw(self, prompt: str, system_prompt: str | None, temperature: float) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": max(0.0, min(float(temperature), 1.0)), "num_predict": self.max_output_tokens},
            "messages": [{"role": "user", "content": str(prompt)}],
        }
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": str(system_prompt)})
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(5, self.request_timeout_seconds),
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                raise RuntimeError("Ollama returned an unexpected redirect.")
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Ollama returned a non-object JSON response.")
            return data
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise RuntimeError("Generation service request failed.") from exc

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        if self._circuit_is_open():
            raise RuntimeError("Ollama circuit breaker is open; generation was skipped.")
        try:
            data = self._generate_raw(prompt, system_prompt, temperature)
        except Exception as exc:
            self._record_failure(exc)
            raise
        self.last_metrics = {k: data.get(k) for k in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration", "total_duration") if k in data}
        message = data.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            self._record_success()
            return message["content"].strip()[:20000]
        error = RuntimeError("Ollama returned an invalid chat response.")
        self._record_failure(error)
        raise error

    def generate_stream(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> Iterator[str]:
        """Yield Ollama chat tokens as they arrive over the HTTP stream.

        A streaming response is only considered successful after Ollama sends its
        terminal ``done`` event. If the connection ends early, the caller receives
        a failure instead of a silently truncated answer being marked healthy.
        """
        if self._circuit_is_open():
            raise RuntimeError("Ollama circuit breaker is open; generation was skipped.")

        payload = {
            "model": self.model,
            "stream": True,
            "options": {"temperature": max(0.0, min(float(temperature), 1.0)), "num_predict": self.max_output_tokens},
            "messages": [{"role": "user", "content": str(prompt)}],
        }
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": str(system_prompt)})

        response = None
        done = False
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(5, self.request_timeout_seconds),
                allow_redirects=False,
                stream=True,
            )
            if 300 <= response.status_code < 400:
                raise RuntimeError("Ollama returned an unexpected redirect.")
            response.raise_for_status()

            emitted = 0
            final_metrics: dict[str, float] = {}
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Ollama returned an invalid streaming event.") from exc
                if not isinstance(data, dict):
                    continue
                message = data.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    token = message["content"]
                    if token:
                        emitted += len(token)
                        if emitted > 20000:
                            raise RuntimeError("Ollama streaming response exceeded the output limit.")
                        yield token
                for key in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration", "total_duration"):
                    if key in data:
                        final_metrics[key] = data[key]
                if bool(data.get("done")):
                    done = True
                    self.last_metrics = final_metrics or None
                    self._record_success()
                    return

            if not done:
                self.last_metrics = final_metrics or None
                raise RuntimeError("Ollama streaming response ended before the terminal done event.")
        except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
            self._record_failure(exc)
            raise RuntimeError("Ollama streaming generation failed.") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
