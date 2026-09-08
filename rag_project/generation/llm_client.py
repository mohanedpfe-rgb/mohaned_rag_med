from __future__ import annotations

import logging
import time

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, stop_after_delay, wait_random_exponential

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
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception()
    _logger.warning("Retry attempt %d failed with: %s", attempt, exc)


class OllamaLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        max_output_tokens: int = 512,
        circuit_threshold: int = 2,
        circuit_open_seconds: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_output_tokens = max(32, int(max_output_tokens))
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
                timeout=(1.0, max(0.5, float(timeout_seconds))),
            )
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            self.last_error = str(exc)
            return False

    def _record_failure(self, exc: Exception) -> None:
        self.last_error = str(exc)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_open_seconds

    def _record_success(self) -> None:
        self.last_error = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    @retry(
        stop=stop_after_attempt(5) | stop_after_delay(45),
        wait=wait_random_exponential(min=0.5, max=8),
        retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
        before_sleep=_before_sleep_cb,
        reraise=True,
    )
    def _generate_raw(self, prompt: str, system_prompt: str | None, temperature: float) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": self.max_output_tokens},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": system_prompt})
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(5, self.timeout_seconds),
            )
            response.raise_for_status()
            data = response.json()
            return data
        except (requests.RequestException, ValueError, TypeError) as exc:
            self.last_error = str(exc)
            raise RuntimeError(f"Generation service failed for model {self.model!r}.") from exc

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        if self._circuit_is_open():
            raise RuntimeError("Ollama circuit breaker is open; generation was skipped.")
        try:
            data = self._generate_raw(prompt, system_prompt, temperature)
        except Exception as exc:
            self._record_failure(exc)
            raise
        self.last_metrics = {
            k: data.get(k)
            for k in ("prompt_eval_count", "eval_count", "eval_duration", "load_duration", "total_duration")
            if k in data
        }
        message = data.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            self._record_success()
            return message["content"].strip()
        error = RuntimeError(f"Ollama returned an invalid chat response for model {self.model!r}.")
        self._record_failure(error)
        raise error
